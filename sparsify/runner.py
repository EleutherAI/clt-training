from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from .sparse_coder import MidDecoder, SparseCoder, MatryoshkaEncoderOutput
from .utils import decoder_impl




"""Matryoshka-style runner that applies top-k selection separately to different slices of the latent space.

**How it works:**
1. For each slice size k_i, computes top-k selection on the corresponding subset of the latent space
2. Computes all losses (FVU, AuxK, Multi-TopK) for each slice independently  
3. Aggregates losses across all slices (summed, not averaged)
4. Uses the largest slice for final output and cross-layer coalescing (if enabled)

**Performance Benefits:**
- No need to materialize pre_acts in the runner (computed only when needed in SparseCoder)
- Eliminates complex gradient detachment/restoration dance
- Only does expensive cross-layer coalescing for the largest slice
- Much faster than the original masking approach

**Cross-layer Coalescing:**
- Inherits all cross-layer coalescing capabilities from CrossLayerRunner
- Only applies coalescing to the largest slice to maintain performance
- Simplified gradient management since we're just combining pre-computed results

Paper:
https://www.lesswrong.com/posts/rKM9b6B2LqwSB5ToN/learning-multi-level-features-with-matryoshka-saes

"""



class MatryoshkaRunner: 
    """Matryoshka-style runner that processes different slices of the latent space independently.
    
    This runner applies top-k selection separately to different subsets of the latent space,
    computes losses for each slice, and aggregates them. It's designed for scenarios where
    you want to analyze different granularities of feature selection simultaneously.
    """
    
    def __init__(self):
        """Initialize the MatryoshkaRunner with empty state tracking.
        
        Attributes:
            outputs: Maps module names to their MidDecoder objects for cross-layer coalescing
            to_restore: Tracks MidDecoder objects that need gradient restoration (rarely used now)
        """
        self.outputs: Dict[str, MidDecoder] = {}
        self.to_restore: Dict[str, Tuple[MidDecoder, bool]] = {}

  
    @staticmethod
    def _matryoshka_sizes(mid: MidDecoder) -> List[int]:
        """Determine the k-values for each Matryoshka slice based on configuration.
        
        This method reads the configuration to determine how many slices to create
        and what size each slice should be. It supports three configuration modes:
        
        Args:
            mid: MidDecoder containing the sparse coder configuration
            
        Returns:
            List of k-values, one for each slice, sorted in ascending order
            
        Configuration priority:
        1. matryoshka_k_values: Direct list of k-values [k1, k2, k3, ...]
        2. matryoshka_expansion_factors: Multipliers applied to input dimension [1.0, 2.0, 4.0, ...]
        3. Default: Single slice with the base k value from configuration
        """
        cfg = mid.sparse_coder.cfg
        if cfg.matryoshka_k_values:
            return cfg.matryoshka_k_values
        if cfg.matryoshka_expansion_factors:
            d_in = mid.sparse_coder.d_in
            return [int(d_in * ef) for ef in cfg.matryoshka_expansion_factors]
        return [cfg.k]

    # ------------------------------------------------------------------
    # Cross‑layer decode helper (verbatim CrossLayerRunner.decode)
    # ------------------------------------------------------------------
    def _decode_slice(
        self,
        mid_out: MidDecoder,
        y: Tensor,
        module_name: str,
        *,
        detach_grad: bool,
        advance: bool,
        **kwargs,
    ):
        """Perform cross-layer coalescing and decoding for a single slice.
        
        This method handles the cross-layer coalescing logic inherited from CrossLayerRunner.
        It combines activations and indices from multiple layers, applies top-k selection
        if enabled, and produces the final decoded output.
        
        The method is simplified compared to the original CrossLayerRunner because:
        - We're working with pre-computed results from SparseCoder
        - No complex gradient detachment/restoration is needed
        - We're just coalescing already-computed slice results
        
        Args:
            mid_out: MidDecoder for the current layer/slice
            y: Target tensor for reconstruction
            module_name: Name of the current module being processed
            detach_grad: Whether to detach gradients (largely unused in this implementation)
            advance: Whether to advance layer indices after processing
            **kwargs: Additional arguments passed to MidDecoder calls
            
        Returns:
            ForwardOutput containing the decoded result and loss information
        """
        # Store the current layer's MidDecoder for cross-layer processing
        self.outputs[module_name] = mid_out

        candidate_indices = []
        candidate_values = []
        hookpoints = []
        layer_mids = []
        output = 0.0
        to_delete = set()
        out, hookpoint = None, None  # type: ignore[misc]

        for i, (hookpoint, layer_mid) in enumerate(self.outputs.items()):
            # Calculate division factor for cross-layer normalization
            # If divide_cross_layer is enabled, divide by (num_layers - 1)
            divide_by = (
                max(1, len(self.outputs) - 1)
                if layer_mid.sparse_coder.cfg.divide_cross_layer
                else 1
            )

            layer_mids.append(layer_mid)
            hookpoints.append(hookpoint)
            offset_indices = layer_mid.latent_indices + i * layer_mid.sparse_coder.num_latents
            candidate_indices.append(offset_indices)
            candidate_values.append(layer_mid.current_latent_acts)

            
            if layer_mid.will_be_last:
                to_delete.add(hookpoint)

            if not mid_out.sparse_coder.cfg.do_coalesce_topk:
                out = layer_mid(
                    y,
                    addition=(0 if hookpoint != module_name else (output / divide_by)),
                    no_extras=hookpoint != module_name,
                    denormalize=hookpoint == module_name,
                    **kwargs,
                )
                if hookpoint != module_name:
                    output += out.sae_out  # type: ignore[operator]
            else:
                layer_mid.next()

        if mid_out.sparse_coder.cfg.do_coalesce_topk:
            candidate_indices = torch.cat(candidate_indices, dim=1)
            candidate_values = torch.cat(candidate_values, dim=1)

            if mid_out.sparse_coder.cfg.topk_coalesced:
                if isinstance(candidate_values, DTensor):

                    def mapper(values, indices):
                        best_vals, best_idxs = torch.topk(values, k=mid_out.sparse_coder.cfg.k, dim=1)
                        best_idxs = torch.gather(indices, 1, best_idxs)
                        return best_vals, best_idxs

                    best_values, best_indices = local_map(
                        mapper,
                        out_placements=(candidate_values.placements, candidate_indices.placements),
                    )(candidate_values, candidate_indices)
                else:
                    best_values, best_indices = torch.topk(candidate_values, k=mid_out.sparse_coder.cfg.k, dim=1)
                    best_indices = torch.gather(candidate_indices, 1, best_indices)
            else:
                best_values = candidate_values
                best_indices = candidate_indices

            if mid_out.sparse_coder.cfg.coalesce_topk == "concat":
                best_indices = best_indices % mid_out.sparse_coder.num_latents
                new_mid_out = mid_out.copy(indices=best_indices, activations=best_values)
                out = new_mid_out(y, index=0, add_post_enc=False, **kwargs)
                if advance:
                    del mid_out.x  # type: ignore[attr-defined]
            elif mid_out.sparse_coder.cfg.coalesce_topk == "per-layer":
                output = 0.0
                for i, layer_mid in enumerate(layer_mids):
                    hookpoint = hookpoints[i]
                    is_ours = hookpoint == module_name
                    if not is_ours:
                        continue
                    num_latents = layer_mid.sparse_coder.num_latents
                    if is_ours:
                        best_indices_local = best_indices
                        best_values_local = best_values
                    else:
                        best_indices_local = None
                        best_values_local = None
                    new_mid_out = layer_mid.copy(indices=best_indices_local, activations=best_values_local)
                    out = new_mid_out(
                        y,
                        layer_mid.index - 1,
                        add_post_enc=False,
                        addition=(0 if hookpoint != module_name else output),
                        no_extras=hookpoint != module_name,
                        denormalize=hookpoint == module_name,
                        **kwargs,
                    )
                    if hookpoint != module_name:
                        output += out.sae_out  # type: ignore[operator]
                    else:
                        if isinstance(out.latent_indices, DTensor):
                            out = replace(
                                out,
                                latent_indices=local_map(
                                    lambda x: (x % num_latents) * (x // num_latents == i),
                                    out_placements=(out.latent_indices.placements,),
                                )(out.latent_indices),
                            )
                        else:
                            out = replace(
                                out,
                                latent_indices=(out.latent_indices % num_latents) * (out.latent_indices // num_latents == i),
                            )
            else:
                raise ValueError("Unknown coalesce_topk mode: " + mid_out.sparse_coder.cfg.coalesce_topk)

        assert hookpoint == module_name  # type: ignore[has-type]

        if not advance:
            for layer_mid in layer_mids:
                layer_mid.prev()
        if advance:
            for hookpoint in to_delete:
                del self.outputs[hookpoint]
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public encode / decode
    # ------------------------------------------------------------------
    def encode(self, x: Tensor, sparse_coder: SparseCoder, **kwargs):
        return sparse_coder(x=x, y=None, return_mid_decoder=True, **kwargs)

    def decode(
        self,
        mid_out: MidDecoder,
        y: Tensor,
        module_name: str,
        *,
        detach_grad: bool = False,
        advance: bool = True,
        **kwargs,
    ):
        """Main Matryoshka decoding method that processes multiple slices independently.
        
        This is the core method that implements the Matryoshka approach. Instead of
        applying top-k once to the whole latent space and then masking, it computes
        slice-specific top-k results directly in the SparseCoder for each slice size.
        
        The method:
        1. Determines slice sizes using _matryoshka_sizes()
        2. Calls SparseCoder.encode_matryoshka_slices() to compute all slice results
        3. Aggregates losses from all slices (summed, not averaged)
        4. Uses the largest slice for final output and cross-layer coalescing
        5. Provides detailed debug information about the process
        
        Args:
            mid_out: MidDecoder containing the encoded input and sparse coder
            y: Target tensor for reconstruction
            module_name: Name of the current module being processed
            detach_grad: Whether to detach gradients (largely unused in this implementation)
            advance: Whether to advance layer indices after processing
            **kwargs: Additional arguments passed to MidDecoder calls
            
        Returns:
            ForwardOutput containing the decoded result and aggregated loss information
        """
        import time
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"MatryoshkaRunner: Processing {module_name}")
        print(f"{'='*60}")
        
        k_values = sorted(self._matryoshka_sizes(mid_out))

        # Use the optimized method that computes slice results directly in SparseCoder
        # This avoids materializing pre_acts in the runner and is much more efficient
        matryoshka_output = mid_out.sparse_coder.encode_matryoshka_slices(
            x=mid_out.x,
            k_values=k_values,
            y=y,
            dead_mask=mid_out.dead_mask,
            loss_mask=kwargs.get('loss_mask'),
        )
        
        # Aggregate losses from all slices by summing them
        # This gives us the total loss across all granularities
        total_fvu = sum(slice_result.fvu for slice_result in matryoshka_output.slice_results)
        total_aux = sum(slice_result.auxk_loss for slice_result in matryoshka_output.slice_results)
        total_multi = sum(slice_result.multi_topk_fvu for slice_result in matryoshka_output.slice_results)
        
        # Get the largest slice result for the final output
        largest_slice = matryoshka_output.slice_results[-1]  # Last slice has largest k
        
        # Create MidDecoder for the largest slice (for cross-layer coalescing if needed)
        largest_mid = MidDecoder(
            mid_out.sparse_coder,
            matryoshka_output.x,
            largest_slice.top_acts,
            largest_slice.top_indices,
            matryoshka_output.dead_mask,
            None,  # pre_acts not needed for cross-layer coalescing
        )
        
        # Only do cross-layer coalescing for the largest slice if needed
        if mid_out.sparse_coder.cfg.do_coalesce_topk:
            # Store the largest slice's MidDecoder for cross-layer processing
            self.outputs[module_name] = largest_mid
            
            # Do full cross-layer coalescing and decoding
            final_output = self._decode_slice(
                largest_mid,
                y,
                module_name,
                detach_grad=detach_grad,
                advance=advance,
                **kwargs,
            )
        else:
            # No cross-layer coalescing needed, use the largest slice directly
            final_output = largest_mid(
                y,
                index=0,
                add_post_enc=False,
                **kwargs,
            )
        
        # Update the final output with aggregated losses
        final_output = replace(
            final_output,
            fvu=total_fvu,
            auxk_loss=total_aux,
            multi_topk_fvu=total_multi,
        )
        
        n_slices = len(k_values)
        avg_fvu = total_fvu / n_slices
        avg_aux = total_aux / n_slices
        avg_multi = total_multi / n_slices
        
        # Only show debug info for the final slice
        print(f"\n{'='*60}")
        print(f"MatryoshkaRunner: {module_name} - Final Results")
        print(f"{'='*60}")
        print(f"Matryoshka k-values: {k_values}")
        print(f"Number of slices: {n_slices}")
        print(f"Final activations shape: {final_output.latent_acts.shape}")
        print(f"Final indices shape: {final_output.latent_indices.shape}")
        print(f"Final sae_out shape: {final_output.sae_out.shape}")
        
        # Activation statistics
        final_acts = final_output.latent_acts
        non_zero_count = (final_acts != 0).sum().item()
        total_elements = final_acts.numel()
        zero_count = total_elements - non_zero_count
        sparsity = (zero_count / total_elements) * 100
        
        print(f"Activation Statistics:")
        print(f"  Non-zero activations: {non_zero_count:,} / {total_elements:,} ({100-sparsity:.1f}%)")
        print(f"  Zero activations: {zero_count:,} / {total_elements:,} ({sparsity:.1f}%)")
        print(f"  Activation range: {final_acts.min().item():.6f} to {final_acts.max().item():.6f}")
        
        # Loss information
        print(f"Loss Results:")
        print(f"  Total FVU: {final_output.fvu.item():.6f}")
        print(f"  Total AuxK: {final_output.auxk_loss.item():.6f}")
        print(f"  Total Multi-TopK: {final_output.multi_topk_fvu.item():.6f}")
        print(f"  Average FVU: {avg_fvu.item():.6f}")
        print(f"  Average AuxK: {avg_aux.item():.6f}")
        print(f"  Average Multi-TopK: {avg_multi.item():.6f}")
        
        total_time = time.time() - start_time
        print(f"MatryoshkaRunner time: {total_time:.4f}s")
        print(f"{'='*60}")
        
        return final_output

    # ------------------------------------------------------------------
    # Wrapper call & housekeeping
    # ------------------------------------------------------------------
    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        sparse_coder: SparseCoder,
        module_name: str,
        detach_grad: bool = False,
        dead_mask: Tensor | None = None,
        loss_mask: Tensor | None = None,
        *,
        encoder_kwargs: Dict | None = None,
        decoder_kwargs: Dict | None = None,
    ):
        encoder_kwargs = encoder_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}
        mid_out = self.encode(x, sparse_coder, dead_mask=dead_mask, **encoder_kwargs)
        return self.decode(
            mid_out,
            y,
            module_name,
            detach_grad=detach_grad,
            loss_mask=loss_mask,
            **decoder_kwargs,
        )

    def restore(self):
        # In our new optimized approach, we typically don't need to restore anything
        # since slice computation happens in SparseCoder and we only create one MidDecoder
        if not self.to_restore:
            return
            
        #print(f"MatryoshkaRunner.restore(): Restoring {len(self.to_restore)} objects")
        for hookpoint, (restorable, was_last) in self.to_restore.items():
            #print(f"  Restoring {hookpoint}: was_last={was_last}")
            if was_last:
                #print(f"    Calling restore(True) on {hookpoint}")
                try:
                    restorable.restore(True)
                    #print(f"    Successfully restored {hookpoint}")
                except Exception as e:
                    print(f"    Error restoring {hookpoint}: {e}")
                    # Continue with other restorations
        self.to_restore.clear()

    def reset(self):
        self.outputs.clear()
        self.to_restore.clear()


class CrossLayerRunner(object):
    def __init__(self):
        self.outputs = {}
        self.to_restore = {}

    def encode(self, x: Tensor, sparse_coder: SparseCoder, **kwargs):
        out_mid = sparse_coder(
            x=x,
            y=None,
            return_mid_decoder=True,
            **kwargs,
        )
        return out_mid

    def decode(
        self,
        mid_out: MidDecoder,
        y: Tensor,
        module_name: str,
        detach_grad: bool = False,
        advance: bool = True,
        **kwargs,
    ):
       
        
        self.outputs[module_name] = mid_out

        candidate_indices = []
        candidate_values = []
        hookpoints = []
        layer_mids = []
        output = 0
        to_delete = set()
        out, hookpoint = None, None
        
       
        
        for i, (hookpoint, layer_mid) in enumerate(self.outputs.items()):
            
            
            if detach_grad:
                layer_mid.detach()
             
                
            if layer_mid.sparse_coder.cfg.divide_cross_layer:
                divide_by = max(1, len(self.outputs) - 1)
            else:
                divide_by = 1
          
            
            layer_mids.append(layer_mid)
            hookpoints.append(hookpoint)
            offset_indices = layer_mid.latent_indices + i * layer_mid.sparse_coder.num_latents
            candidate_indices.append(offset_indices)
            candidate_values.append(layer_mid.current_latent_acts)
            

            
            if detach_grad and advance:
                self.to_restore[hookpoint] = (layer_mid, layer_mid.will_be_last)
            if layer_mid.will_be_last:
                to_delete.add(hookpoint)
           
                
            if not mid_out.sparse_coder.cfg.do_coalesce_topk:
                
                out = layer_mid(
                    y,
                    addition=(0 if hookpoint != module_name else (output / divide_by)),
                    no_extras=hookpoint != module_name,
                    denormalize=hookpoint == module_name,
                    **kwargs,
                )
                if hookpoint != module_name:
                    output += out.sae_out
           
             
            else:
                layer_mid.next()
        if mid_out.sparse_coder.cfg.do_coalesce_topk:
         
            
            candidate_indices = torch.cat(candidate_indices, dim=1)
            candidate_values = torch.cat(candidate_values, dim=1)
            
          
            
            if mid_out.sparse_coder.cfg.topk_coalesced:
                
                if isinstance(candidate_values, DTensor):

                    def mapper(candidate_values, candidate_indices):
                        best_values, best_indices = torch.topk(
                            candidate_values, k=mid_out.sparse_coder.cfg.k, dim=1
                        )
                        best_indices = torch.gather(candidate_indices, 1, best_indices)
                        return best_values, best_indices

                    best_values, best_indices = local_map(
                        mapper,
                        out_placements=(
                            candidate_values.placements,
                            candidate_indices.placements,
                        ),
                    )(candidate_values, candidate_indices)
                else:
                    best_values, best_indices = torch.topk(
                        candidate_values, k=mid_out.sparse_coder.cfg.k, dim=1
                    )
                    best_indices = torch.gather(candidate_indices, 1, best_indices)
                
                
            else:
                best_values = candidate_values
                best_indices = candidate_indices
    
            

            
            if mid_out.sparse_coder.cfg.coalesce_topk == "concat":
  
                best_indices = best_indices % mid_out.sparse_coder.num_latents
      
                
                new_mid_out = mid_out.copy(
                    indices=best_indices,
                    activations=best_values,
                )

                out = new_mid_out(y, index=0, add_post_enc=False, **kwargs)
                if advance:
                    del mid_out.x
                    
            elif mid_out.sparse_coder.cfg.coalesce_topk == "per-layer":
         
                output = 0
                for i, layer_mid in enumerate(layer_mids):
                    hookpoint = hookpoints[i]
                    is_ours = hookpoint == module_name
                    if not is_ours:
                        continue
                        
  
                    num_latents = layer_mid.sparse_coder.num_latents
  
                    
                    if is_ours:
                        best_indices_local = best_indices
                        best_values_local = best_values
                        
                    else:
                        best_indices_local = None
                        best_values_local = None
        
                        
                    new_mid_out = layer_mid.copy(
                        indices=best_indices_local,
                        activations=best_values_local,
                    )
                    out = new_mid_out(
                        y,
                        layer_mid.index - 1,
                        add_post_enc=False,
                        addition=(0 if hookpoint != module_name else output),
                        no_extras=hookpoint != module_name,
                        denormalize=hookpoint == module_name,
                        **kwargs,
                    )
                    if hookpoint != module_name:
                        output += out.sae_out
                        
                    else:
                        
                        if isinstance(out.latent_indices, DTensor):
                            out = replace(
                                out,
                                latent_indices=local_map(
                                    lambda x: (x % num_latents)
                                    * (x // num_latents == i),
                                    out_placements=(out.latent_indices.placements,),
                                )(out.latent_indices),
                            )
                        else:
                            out = replace(
                                out,
                                latent_indices=(out.latent_indices % num_latents)
                                * (out.latent_indices // num_latents == i),
                            )
                      
            else:
                raise ValueError("Not implemented")

        # last output guaranteed to be the current layer
        assert hookpoint == module_name

        if not advance:
            for layer_mid in layer_mids:
                layer_mid.prev()
            

        if advance:
            for hookpoint in to_delete:
                del self.outputs[hookpoint]
           



        return out

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        sparse_coder: SparseCoder,
        module_name: str,
        detach_grad: bool = False,
        dead_mask: Tensor | None = None,
        loss_mask: Tensor | None = None,
        *,
        encoder_kwargs: dict = {},
        decoder_kwargs: dict = {},
    ):
        mid_out = self.encode(x, sparse_coder, dead_mask=dead_mask, **encoder_kwargs)
        return self.decode(
            mid_out, y, module_name, detach_grad, loss_mask=loss_mask, **decoder_kwargs
        )

    def restore(self):
        for restorable, was_last in self.to_restore.values():
            if was_last:
                restorable.restore(True)
        self.to_restore.clear()

    def reset(self):
        self.outputs.clear()
        self.to_restore.clear()
