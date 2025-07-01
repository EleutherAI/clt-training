from __future__ import annotations

from dataclasses import replace
from typing import List

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from .sparse_coder import MidDecoder, SparseCoder
from .utils import decoder_impl

__all__ = ["MatryoshkaRunner"]




"""Matryoshka‑style runner that *inherits all the cross‑layer coalescing tricks*
from ``CrossLayerRunner`` **and** fixes the slice logic so it works when the
base top‑k (cfg.k) is *smaller* than the Matryoshka slice sizes.

**Key change:** instead of carving contiguous blocks out of
``latent_acts/latent_indices`` we *mask* out latents whose indices are ≥ kᵢ.
This avoids empty tensors (and the associated ``min()`` error) when
``cfg.k`` ≪ kᵢ.
"""



class MatryoshkaRunner:  # noqa: D101
    # ---------------------------------------------------------------------
    # Construction & bookkeeping
    # ---------------------------------------------------------------------
    def __init__(self):
        self.outputs: Dict[str, MidDecoder] = {}
        self.to_restore: Dict[str, Tuple[MidDecoder, bool]] = {}

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _matryoshka_sizes(mid: MidDecoder) -> List[int]:
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
        """Single‑slice cross‑layer decode copied from ``CrossLayerRunner``."""
        # The body is identical to the original CrossLayerRunner.decode, so we
        # keep it verbatim to preserve behaviour.  For brevity, only the Δ from
        # upstream is commented.

        self.outputs[module_name] = mid_out

        candidate_indices = []
        candidate_values = []
        hookpoints = []
        layer_mids = []
        output = 0.0
        to_delete = set()
        out, hookpoint = None, None  # type: ignore[misc]

        for i, (hookpoint, layer_mid) in enumerate(self.outputs.items()):
            if detach_grad:
                layer_mid.detach()

            divide_by = (
                max(1, len(self.outputs) - 1)
                if layer_mid.sparse_coder.cfg.divide_cross_layer
                else 1
            )

            layer_mids.append(layer_mid)
            hookpoints.append(hookpoint)
            candidate_indices.append(layer_mid.latent_indices + i * layer_mid.sparse_coder.num_latents)
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
        """Matryoshka decoding with per-slice top-k selection.

        Instead of applying top-k once to the whole latent space and then masking,
        we apply top-k separately to each subset of the latent space for each slice.
        """
        print(f"\n{'='*60}")
        print(f"MatryoshkaRunner: Processing {module_name}")
        print(f"{'='*60}")
        
        k_values = sorted(self._matryoshka_sizes(mid_out))
        print(f"Matryoshka k-values: {k_values}")
        print(f"Original activations shape: {mid_out.latent_acts.shape}")
        print(f"Original indices shape: {mid_out.latent_indices.shape}")
        print(f"Original indices range: {mid_out.latent_indices.min().item():.0f} to {mid_out.latent_indices.max().item():.0f}")

        total_fvu = mid_out.latent_acts.new_tensor(0.0)
        total_aux = mid_out.latent_acts.new_tensor(0.0)
        total_multi = mid_out.latent_acts.new_tensor(0.0)
        slice_outputs = []

        print(f"\n{'='*40}")
        print(f"Processing {len(k_values)} Matryoshka slices (per-slice top-k approach):")
        print(f"{'='*40}")

        # Get the pre-activations from the original encoding
        # We need to re-encode to get access to the full pre-activations
        if mid_out.pre_acts is None:
            print("Warning: pre_acts is None, falling back to masking approach")
            # Fallback to the original masking approach if pre_acts is not available
            return self._decode_with_masking(mid_out, y, module_name, detach_grad, advance, **kwargs)
        
        pre_acts = mid_out.pre_acts
        print(f"Pre-activations shape: {pre_acts.shape}")
        print(f"Pre-activations range: {pre_acts.min().item():.6f} to {pre_acts.max().item():.6f}")

        for i, k_i in enumerate(k_values):
            print(f"\n--- Slice {i+1}/{len(k_values)}: k={k_i} ---")
            
            # --------------------------------------------------
            # Apply top-k to the subset of latent space for this slice
            # --------------------------------------------------
            # Create a mask for the subset of latent space for this slice
            subset_mask = torch.arange(pre_acts.shape[1], device=pre_acts.device) < k_i
            
            # Apply the mask to get the subset of pre-activations
            subset_pre_acts = pre_acts * subset_mask.to(pre_acts.dtype)
            
            print(f"  Subset mask shape: {subset_mask.shape}")
            print(f"  Subset mask sum (active features): {subset_mask.sum().item():.0f}")
            print(f"  Subset mask percentage: {subset_mask.float().mean().item()*100:.1f}%")
            print(f"  Subset pre-activations shape: {subset_pre_acts.shape}")
            print(f"  Subset pre-activations non-zero: {(subset_pre_acts != 0).sum().item():.0f}")

            # Apply top-k to the subset
            # Use the same k as the original encoding (cfg.k) but only within the subset
            subset_k = min(mid_out.sparse_coder.cfg.k, k_i)
            
            if isinstance(subset_pre_acts, torch.distributed.tensor.DTensor):
                # Handle distributed tensor case
                local_pre_acts = subset_pre_acts.to_local()
                local_values, local_indices = local_pre_acts.topk(subset_k, dim=1, sorted=False)
                # Adjust indices to account for the subset offset
                local_indices = local_indices + (local_indices < k_i).to(local_indices.dtype) * 0  # No offset needed since we masked
                values = torch.distributed.tensor.DTensor.from_local(
                    local_values,
                    subset_pre_acts.device_mesh,
                    (torch.distributed.tensor.Shard(0), torch.distributed.tensor.Replicate()),
                )
                indices = torch.distributed.tensor.DTensor.from_local(
                    local_indices,
                    subset_pre_acts.device_mesh,
                    (torch.distributed.tensor.Shard(0), torch.distributed.tensor.Replicate()),
                )
            else:
                # Handle regular tensor case
                values, indices = torch.topk(subset_pre_acts, subset_k, dim=1, sorted=False)
                # Adjust indices to account for the subset offset
                indices = indices + (indices < k_i).to(indices.dtype) * 0  # No offset needed since we masked

            print(f"  Top-k applied with k={subset_k}")
            print(f"  Selected values shape: {values.shape}")
            print(f"  Selected indices shape: {indices.shape}")
            print(f"  Selected indices range: {indices.min().item():.0f} to {indices.max().item():.0f}")
            print(f"  Selected values non-zero: {(values != 0).sum().item():.0f}")

            # Create a new MidDecoder with the slice-specific activations and indices
            sliced_mid = mid_out.copy(activations=values, indices=indices)
            
            # --------------------------------------------------
            # Decode this slice with full coalescing logic
            # --------------------------------------------------
            out_slice = self._decode_slice(
                sliced_mid,
                y,
                module_name,
                detach_grad=detach_grad,
                advance=(advance and i == len(k_values) - 1),
                **kwargs,
            )

            print(f"  Slice {i+1} results:")
            print(f"    FVU: {out_slice.fvu.item():.6f}")
            print(f"    AuxK: {out_slice.auxk_loss.item():.6f}")
            print(f"    Multi-TopK: {out_slice.multi_topk_fvu.item():.6f}")
            print(f"    Output shape: {out_slice.sae_out.shape}")

            slice_outputs.append(out_slice)
            total_fvu += out_slice.fvu
            total_aux += out_slice.auxk_loss
            total_multi += out_slice.multi_topk_fvu

        n_slices = len(k_values)
        avg_fvu = total_fvu / n_slices
        avg_aux = total_aux / n_slices
        avg_multi = total_multi / n_slices

        print(f"\n{'='*40}")
        print(f"Loss Aggregation Results:")
        print(f"{'='*40}")
        print(f"Number of slices: {n_slices}")
        print(f"Total FVU: {total_fvu.item():.6f}")
        print(f"Total AuxK: {total_aux.item():.6f}")
        print(f"Total Multi-TopK: {total_multi.item():.6f}")
        print(f"Average FVU: {avg_fvu.item():.6f}")
        print(f"Average AuxK: {avg_aux.item():.6f}")
        print(f"Average Multi-TopK: {avg_multi.item():.6f}")

        # ------------------------------------------------------------------
        # Build combined ``ForwardOutput`` using the last slice's object as a
        # template, but patch the losses to averaged versions.
        # ------------------------------------------------------------------
        main_out = slice_outputs[-1]
        final_output = replace(
            main_out,
            fvu=avg_fvu,
            auxk_loss=avg_aux,
            multi_topk_fvu=avg_multi,
        )
        
        print(f"\n{'='*40}")
        print(f"Final Output:")
        print(f"{'='*40}")
        print(f"Final activations shape: {final_output.latent_acts.shape}")
        print(f"Final indices shape: {final_output.latent_indices.shape}")
        print(f"Final sae_out shape: {final_output.sae_out.shape}")
        print(f"Final losses - FVU: {final_output.fvu.item():.6f}, AuxK: {final_output.auxk_loss.item():.6f}, Multi-TopK: {final_output.multi_topk_fvu.item():.6f}")
        print(f"{'='*60}\n")
        
        return final_output

    def _decode_with_masking(
        self,
        mid_out: MidDecoder,
        y: Tensor,
        module_name: str,
        *,
        detach_grad: bool = False,
        advance: bool = True,
        **kwargs,
    ):
        """Fallback method using the original masking approach."""
        print(f"\n{'='*60}")
        print(f"MatryoshkaRunner: Processing {module_name} (masking fallback)")
        print(f"{'='*60}")
        
        k_values = sorted(self._matryoshka_sizes(mid_out))
        print(f"Matryoshka k-values: {k_values}")
        print(f"Original activations shape: {mid_out.latent_acts.shape}")
        print(f"Original indices shape: {mid_out.latent_indices.shape}")
        print(f"Original indices range: {mid_out.latent_indices.min().item():.0f} to {mid_out.latent_indices.max().item():.0f}")

        total_fvu = mid_out.latent_acts.new_tensor(0.0)
        total_aux = mid_out.latent_acts.new_tensor(0.0)
        total_multi = mid_out.latent_acts.new_tensor(0.0)
        slice_outputs = []

        print(f"\n{'='*40}")
        print(f"Processing {len(k_values)} Matryoshka slices (masking approach):")
        print(f"{'='*40}")

        for i, k_i in enumerate(k_values):
            print(f"\n--- Slice {i+1}/{len(k_values)}: k={k_i} ---")
            
            # --------------------------------------------------
            # Mask activations outside the slice (idx ≥ kᵢ)
            # --------------------------------------------------
            mask = mid_out.latent_indices < k_i  # shape (B, cfg.k)
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask sum (active features): {mask.sum().item():.0f}")
            print(f"  Mask percentage: {mask.float().mean().item()*100:.1f}%")

            # Keep tensor shapes fixed so downstream code is happy
            acts_slice = mid_out.latent_acts * mask.to(mid_out.latent_acts.dtype)
            indices_slice = mid_out.latent_indices  # unchanged; activations outside slice are zero
            sliced_mid = mid_out.copy(activations=acts_slice, indices=indices_slice)
            
            print(f"  Masked activations shape: {acts_slice.shape}")
            print(f"  Masked activations non-zero: {(acts_slice != 0).sum().item():.0f}")
            print(f"  Indices shape: {indices_slice.shape} (unchanged)")

            # --------------------------------------------------
            # Decode this slice with full coalescing logic
            # --------------------------------------------------
            out_slice = self._decode_slice(
                sliced_mid,
                y,
                module_name,
                detach_grad=detach_grad,
                advance=(advance and i == len(k_values) - 1),
                **kwargs,
            )

            print(f"  Slice {i+1} results:")
            print(f"    FVU: {out_slice.fvu.item():.6f}")
            print(f"    AuxK: {out_slice.auxk_loss.item():.6f}")
            print(f"    Multi-TopK: {out_slice.multi_topk_fvu.item():.6f}")
            print(f"    Output shape: {out_slice.sae_out.shape}")

            slice_outputs.append(out_slice)
            total_fvu += out_slice.fvu
            total_aux += out_slice.auxk_loss
            total_multi += out_slice.multi_topk_fvu

        n_slices = len(k_values)
        avg_fvu = total_fvu / n_slices
        avg_aux = total_aux / n_slices
        avg_multi = total_multi / n_slices

        print(f"\n{'='*40}")
        print(f"Loss Aggregation Results:")
        print(f"{'='*40}")
        print(f"Number of slices: {n_slices}")
        print(f"Total FVU: {total_fvu.item():.6f}")
        print(f"Total AuxK: {total_aux.item():.6f}")
        print(f"Total Multi-TopK: {total_multi.item():.6f}")
        print(f"Average FVU: {avg_fvu.item():.6f}")
        print(f"Average AuxK: {avg_aux.item():.6f}")
        print(f"Average Multi-TopK: {avg_multi.item():.6f}")

        # ------------------------------------------------------------------
        # Build combined ``ForwardOutput`` using the last slice's object as a
        # template, but patch the losses to averaged versions.
        # ------------------------------------------------------------------
        main_out = slice_outputs[-1]
        final_output = replace(
            main_out,
            fvu=avg_fvu,
            auxk_loss=avg_aux,
            multi_topk_fvu=avg_multi,
        )
        
        print(f"\n{'='*40}")
        print(f"Final Output:")
        print(f"{'='*40}")
        print(f"Final activations shape: {final_output.latent_acts.shape}")
        print(f"Final indices shape: {final_output.latent_indices.shape}")
        print(f"Final sae_out shape: {final_output.sae_out.shape}")
        print(f"Final losses - FVU: {final_output.fvu.item():.6f}, AuxK: {final_output.auxk_loss.item():.6f}, Multi-TopK: {final_output.multi_topk_fvu.item():.6f}")
        print(f"{'='*60}\n")
        
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
        for restorable, was_last in self.to_restore.values():
            if was_last:
                restorable.restore(True)
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
        print(f"\n{'='*60}")
        print(f"CrossLayerRunner: Processing {module_name}")
        print(f"{'='*60}")
        
        self.outputs[module_name] = mid_out

        candidate_indices = []
        candidate_values = []
        hookpoints = []
        layer_mids = []
        output = 0
        to_delete = set()
        out, hookpoint = None, None
        
        print(f"Number of layers in outputs: {len(self.outputs)}")
        print(f"Current layer activations shape: {mid_out.latent_acts.shape}")
        print(f"Current layer indices shape: {mid_out.latent_indices.shape}")
        print(f"Current layer indices range: {mid_out.latent_indices.min().item():.0f} to {mid_out.latent_indices.max().item():.0f}")
        print(f"Cross-layer coalescing enabled: {mid_out.sparse_coder.cfg.do_coalesce_topk}")
        
        for i, (hookpoint, layer_mid) in enumerate(self.outputs.items()):
            print(f"\n--- Processing layer {i+1}/{len(self.outputs)}: {hookpoint} ---")
            
            if detach_grad:
                layer_mid.detach()
                print(f"  Detached gradients for layer {hookpoint}")
                
            if layer_mid.sparse_coder.cfg.divide_cross_layer:
                divide_by = max(1, len(self.outputs) - 1)
            else:
                divide_by = 1
            print(f"  Divide by factor: {divide_by}")
            
            layer_mids.append(layer_mid)
            hookpoints.append(hookpoint)
            candidate_indices.append(
                layer_mid.latent_indices + i * layer_mid.sparse_coder.num_latents
            )
            candidate_values.append(layer_mid.current_latent_acts)
            
            print(f"  Layer {hookpoint} info:")
            print(f"    Original indices shape: {layer_mid.latent_indices.shape}")
            print(f"    Original indices range: {layer_mid.latent_indices.min().item():.0f} to {layer_mid.latent_indices.max().item():.0f}")
            print(f"    Offset indices shape: {layer_indices.shape}")
            print(f"    Offset indices range: {layer_indices.min().item():.0f} to {layer_indices.max().item():.0f}")
            print(f"    Activations shape: {layer_mid.current_latent_acts.shape}")
            print(f"    Activations non-zero: {(layer_mid.current_latent_acts != 0).sum().item():.0f}")
            
            if detach_grad and advance:
                self.to_restore[hookpoint] = (layer_mid, layer_mid.will_be_last)
            if layer_mid.will_be_last:
                to_delete.add(hookpoint)
                print(f"  Layer {hookpoint} will be deleted after processing")
                
            if not mid_out.sparse_coder.cfg.do_coalesce_topk:
                print(f"  Processing layer {hookpoint} without coalescing")
                out = layer_mid(
                    y,
                    addition=(0 if hookpoint != module_name else (output / divide_by)),
                    no_extras=hookpoint != module_name,
                    denormalize=hookpoint == module_name,
                    **kwargs,
                )
                if hookpoint != module_name:
                    output += out.sae_out
                    print(f"  Added layer output to running sum, current sum shape: {output.shape}")
                else:
                    print(f"  This is the target layer, using output directly")
            else:
                layer_mid.next()
                print(f"  Advanced layer {hookpoint} index to {layer_mid.index}")

        if mid_out.sparse_coder.cfg.do_coalesce_topk:
            print(f"\n{'='*40}")
            print(f"Cross-layer coalescing processing:")
            print(f"{'='*40}")
            
            candidate_indices = torch.cat(candidate_indices, dim=1)
            candidate_values = torch.cat(candidate_values, dim=1)
            
            print(f"Concatenated indices shape: {candidate_indices.shape}")
            print(f"Concatenated indices range: {candidate_indices.min().item():.0f} to {candidate_indices.max().item():.0f}")
            print(f"Concatenated values shape: {candidate_values.shape}")
            print(f"Concatenated values non-zero: {(candidate_values != 0).sum().item():.0f}")
            
            if mid_out.sparse_coder.cfg.topk_coalesced:
                print(f"Applying top-k coalescing with k={mid_out.sparse_coder.cfg.k}")
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
                
                print(f"Top-k coalesced values shape: {best_values.shape}")
                print(f"Top-k coalesced indices shape: {best_indices.shape}")
                print(f"Top-k coalesced indices range: {best_indices.min().item():.0f} to {best_indices.max().item():.0f}")
            else:
                best_values = candidate_values
                best_indices = candidate_indices
                print(f"No top-k coalescing applied, using all concatenated values")
            
            print(f"Coalesce mode: {mid_out.sparse_coder.cfg.coalesce_topk}")
            
            if mid_out.sparse_coder.cfg.coalesce_topk == "concat":
                print(f"Using concat mode - applying modulo to indices")
                best_indices = best_indices % mid_out.sparse_coder.num_latents
                print(f"Modulo indices shape: {best_indices.shape}")
                print(f"Modulo indices range: {best_indices.min().item():.0f} to {best_indices.max().item():.0f}")
                
                new_mid_out = mid_out.copy(
                    indices=best_indices,
                    activations=best_values,
                )
                print(f"Created new MidDecoder with coalesced activations")
                out = new_mid_out(y, index=0, add_post_enc=False, **kwargs)
                if advance:
                    del mid_out.x
                    
            elif mid_out.sparse_coder.cfg.coalesce_topk == "per-layer":
                print(f"Using per-layer mode - processing each layer separately")
                output = 0
                for i, layer_mid in enumerate(layer_mids):
                    hookpoint = hookpoints[i]
                    is_ours = hookpoint == module_name
                    if not is_ours:
                        continue
                        
                    print(f"  Processing layer {hookpoint} (is_ours={is_ours})")
                    num_latents = layer_mid.sparse_coder.num_latents
                    print(f"    Layer num_latents: {num_latents}")
                    
                    if is_ours:
                        best_indices_local = best_indices
                        best_values_local = best_values
                        print(f"    Using global best indices/values for target layer")
                    else:
                        best_indices_local = None
                        best_values_local = None
                        print(f"    Using None for non-target layer")
                        
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
                        print(f"    Added to running sum, current sum shape: {output.shape}")
                    else:
                        print(f"    This is target layer, using output directly")
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
                        print(f"    Applied per-layer index masking")
            else:
                raise ValueError("Not implemented")

        # last output guaranteed to be the current layer
        assert hookpoint == module_name

        if not advance:
            for layer_mid in layer_mids:
                layer_mid.prev()
            print(f"Reversed layer indices (not advancing)")

        if advance:
            for hookpoint in to_delete:
                del self.outputs[hookpoint]
            print(f"Deleted {len(to_delete)} layers from outputs")

        print(f"\n{'='*40}")
        print(f"Final Output:")
        print(f"{'='*40}")
        print(f"Final activations shape: {out.latent_acts.shape}")
        print(f"Final indices shape: {out.latent_indices.shape}")
        print(f"Final sae_out shape: {out.sae_out.shape}")
        print(f"Final losses - FVU: {out.fvu.item():.6f}, AuxK: {out.auxk_loss.item():.6f}, Multi-TopK: {out.multi_topk_fvu.item():.6f}")
        print(f"{'='*60}\n")

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
