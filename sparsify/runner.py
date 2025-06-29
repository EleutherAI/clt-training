from dataclasses import replace

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from .sparse_coder import MidDecoder, SparseCoder


class MatryoshkaRunner:
    """
    Runner for Matryoshka-style transcoder training.
    Performs multiple forward passes with different encoder/decoder slices
    and combines the losses.
    """
    
    def __init__(self):
        self.outputs = {}
        self.to_restore = {}
    
    def encode(self, x: Tensor, sparse_coder: SparseCoder, **kwargs):
        """Encode using the full encoder and return MidDecoder."""
        out_mid = sparse_coder(
            x=x,
            y=None,
            return_mid_decoder=True,
            **kwargs,
        )
        return out_mid
    
    def decode_slice(
        self,
        mid_out: MidDecoder,
        y: Tensor,
        module_name: str,
        k_start: int,
        k_end: int,
        detach_grad: bool = False,
        **kwargs,
    ):
        """Decode using a slice of the encoder/decoder."""
        # Create a copy of the MidDecoder with sliced activations and indices
        sliced_activations = mid_out.latent_acts[:, k_start:k_end]
        sliced_indices = mid_out.latent_indices[:, k_start:k_end]
        
        # Adjust indices to be relative to the slice
        if k_start > 0:
            sliced_indices = sliced_indices - k_start
        
        # Create a temporary MidDecoder for this slice
        sliced_mid = mid_out.copy(
            activations=sliced_activations,
            indices=sliced_indices,
        )
        
        # Temporarily modify the sparse coder to use sliced decoder
        original_W_dec = sliced_mid.sparse_coder.W_dec
        original_b_dec = sliced_mid.sparse_coder.b_dec
        
        # Store original encoder bias for restoration
        original_b_enc = sliced_mid.sparse_coder.encoder.bias
        
        # Use slices of the decoder weights and bias
        if hasattr(sliced_mid.sparse_coder, "W_decs"):
            sliced_mid.sparse_coder.W_decs[0] = original_W_dec[k_start:k_end, :]
            sliced_mid.sparse_coder.b_decs[0] = original_b_dec  # Keep decoder bias shared globally
        else:
            sliced_mid.sparse_coder.W_dec = original_W_dec[k_start:k_end, :]
            sliced_mid.sparse_coder.b_dec = original_b_dec  # Keep decoder bias shared globally
        
        print(f"Matryoshka: Decoder bias kept shared globally: {original_b_dec.shape}")
        
        # Slice the encoder bias: b_enc[k_start:k_end]
        if original_b_enc is not None:
            sliced_mid.sparse_coder.encoder.bias = original_b_enc[k_start:k_end]
            print(f"Matryoshka: Sliced encoder bias from {original_b_enc.shape} to {sliced_mid.sparse_coder.encoder.bias.shape} (k_start={k_start}, k_end={k_end})")
        else:
            print(f"Matryoshka: No encoder bias to slice (k_start={k_start}, k_end={k_end})")
        
        # Perform decoding
        if detach_grad:
            sliced_mid.detach()
        
        out = sliced_mid(
            y,
            addition=0,
            no_extras=False,
            denormalize=True,
            **kwargs,
        )
        
        # Restore original decoder weights and encoder bias
        if hasattr(sliced_mid.sparse_coder, "W_decs"):
            sliced_mid.sparse_coder.W_decs[0] = original_W_dec
            sliced_mid.sparse_coder.b_decs[0] = original_b_dec
        else:
            sliced_mid.sparse_coder.W_dec = original_W_dec
            sliced_mid.sparse_coder.b_dec = original_b_dec
        
        # Restore original encoder bias
        if original_b_enc is not None:
            sliced_mid.sparse_coder.encoder.bias = original_b_enc
        
        return out
    
    def decode(
        self,
        mid_out: MidDecoder,
        y: Tensor,
        module_name: str,
        detach_grad: bool = False,
        advance: bool = True,
        **kwargs,
    ):
        """Decode using multiple slices and combine losses."""
        self.outputs[module_name] = mid_out
        
        # Get Matryoshka sizes
        if mid_out.sparse_coder.cfg.matroshka_k_values:
            matryoshka_sizes = mid_out.sparse_coder.cfg.matroshka_k_values
            print(f"Matryoshka: Using k_values: {matryoshka_sizes}")
        elif mid_out.sparse_coder.cfg.matroshka_expansion_factors:
            # Extrapolate k values from expansion factors
            d_in = mid_out.sparse_coder.d_in
            matryoshka_sizes = [d_in * ef for ef in mid_out.sparse_coder.cfg.matroshka_expansion_factors]
            print(f"Matryoshka: Using expansion_factors {mid_out.sparse_coder.cfg.matroshka_expansion_factors} -> sizes: {matryoshka_sizes} (d_in={d_in})")
        else:
            # Default to single size
            matryoshka_sizes = [mid_out.sparse_coder.cfg.k]
            print(f"Matryoshka: Using default k: {matryoshka_sizes}")
        
        # Calculate cumulative k values for slicing
        cumulative_k = [0]
        for k in matryoshka_sizes:
            cumulative_k.append(cumulative_k[-1] + k)
        
        print(f"Matryoshka: Cumulative k values: {cumulative_k}")
        
        # Perform decoding for each slice
        slice_outputs = []
        total_fvu = 0.0
        total_auxk_loss = 0.0
        total_multi_topk_fvu = 0.0
        
        for i in range(len(matryoshka_sizes)):
            k_start = cumulative_k[i]
            k_end = cumulative_k[i + 1]
            
            print(f"Matryoshka: Processing slice {i}: k_start={k_start}, k_end={k_end}")
            
            slice_out = self.decode_slice(
                mid_out,
                y,
                module_name,
                k_start,
                k_end,
                detach_grad,
                **kwargs,
            )
            
            slice_outputs.append(slice_out)
            total_fvu += slice_out.fvu
            total_auxk_loss += slice_out.auxk_loss
            total_multi_topk_fvu += slice_out.multi_topk_fvu
        
        # Average the losses across all slices
        num_slices = len(matryoshka_sizes)
        avg_fvu = total_fvu / num_slices
        avg_auxk_loss = total_auxk_loss / num_slices
        avg_multi_topk_fvu = total_multi_topk_fvu / num_slices
        
        print(f"Matryoshka: Final losses - avg_fvu={avg_fvu:.6f}, avg_auxk={avg_auxk_loss:.6f}, avg_multi_topk={avg_multi_topk_fvu:.6f}")
        
        # Use the output from the largest slice as the main output
        main_output = slice_outputs[-1]
        
        # Create combined ForwardOutput
        combined_output = replace(
            main_output,
            fvu=avg_fvu,
            auxk_loss=avg_auxk_loss,
            multi_topk_fvu=avg_multi_topk_fvu,
        )
        
        if advance:
            for hookpoint in list(self.outputs.keys()):
                if hookpoint == module_name:
                    del self.outputs[hookpoint]
        
        return combined_output
    
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
            candidate_indices.append(
                layer_mid.latent_indices + i * layer_mid.sparse_coder.num_latents
            )
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
