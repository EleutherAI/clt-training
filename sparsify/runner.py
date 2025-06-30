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
        """Matryoshka decoding with safe slicing.

        Instead of cutting contiguous blocks, we *mask* latents whose indices
        exceed kᵢ.  This works even when ``cfg.k`` is smaller than any kᵢ.
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
