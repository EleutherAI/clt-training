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


"""Matryoshka‑style runner that *inherits all the cross‑layer coalescing tricks* of
`CrossLayerRunner` while still performing the slice‑by‑slice Matryoshka loss
ladder.

Public API intentionally mirrors `CrossLayerRunner` so you can swap the runner
with a single line:

```python
runner = MatryoshkaRunner()  # instead of CrossLayerRunner
```
"""

class MatryoshkaRunner:
    """Runs Matryoshka training *and* cross‑layer coalescing in one place."""

    def __init__(self):
        # One ``MidDecoder`` per *layer* that has been seen so far
        self.outputs: dict[str, MidDecoder] = {}

        # Stores (MidDecoder, will_be_last) for later gradient restoration
        self.to_restore: dict[str, tuple[MidDecoder, bool]] = {}

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _matryoshka_sizes(mid: MidDecoder) -> List[int]:
        cfg = mid.sparse_coder.cfg
        if cfg.matryoshka_k_values:
            return cfg.matryoshka_k_values
        if cfg.matryoshka_expansion_factors:
            return [mid.sparse_coder.d_in * ef for ef in cfg.matryoshka_expansion_factors]
        # Fallback to single‑k
        return [cfg.k]

    # ------------------------------------------------------------------
    # Cross‑layer decode for a *single* Matryoshka slice ----------------
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
        """A verbatim port of ``CrossLayerRunner.decode`` that operates on *one*
        ``MidDecoder`` (already sliced for a particular *k*).  It returns the
        usual ``ForwardOutput``.
        """
        # The original logic is copied almost 1‑for‑1 so that all coalescing
        # modes ("concat", "per-layer", or *none*) behave exactly the same.

        # ------------------------------------------------------------------
        # NB: We shadow *mid_out* inside the copied code so variable names match
        # the upstream implementation.  pylint/flake8 will warn, but it's safe.
        # ------------------------------------------------------------------
        self.outputs[module_name] = mid_out  # type: ignore[assignment]

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

            divide_by = max(1, len(self.outputs) - 1) if layer_mid.sparse_coder.cfg.divide_cross_layer else 1

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

        # Last output is guaranteed to be the current layer
        assert hookpoint == module_name  # noqa: B013, F821

        if not advance:
            for layer_mid in layer_mids:
                layer_mid.prev()

        if advance:
            for hookpoint in to_delete:
                del self.outputs[hookpoint]

        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public encode/decode interface -----------------------------------
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
        """Apply Matryoshka slicing *and* cross‑layer coalescing.

        For every kᵢ in the Matryoshka ladder we:
        1. Slice ``mid_out`` to that latent window.
        2. Run the full CrossLayer coalescing pipeline on the slice.
        3. Aggregate losses across slices.

        The *largest* slice's ``ForwardOutput`` fields are kept, but its loss
        metrics are replaced with the *mean* across all slices, matching the
        description in the paper.
        """
        print(f"\n{'='*60}")
        print(f"MatryoshkaRunner: Processing {module_name}")
        print(f"{'='*60}")
        
        # --------------------------------------------------------------
        # Determine k‑ladder boundaries
        # --------------------------------------------------------------
        k_values = self._matryoshka_sizes(mid_out)
        cumulative = [0]
        for k in k_values:
            cumulative.append(cumulative[-1] + k)
        
        print(f"Matryoshka k-values: {k_values}")
        print(f"Matryoshka cumulative boundaries: {cumulative}")
        print(f"Total features: {cumulative[-1]}")
        print(f"Original activations shape: {mid_out.latent_acts.shape}")
        print(f"Original indices shape: {mid_out.latent_indices.shape}")

        slice_outputs = []
        total_fvu = mid_out.latent_acts.new_tensor(0.0)
        total_aux = mid_out.latent_acts.new_tensor(0.0)
        total_multi = mid_out.latent_acts.new_tensor(0.0)

        # --------------------------------------------------------------
        # Run every slice
        # --------------------------------------------------------------
        print(f"\n{'='*40}")
        print(f"Processing {len(k_values)} Matryoshka slices:")
        print(f"{'='*40}")
        
        for i in range(len(k_values)):
            k_start, k_end = cumulative[i], cumulative[i + 1]
            k_size = k_end - k_start
            
            print(f"\n--- Slice {i+1}/{len(k_values)}: k={k_size} (indices {k_start}:{k_end}) ---")

            activations = mid_out.latent_acts[:, k_start:k_end]
            indices = mid_out.latent_indices[:, k_start:k_end]
            sliced_mid = mid_out.copy(activations=activations, indices=indices)
            
            print(f"  Slice activations shape: {activations.shape}")
            print(f"  Slice indices shape: {indices.shape}")
            print(f"  Slice indices range: {indices.min().item():.0f} to {indices.max().item():.0f}")

            out_slice = self._decode_slice(
                sliced_mid,
                y,
                module_name,
                detach_grad=detach_grad,
                advance=(advance and i == len(k_values) - 1),  # free only on last
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

        # --------------------------------------------------------------
        # Aggregate losses & craft final output (largest slice's output)
        # --------------------------------------------------------------
        n = len(k_values)
        avg_fvu = total_fvu / n
        avg_aux = total_aux / n
        avg_multi = total_multi / n

        print(f"\n{'='*40}")
        print(f"Loss Aggregation Results:")
        print(f"{'='*40}")
        print(f"Number of slices: {n}")
        print(f"Total FVU: {total_fvu.item():.6f}")
        print(f"Total AuxK: {total_aux.item():.6f}")
        print(f"Total Multi-TopK: {total_multi.item():.6f}")
        print(f"Average FVU: {avg_fvu.item():.6f}")
        print(f"Average AuxK: {avg_aux.item():.6f}")
        print(f"Average Multi-TopK: {avg_multi.item():.6f}")

        # --------------------------------------------------------------
        # Concatenate all slices back together for full activations/indices
        # --------------------------------------------------------------
        all_activations = torch.cat([out.latent_acts for out in slice_outputs], dim=1)
        all_indices = torch.cat([out.latent_indices for out in slice_outputs], dim=1)
        
        print(f"\n{'='*40}")
        print(f"Final Output Construction:")
        print(f"{'='*40}")
        print(f"Concatenated activations shape: {all_activations.shape}")
        print(f"Concatenated indices shape: {all_indices.shape}")
        print(f"Concatenated indices range: {all_indices.min().item():.0f} to {all_indices.max().item():.0f}")
        
        # Use the largest slice's output as the base, but replace activations/indices
        main_output = slice_outputs[-1]
        combined = replace(
            main_output, 
            latent_acts=all_activations,
            latent_indices=all_indices,
            fvu=avg_fvu, 
            auxk_loss=avg_aux, 
            multi_topk_fvu=avg_multi
        )
        
        print(f"Final output activations shape: {combined.latent_acts.shape}")
        print(f"Final output indices shape: {combined.latent_indices.shape}")
        print(f"Final output sae_out shape: {combined.sae_out.shape}")
        print(f"Final losses - FVU: {combined.fvu.item():.6f}, AuxK: {combined.auxk_loss.item():.6f}, Multi-TopK: {combined.multi_topk_fvu.item():.6f}")
        print(f"{'='*60}\n")
        
        return combined

    # ------------------------------------------------------------------
    # Convenience wrappers --------------------------------------------
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
        encoder_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
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

    # ------------------------------------------------------------------
    # House‑keeping ----------------------------------------------------
    # ------------------------------------------------------------------
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
