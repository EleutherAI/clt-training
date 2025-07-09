from dataclasses import replace

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from .sparse_coder import MidDecoder, SparseCoder


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
                    no_extras=(hookpoint != module_name)
                    or mid_out.sparse_coder.cfg.secondary_target_tied,
                    denormalize=(hookpoint == module_name)
                    and not mid_out.sparse_coder.cfg.secondary_target_tied,
                    **kwargs,
                )
                if hookpoint != module_name:
                    output += out.sae_out
            else:
                layer_mid.next()

        if mid_out.sparse_coder.cfg.secondary_target_tied:
            candidate_indices = torch.cat(candidate_indices, dim=1)
            candidate_values = torch.cat(candidate_values, dim=1)
            candidate_indices = candidate_indices % mid_out.sparse_coder.num_latents
            new_mid_out = mid_out.copy(
                indices=candidate_indices,
                activations=candidate_values,
            )
            out = new_mid_out(
                y,
                index=0,
                add_post_enc=False,
                no_extras=False,
                denormalize=True,
                addition=out.sae_out,
                **kwargs,
            )

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
                num_latents = mid_out.sparse_coder.num_latents
                best_indices_local = best_indices
                best_values_local = best_values
                new_mid_out = mid_out.copy(
                    indices=best_indices_local,
                    activations=best_values_local,
                )
                out = new_mid_out(
                    y,
                    index=0,
                    add_post_enc=False,
                    **kwargs,
                )
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
