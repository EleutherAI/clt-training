from typing import Literal, NamedTuple

import torch
import torch.distributed.tensor as dtensor
import torch.nn.functional as F

from .kernels import triton_sparse_transpose_dense_matmul
from .utils import decoder_impl


class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""

    pre_acts: torch.Tensor
    """Activations before the top-k selection."""


CONTRIB_BATCH_SIZE = 4096


class FusedEncoder(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        k: int,
        activation: Literal["groupmax", "topk", "batchtopk"],
    ):
        """
        input:  (N, D)
        weight: (M, D)
        bias:   (M,)
        k:      int (number of top elements to select along dim=1)
        """
        preacts = F.relu(F.linear(input, weight, bias))

        if activation == "batchtopk":
            expected_k = k * preacts.shape[0]
            if isinstance(preacts, dtensor.DTensor):
                local_preacts = preacts.to_local()
                local_threshold = torch.topk(
                    local_preacts.flatten(), expected_k, sorted=False
                ).values[-1]
                local_preacts[local_preacts < local_threshold] = 0
            else:
                threshold = torch.topk(
                    preacts.flatten(), expected_k, sorted=False
                ).values[-1]
                preacts[preacts < threshold] = 0
            k *= 4
            activation = "topk"

        # Get top-k values and indices for each row
        if activation == "topk":
            if (
                isinstance(preacts, dtensor.DTensor)
                and preacts.device_mesh.shape[1] == 1
            ):
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                local_values, local_indices = local_acts.topk(k, dim=1, sorted=False)
                values = dtensor.DTensor.from_local(
                    local_values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
                indices = dtensor.DTensor.from_local(
                    local_indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
            elif isinstance(preacts, dtensor.DTensor):
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                local_values, local_indices = local_acts.topk(k, dim=1, sorted=False)
                local_indices += mesh.get_local_rank(1) * local_acts.shape[1]
                values = dtensor.DTensor.from_local(
                    local_values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                ).redistribute(mesh, (dtensor.Shard(0), dtensor.Replicate()))
                indices = dtensor.DTensor.from_local(
                    local_indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                ).redistribute(mesh, (dtensor.Shard(0), dtensor.Replicate()))
                local_values, local_indices = values.to_local(), indices.to_local()
                local_values, local_indices_ = local_values.topk(k, dim=1, sorted=False)
                local_indices = torch.gather(local_indices, 1, local_indices_)
                values = dtensor.DTensor.from_local(
                    local_values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
                indices = dtensor.DTensor.from_local(
                    local_indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
            else:
                values, indices = torch.topk(preacts, k, dim=-1, sorted=False)
        elif activation == "groupmax":
            if isinstance(preacts, dtensor.DTensor):
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                assert k % mesh.shape[1] == 0
                local_k = k // mesh.shape[1]
                local_values, local_indices = local_acts.unflatten(
                    -1, (local_k, -1)
                ).max(dim=-1)
                offsets = torch.arange(
                    0,
                    local_acts.shape[1],
                    local_acts.shape[1] // local_k,
                    device=preacts.device,
                )
                mesh_offset = mesh.get_local_rank(1) * local_k
                indices = mesh_offset + offsets + local_indices
                values = local_values
                values = dtensor.DTensor.from_local(
                    values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                )
                indices = dtensor.DTensor.from_local(
                    indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                )
                values = values.redistribute(
                    mesh, (dtensor.Shard(0), dtensor.Replicate())
                )
                indices = indices.redistribute(
                    mesh, (dtensor.Shard(0), dtensor.Replicate())
                )
            else:
                num_latents = preacts.shape[1]
                values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)
                offsets = torch.arange(
                    0, num_latents, num_latents // k, device=preacts.device
                )
                indices = offsets + indices
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, indices)
        ctx.k = k
        ctx.activation = activation
        return values, indices, preacts

    @torch.compile
    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_values, grad_indices, grad_preacts):
        input, weight, bias, indices = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        activation = ctx.activation

        # --- Grad w.r.t. input ---
        if ctx.needs_input_grad[0]:
            grad_input = decoder_impl(
                indices,
                grad_values,
                weight,
            )

        if isinstance(grad_values, dtensor.DTensor):
            mesh = grad_values.device_mesh
            local_size = weight.to_local().shape[0]
            start_feature = mesh.get_local_rank(1) * local_size
            end_feature = start_feature + local_size

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            if isinstance(bias, dtensor.DTensor):
                mesh = bias.device_mesh
                grad_bias = torch.zeros_like(bias.to_local())
                all_indices = indices.flatten()
                all_indices = all_indices.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()
                all_values = grad_values.flatten()
                all_values = all_values.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()

                # TODO bespoke all-to-all gradient communication
                # likely won't be necessary, the encoder backward pass is fast
                mask = (all_indices >= start_feature) & (all_indices < end_feature)
                all_indices = all_indices[mask] - start_feature
                all_values = all_values[mask]

                grad_bias.index_add_(
                    0, all_indices, all_values.type_as(bias.to_local())
                )
                grad_bias = dtensor.DTensor.from_local(
                    grad_bias, mesh, (dtensor.Replicate(), dtensor.Shard(0))
                )
            else:
                grad_bias = torch.zeros_like(bias)
                grad_bias.index_add_(
                    0, indices.flatten(), grad_values.flatten().type_as(bias)
                )

        # --- Grad w.r.t. weight ---
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)

            # Accumulate contributions into the correct rows of grad_weight.
            _, D = input.shape
            if not isinstance(grad_weight, dtensor.DTensor):
                # Compute contributions from each top-k element:
                # computed as grad_values * input for each top-k location.
                contributions = grad_values.unsqueeze(2) * input.unsqueeze(1)
                # Flatten contributions to shape (N*k, D)
                contributions = contributions.reshape(-1, D)
                # print(grad_weight)  # (M, D); TP sharded along dim=0
                # print(indices.flatten())  # (N*k); DP sharded along dim=0
                # print(contributions)  # (N*k, D); DP sharded along dim=0
                grad_weight.index_add_(
                    0, indices.flatten(), contributions.type_as(weight)
                )
            else:
                mesh = grad_weight.device_mesh
                local_grad_weight = grad_weight.to_local()
                gathered_input = input.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()
                if activation == "groupmax":
                    indices = indices.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Shard(1))
                    ).to_local()
                    values = grad_values.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Shard(1))
                    ).to_local()
                    local_k = ctx.k // mesh.shape[1]
                    start_f = mesh.get_local_rank(1) * local_k
                    indices = indices - start_f
                else:
                    gathered_indices = indices.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Replicate())
                    ).to_local()
                    gathered_values = grad_values.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Replicate())
                    ).to_local()

                    indices = gathered_indices.view(-1, ctx.k)
                    values = gathered_values.view(-1, ctx.k)

                    mask = (indices >= start_feature) & (indices < end_feature)
                    values *= mask.type_as(values)
                    indices = (indices - start_feature).clamp(
                        0, local_grad_weight.shape[0] - 1
                    )
                local_grad_weight += triton_sparse_transpose_dense_matmul(
                    indices,
                    values.float(),
                    gathered_input,
                    N=local_grad_weight.shape[0],
                )

        # The k parameter is an int, so return None for its gradient.
        return grad_input, grad_weight, grad_bias, None, None


def fused_encoder(
    input,
    weight,
    bias,
    k: int,
    activation: Literal["groupmax", "topk"],
) -> EncoderOutput:
    """
    Convenience wrapper that performs an nn.Linear followed by `activation` with
    a backward pass optimized using index_add.
    """
    return EncoderOutput(
        *FusedEncoder.apply(input, weight, bias, k, activation)  # type: ignore
    )
