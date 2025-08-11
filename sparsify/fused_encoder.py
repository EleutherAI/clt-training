import os
from typing import Literal, NamedTuple

import torch
import torch.distributed.tensor as dtensor
import torch.nn.functional as F

try:
    import rtopk
except ImportError:
    rtopk = None

from .kernels import (
    COODecoder,
    triton_coo_sparse_dense_matmul,
    triton_sparse_transpose_dense_matmul,
)
from .nanogpt import linear
from .utils import decoder_impl

NO_COMPILE = os.environ.get("SPARSIFY_NO_COMPILE", "0") == "1"
NO_RTOPK = os.environ.get("SPARSIFY_NO_RTOPK", "1") == "1"

MAX_SIZE = 1024


@torch.compile
def rtopk_topk(data, k: int, max_iter=10, k_div: int = 1):
    if rtopk is None or NO_RTOPK:
        return torch.topk(data, k, dim=1, sorted=False)
    else:
        if data.shape[-1] < MAX_SIZE:
            return rtopk.ops.rtopk(data, k, max_iter=max_iter)
        if data.shape[-1] % MAX_SIZE != 0:
            data = torch.nn.functional.pad(data, (0, data.shape[-1] % MAX_SIZE))
        data = data.unflatten(-1, (-1, MAX_SIZE))
        if data.shape[-1] <= k:
            indices = torch.arange(
                data.shape[-1], device=data.device, dtype=torch.int32
            )
            indices = indices.unsqueeze(0).expand(data.shape[0], -1)
            values = data
        else:
            values, indices = rtopk.ops.rtopk(data, k=k // k_div, max_iter=max_iter)

        values_l2, indices_l2 = rtopk_topk(
            values.flatten(-2), k=k, max_iter=max_iter, k_div=k_div
        )
        indices = (
            (
                indices
                + torch.arange(
                    data.shape[-2], device=indices.device, dtype=torch.int32
                )[:, None]
                * data.shape[-1]
            )
            .flatten(-2)
            .gather(-1, indices_l2.long())
        )
        return values_l2, indices


class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""


class FusedEncoder(torch.autograd.Function):
    @torch.compile(disable=NO_COMPILE)
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        values,
        indices,
        activation: Literal["groupmax"] | str | None = None,
    ):
        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, values, indices)
        ctx.k = values.shape[-1]
        ctx.activation = activation
        return values

    # @torch.compile
    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_values):
        input, weight, bias, values, indices = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        activation = ctx.activation

        grad_values = grad_values * (values > 0).to(grad_values)

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
            # Accumulate contributions into the correct rows of grad_weight.
            _, D = input.shape
            if not isinstance(grad_values, dtensor.DTensor):
                grad_weight = triton_sparse_transpose_dense_matmul(
                    indices,
                    grad_values.float(),
                    input,
                    N=weight.shape[0],
                )
            else:
                mesh = grad_values.device_mesh
                local_weight = weight.to_local()
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
                        0, local_weight.shape[0] - 1
                    )
                local_grad_weight = triton_sparse_transpose_dense_matmul(
                    indices,
                    values.float(),
                    gathered_input,
                    N=local_weight.shape[0],
                )
                grad_weight = dtensor.DTensor.from_local(
                    local_grad_weight,
                    mesh,
                    (dtensor.Replicate(), dtensor.Shard(0)),
                )

        # The k parameter is an int, so return None for its gradient.
        return grad_input, grad_weight, grad_bias, None, None, None


def batch_topk(preacts, k, return_indices=False):
    expected_k = k * preacts.shape[0]
    if isinstance(preacts, dtensor.DTensor):
        mesh = preacts.device_mesh
        local_preacts = preacts.to_local()
        expected_local_k = k * local_preacts.shape[0]
        local_values_less, original_indices = rtopk_topk(local_preacts, k=k * 4)
        original_indices = original_indices.long()
        local_values = torch.topk(
            local_values_less.flatten(), expected_local_k, sorted=False
        ).values
        all_values = dtensor.DTensor.from_local(
            local_values,
            mesh,
            (dtensor.Shard(0), dtensor.Shard(0)),
        )
        all_values = all_values.redistribute(
            mesh, (dtensor.Shard(0), dtensor.Replicate())
        )
        combined_values = all_values.to_local()
        values, indices = torch.topk(combined_values, expected_k, sorted=False)
        values, indices = values[0], indices[0]
        if return_indices:
            return values, indices
        else:
            threshold = values.min()
            local_preacts[local_preacts < threshold] = 0
            return preacts, original_indices
    else:
        values_less, original_indices = rtopk_topk(preacts, k=k * 4)
        values, indices = torch.topk(values_less.flatten(), expected_k, sorted=False)
        if return_indices:
            return values, indices
        else:
            threshold = values.min()
            preacts[preacts < threshold] = 0
            return preacts, original_indices


class FusedEncoderCOO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, k: int, use_fp8: bool = False):
        """Forward pass for BatchTopK with COO sparse kernels."""
        preacts = F.relu(linear(input, weight, bias, use_fp8))
        preact_values, preact_indices = preacts.topk(k * 4, dim=-1, sorted=False)
        values, indices = batch_topk(preact_values, k, return_indices=True)
        indices = preact_indices.flatten()[indices]
        row_indices, col_indices = (
            indices // preacts.shape[1],
            indices % preacts.shape[1],
        )
        ctx.save_for_backward(input, weight, bias, row_indices, col_indices, values)
        return values, row_indices, col_indices

    @staticmethod
    def backward(ctx, grad_values, _grad_row_indices, _grad_col_indices):
        input, weight, bias, row_indices, col_indices, values = ctx.saved_tensors
        grad_values = grad_values * (values > 0).to(grad_values)
        grad_input = grad_weight = grad_bias = None

        # --- Grad w.r.t. input ---
        if ctx.needs_input_grad[0]:
            grad_input = COODecoder.apply(
                row_indices,
                col_indices,
                grad_values,
                weight,
            )

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros_like(bias)
            grad_bias.index_add_(
                0, row_indices.flatten(), grad_values.flatten().type_as(bias)
            )
            grad_bias = grad_bias

        # --- Grad w.r.t. weight ---
        if ctx.needs_input_grad[1]:
            grad_weight = triton_coo_sparse_dense_matmul(
                torch.stack([row_indices, col_indices]),
                grad_values.float(),
                input,
                N=weight.shape[0],
            )

        return grad_input, grad_weight, grad_bias, None, None, None


def fused_encoder(
    input,
    weight,
    bias,
    k: int,
    activation: Literal["groupmax", "topk"],
    use_fp8: bool = False,
) -> EncoderOutput:
    """
    Convenience wrapper that performs an nn.Linear followed by `activation` with
    a backward pass optimized using index_add.

    input:  (N, D)
    weight: (M, D)
    bias:   (M,)
    k:      int (number of top elements to select along dim=1)
    """
    with torch.no_grad():
        preacts = linear(input, weight, bias, use_fp8)
        preacts.relu_()

        original_indices = None
        if activation == "batchtopk":
            preacts, original_indices = batch_topk(preacts, k)
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
                local_values, local_indices = rtopk_topk(local_acts, k=k)
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
                if original_indices is not None:
                    local_indices = original_indices
                    local_values = torch.gather(local_acts, 1, original_indices)
                else:
                    local_values, local_indices = rtopk_topk(local_acts, k=k)
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
                local_values, local_indices_ = rtopk_topk(local_values, k=k)
                local_indices = torch.gather(local_indices, 1, local_indices_.long())
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
                values, indices = rtopk_topk(preacts, k=k)
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
                # values = values.redistribute(
                #     mesh, (dtensor.Shard(0), dtensor.Replicate())
                # )
                # indices = indices.redistribute(
                #     mesh, (dtensor.Shard(0), dtensor.Replicate())
                # )
            else:
                num_latents = preacts.shape[1]
                values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)
                offsets = torch.arange(
                    0, num_latents, num_latents // k, device=preacts.device
                )
                indices = offsets + indices
        else:
            raise ValueError(f"Unknown activation: {activation}")

    values = FusedEncoder.apply(input, weight, bias, values, indices, activation)

    return EncoderOutput(
        top_acts=values,
        top_indices=indices,
    )


class DeadLatentLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, loss_weight):
        ctx.save_for_backward(inputs, weight, bias, loss_weight)
        return inputs.new_tensor(0.0)

    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    @staticmethod
    def backward(ctx, grad):
        inputs, weight, bias, loss_weight_mul = ctx.saved_tensors
        was_dtensor = False
        lwmul_was_dtensor = isinstance(loss_weight_mul, dtensor.DTensor)
        if isinstance(weight, dtensor.DTensor):
            was_dtensor = True
            mesh = weight.device_mesh
            placements = weight.placements
            if not lwmul_was_dtensor:
                loss_weight_mul = dtensor.DTensor.from_local(
                    loss_weight_mul, mesh, (dtensor.Replicate(), dtensor.Replicate())
                )
            loss_weight_mul = loss_weight_mul.redistribute(mesh, placements)
        inputs, weight, bias, loss_weight_mul, grad = (
            inputs.to_local(),
            weight.to_local(),
            bias.to_local(),
            loss_weight_mul.to_local(),
            grad.to_local(),
        )
        loss_weight = grad * loss_weight_mul
        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[1]:
            grad_weight = -torch.einsum("...x,yx,y->yx", inputs, weight, loss_weight)
            if was_dtensor:
                grad_weight = dtensor.DTensor.from_local(grad_weight, mesh, placements)
        grad_bias = None
        if ctx.needs_input_grad[2]:
            grad_bias = -loss_weight.broadcast_to(bias.shape)
            if was_dtensor:
                grad_bias = dtensor.DTensor.from_local(grad_bias, mesh, placements)
        grad_weight_mul = None
        if ctx.needs_input_grad[3]:
            grad_weight_mul = -torch.einsum("...x,yx,->y", inputs, weight, grad)
            if was_dtensor:
                grad_weight_mul = dtensor.DTensor.from_local(
                    grad_weight_mul, mesh, placements
                )
                grad_weight_mul = grad_weight_mul.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                )
            if not lwmul_was_dtensor:
                grad_weight_mul = grad_weight_mul.to_local()
        return None, grad_weight, grad_bias, grad_weight_mul
