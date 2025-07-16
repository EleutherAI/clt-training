import os
from collections import defaultdict
from typing import Any, Optional, Type, TypeVar, cast

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from transformers import PreTrainedModel

try:
    import torchao.optim.quant_utils

    # https://github.com/pytorch/ao/issues/2296
    def _fp32_to_bf16_sr(x_f32: Tensor) -> Tensor:
        # For an FP32 number      [a31, ..., a16, a15, ..., a0] to be converted to BF16
        # - Round towards zero:   [a31, ..., a16,   0, ...,  0]
        # - Round away from zero: [a31, ..., a16+1, 0, ...,  0]
        # (since the value can be negative, we use round towards/away from zero instead
        # of round up/down)
        #
        # For stochastic rounding, we round away from zero with the probability of
        # [a15, ..., a0] / 2^16, where the bit pattern [a15, ..., a0] is interpreted
        # as uint16
        #
        # we have to use int32 since most arithmetic ops are not implemented
        # for uint32/int16/uint16
        if isinstance(x_f32, DTensor):
            rand_16bit = torch.randint(
                0,
                1 << 16,
                x_f32.to_local().shape,
                device=x_f32.device,
                dtype=torch.int32,
            )
        else:
            rand_16bit = torch.randint(
                0, 1 << 16, x_f32.shape, device=x_f32.device, dtype=torch.int32
            )
        x_f32_bits = x_f32.view(torch.int32)
        x_fraction = x_f32_bits & 0xFFFF  # lower 16 bits
        x_bf16_towards_zero = x_f32_bits & 0xFFFF0000  # upper 16 bits

        if isinstance(x_fraction, DTensor):
            rand_16bit = DTensor.from_local(
                rand_16bit,
                device_mesh=x_fraction.device_mesh,
                placements=x_fraction.placements,
            )

        x_f32_bits = torch.where(
            rand_16bit < x_fraction,  # this is True with the probability of p_fraction
            x_bf16_towards_zero
            + 0x10000,  # this might overflow, which will result in UB due to signed int
            x_bf16_towards_zero,
        )
        # alternative, slightly faster
        # x_f32_bits = (x_f32_bits + rand_16bit) & 0xFFFF0000
        return x_f32_bits.view(torch.float32).bfloat16()

    torchao.optim.quant_utils._fp32_to_bf16_sr = _fp32_to_bf16_sr
    import importlib

    importlib.reload(torchao.optim.adam)

except ImportError:
    print("torchao not installed, using default implementation of stochastic rounding.")

try:
    from torchao.optim.subclass_8bit import OptimState8bit
except ImportError:
    OptimState8bit = object

    # dummy class for when torchao is not installed
    class OptimState8bit(object):  # noqa
        pass


T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        (name, mod)
        for (name, mod) in model.base_model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


def resolve_widths(
    model: PreTrainedModel,
    module_names: list[str],
    dim: int = -1,
    mesh: Optional[DeviceMesh] = None,
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {
        model.base_model.get_submodule(name): name for name in module_names
    }
    shapes: dict[str, int] = {}

    def hook(module, _, output):
        # Unpack tuples if needed
        if isinstance(output, tuple):
            output, *_ = output

        name = module_to_name[module]
        shapes[name] = output.shape[dim]

    handles = [mod.register_forward_hook(hook) for mod in module_to_name]
    with torch.inference_mode() if mesh is None else torch.no_grad():
        dummy = {
            k: v.to(model.device) if mesh is None else distribute_tensor(v, mesh)
            for k, v in model.dummy_inputs.items()
        }
        try:
            model(**dummy)
        finally:
            for handle in handles:
                handle.remove()

    return shapes


def set_submodule(model: nn.Module, submodule_path: str, new_submodule: nn.Module):
    """
    Replaces a submodule in a PyTorch model dynamically.

    Args:
        model (nn.Module): The root model containing the submodule.
        submodule_path (str): Dotted path to the submodule.
        new_submodule (nn.Module): The new module to replace the existing one.

    Example:
        set_submodule(model, "encoder.layer.0.attention.self", nn.Identity())
    """
    parent_path, _, last_name = submodule_path.rpartition(".")
    parent_module = model.get_submodule(parent_path) if parent_path else model
    setattr(parent_module, last_name, new_submodule)


def sharded_axis(
    state_dict: dict[str, DTensor],
) -> dict[str, Optional[int]]:
    """
    Checks which axis each DTensor is sharded on.
    Returns a dictionary mapping each key to the axis it is sharded on,
    or None if it is not sharded.

    Args:
        state_dict (dict[str, DTensor]): The state dictionary containing DTensors.

    Returns:
        dict[str, Optional[int]]: A dictionary mapping each key to the
        axis it is sharded on.
    """
    sharded_axes = {}
    for key, tensor in state_dict.items():
        try:
            sharding = tensor.placements
        except AttributeError:
            print(f"Warning: key {key} is not a DTensor, skipping sharded axis check.")
            sharded_axes[key] = None
            continue
        assert isinstance(sharding[0], Replicate)
        if not isinstance(sharding[1], Shard):
            sharded_axes[key] = None
        else:
            sharded_axes[key] = sharding[1].dim
    return sharded_axes


def flatten_dict(d):
    if isinstance(d, list):
        combined = {}
        for i, x in enumerate(d):
            for k, v in flatten_dict(x).items():
                combined[(("list", i),) + k] = v
        return combined
    elif isinstance(d, dict):
        combined = {}
        for k, v in d.items():
            for k2, v2 in flatten_dict(v).items():
                combined[(k,) + k2] = v2
        return combined
    elif isinstance(d, OptimState8bit) or (
        isinstance(d, DTensor) and isinstance(d.to_local(), OptimState8bit)
    ):
        key_base = "int8state"
        tensor = d
        if isinstance(d, DTensor):
            tensor = tensor.to_local()
        attrs, ctx = tensor.__tensor_flatten__()
        attributes = {}
        for k in attrs:
            v = getattr(tensor, k)
            if isinstance(v, torch.Tensor) and isinstance(d, DTensor):
                if k == "qmap":
                    attributes[k] = DTensor.from_local(
                        v, d.device_mesh, (Replicate(), Replicate())
                    )
                    continue
                placements = d.placements
                if k == "scale" and isinstance(placements[1], Shard):
                    placements = (Replicate(), Shard(0))
                attributes[k] = DTensor.from_local(v, d.device_mesh, placements)
        combined = {}
        for k, v in flatten_dict(ctx).items():
            combined[((key_base, "ctx"),) + k] = v
        for k, v in flatten_dict(attributes).items():
            combined[((key_base, "attrs"),) + k] = v
        return combined
    else:
        return {(): d}


def unflatten_dict(d, path_here=()):
    if len(d) == 0:
        return {}
    if len(d) == 1 and next(iter(d.keys())) == ():
        return d[()]
    by_first_key = defaultdict(dict)
    for k, v in d.items():
        by_first_key[k[0]][k[1:]] = v
    first_first_key = next(iter(by_first_key.keys()))
    if isinstance(first_first_key, tuple) and len(first_first_key) == 2:
        if first_first_key[0] == "list":
            return [
                unflatten_dict(by_first_key[k]) for k in sorted(by_first_key.keys())
            ]
        elif first_first_key[0] == "int8state":
            attrs_flat, ctx_flat = (
                by_first_key[("int8state", "attrs")],
                by_first_key[("int8state", "ctx")],
            )
            placements = None
            if any(isinstance(v, DTensor) for v in attrs_flat.values()):
                placements = attrs_flat[("codes",)].placements
                device_mesh = attrs_flat[("codes",)].device_mesh
                new_attrs_flat = {}
                for k, v in attrs_flat.items():
                    if isinstance(v, DTensor):
                        new_attrs_flat[k] = v.to_local()
                    else:
                        new_attrs_flat[k] = v
                attrs_flat = new_attrs_flat
            attrs = unflatten_dict(attrs_flat)
            ctx = unflatten_dict(ctx_flat)
            tensor = OptimState8bit.__tensor_unflatten__(attrs, ctx)
            if placements is not None:
                return DTensor.from_local(tensor, device_mesh, placements)
            else:
                return tensor
        else:
            raise ValueError()
    else:
        return {k: unflatten_dict(v, path_here + (k,)) for k, v in by_first_key.items()}


def save_sharded(
    start_state_dict: dict[str, DTensor | torch.Tensor | Any],
    filename: str,
    mesh: Optional[DeviceMesh] = None,
    save_st: bool = True,
    unflatten: bool | None = None,
):
    if unflatten is None:
        unflatten = not save_st
    if mesh is not None and mesh.get_local_rank(0) != 0:
        torch.distributed.barrier()
        return False

    if unflatten:
        start_state_dict = flatten_dict(start_state_dict)

    tensor_state_dict = {
        k: v
        for k, v in start_state_dict.items()
        if isinstance(v, (DTensor, torch.Tensor))
    }
    non_tensor_state_dict = {
        k: v
        for k, v in start_state_dict.items()
        if not isinstance(v, (DTensor, torch.Tensor))
    }
    if mesh is None:
        state_dict = {k: v.cpu() for k, v in tensor_state_dict.items()}
    else:
        cpu_state_dict = {
            k: (v.to_local() if isinstance(v, DTensor) else v).cpu()
            for k, v in tensor_state_dict.items()
        }
        state_dict = {}
        shard_dims = sharded_axis(tensor_state_dict)

        for k, v in cpu_state_dict.items():
            replicated_dim = shard_dims[k]
            if replicated_dim is None:
                state_dict[k] = v
                continue
            out_tensor = (
                [torch.empty_like(v) for _ in range(mesh.shape[1])]
                if torch.distributed.get_rank() == 0
                else []
            )
            torch.distributed.gather_object(
                v,
                out_tensor,
                dst=0,
                group=mesh.get_group(1),
            )
            if torch.distributed.get_rank() == 0:
                out_tensor = torch.cat(out_tensor, dim=replicated_dim)
                state_dict[k] = out_tensor
        if torch.distributed.get_rank() != 0:
            torch.distributed.barrier()
            return False
    state_dict = non_tensor_state_dict | state_dict

    # state_dict = {k: ((v.redistribute(
    #     mesh,
    #     (Replicate(), Replicate())
    #     ).to_local().cpu()
    #                   if isinstance(v, DTensor)
    #                   else v.cpu())
    #                   if isinstance(v, (torch.Tensor, DTensor))
    #                   else v)
    #               for k, v in start_state_dict.items()}

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
    #     torch.distributed.barrier()
    #     return False

    if save_st:
        save_file(state_dict, filename)
    else:
        torch.save(state_dict, filename)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    print(f"Saved {filename}")
    return True


def load_sharded(
    filename: str,
    current_state_dict: dict[str, DTensor],
    mesh: DeviceMesh,
    load_st: bool = True,
    load_fast: bool = True,
    unflatten: bool | None = None,
):
    if unflatten is None:
        unflatten = not load_st
    torch.distributed.barrier()
    if unflatten:
        current_state_dict = flatten_dict(current_state_dict)
        current_state_dict = {
            k: v for k, v in current_state_dict.items() if isinstance(v, DTensor)
        }
    if torch.distributed.get_rank() == 0 or load_fast:
        shard_dims = sharded_axis(current_state_dict)
        if load_st:
            cpu_state_dict = load_file(
                filename,
                device="cpu",
            )
        else:
            cpu_state_dict = torch.load(filename, map_location="cpu")
        dp_size, tp_size = mesh.shape
        partitioned_state_dict = {
            k: (
                v.chunk(tp_size, dim=shard_dims[k])
                if shard_dims.get(k) is not None
                else [v] * tp_size
            )
            for k, v in cpu_state_dict.items()
        }
        state_dicts = [
            {k: v[i] for k, v in partitioned_state_dict.items()} for i in range(tp_size)
        ]
        if not load_fast:
            state_dicts *= dp_size
            torch.distributed.barrier()
            torch.distributed.scatter_object_list(
                [None],
                state_dicts,
                src=0,
            )
        cpu_state_dict = state_dicts[mesh.get_local_rank(1)]
    else:
        torch.distributed.barrier()
        obj_list = [None]
        torch.distributed.scatter_object_list(
            obj_list,
            None,
            src=0,
        )
        cpu_state_dict = obj_list[0]
    state_dict = {
        k: (
            (
                DTensor.from_local(
                    v.to(torch.get_default_device()),
                    mesh,
                    current_state_dict[k].placements,
                )
                if k in current_state_dict
                else v.to(torch.get_default_device())
            )
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in cpu_state_dict.items()
    }
    if unflatten:
        state_dict = unflatten_dict(state_dict)
    print(f"Loaded {filename}")
    return state_dict


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return nn.functional.embedding_bag(
        top_indices, W_dec, per_sample_weights=top_acts.float(), mode="sum"
    )


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    assert not isinstance(top_acts, DTensor)
    assert not isinstance(W_dec, DTensor)
    assert top_acts.ndim == 2
    assert W_dec.ndim == 2
    if USE_XFORMERS:
        return xformers_embedding_bag(top_indices, W_dec, top_acts)
    else:
        return TritonDecoder.apply(top_indices, top_acts, W_dec.T)


class AvgGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, group: torch.distributed.ProcessGroup):
        ctx.group = group
        x = x.clone()
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.AVG, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # we use this for different output groups in the decoder, so we need to
        # sum the gradients
        grad_output = grad_output.clone()
        torch.distributed.all_reduce(
            grad_output, op=torch.distributed.ReduceOp.SUM, group=ctx.group
        )
        return grad_output, None


def parallelize_decoder(decoder):
    """
    Decorator to make the decoder function work on torch.DTensor.
    """

    # @torch.compile
    def wrapper(
        top_indices: Tensor | DTensor,
        top_acts: Tensor | DTensor,
        W_dec: Tensor | DTensor,
    ):
        # Check if the input is a DTensor
        if (
            isinstance(top_indices, DTensor)
            and isinstance(top_acts, DTensor)
            and isinstance(W_dec, DTensor)
        ):
            assert top_indices.device_mesh == top_acts.device_mesh == W_dec.device_mesh
            mesh = top_indices.device_mesh
            assert top_indices.placements == top_acts.placements
            placement = {}
            local_acts = top_acts.to_local()
            decoder_sharded_across_features = (
                isinstance(W_dec.placements[1], Shard) and W_dec.placements[1].dim == 0
            )
            features_replicated = isinstance(top_indices.placements[1], Replicate)
            if features_replicated or not decoder_sharded_across_features:
                if not isinstance(top_acts.placements[1], Replicate):
                    top_acts = top_acts.redistribute(
                        mesh, (top_acts.placements[0], Replicate())
                    )
                local_acts = top_acts.to_local()
                local_acts = AvgGrad.apply(local_acts, mesh.get_group(1))

            local_indices = top_indices.to_local()
            for i, p in enumerate(W_dec.placements):
                if isinstance(p, Shard):
                    if p.dim == 1:
                        placement[i] = Shard(1)
                    # decoder sharded across feature dimension, features not sharded
                    else:
                        features_per_rank = W_dec.to_local().shape[0]
                        rank = mesh.get_local_rank(1)
                        start_idx = rank * features_per_rank
                        end_idx = start_idx + features_per_rank
                        local_acts = (
                            local_acts
                            * (start_idx <= local_indices)
                            * (local_indices < end_idx)
                        )
                        local_indices = (local_indices - start_idx) % features_per_rank
                        placement[i] = Partial("sum")
            for i, p in enumerate(top_indices.placements):
                if isinstance(p, Shard) and p.dim == 0:
                    placement[i] = Shard(0)
            placement = [
                placement.get(i, Replicate()) for i in range(len(W_dec.placements))
            ]
            result_local = decoder(local_indices, local_acts, W_dec.to_local())
            result = DTensor.from_local(
                result_local, top_indices.device_mesh, placement
            )
            return result
        else:
            # If not a DTensor, call the decoder function directly
            return decoder(top_indices, top_acts, W_dec)

    return wrapper


try:
    from .kernels import TritonDecoder
    from .xformers import xformers_embedding_bag
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of sparse decoder.")
else:
    if os.environ.get("SPARSIFY_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of sparse decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode
decoder_impl = parallelize_decoder(decoder_impl)

USE_XFORMERS: bool = os.environ.get("SPARSIFY_USE_XFORMERS", "1") == "1"

DISTRIBUTE_MODEL: bool = os.environ.get("SPARSIFY_DISTRIBUTE_MODEL", "0") == "1"
if DISTRIBUTE_MODEL:

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    sdpa_val = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def new_sdpa(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
    ):
        return sdpa_val(
            module,
            query,
            key,
            value,
            is_causal=True,
            attention_mask=None,
            scaling=scaling,
            **kwargs,
        )

    ALL_ATTENTION_FUNCTIONS["sdpa"] = new_sdpa


class BackwardPrint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, message: str):
        ctx.message = message
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        print(ctx.message)
        return grad_output, None
