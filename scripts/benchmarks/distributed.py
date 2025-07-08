from sparsify import SparseCoder, SparseCoderConfig
import torch
import torch.distributed as dist
from datetime import timedelta
import os
from torch.distributed.tensor import init_device_mesh, DTensor, Replicate, Shard
import torch.utils.benchmark
from contextlib import nullcontext, redirect_stdout


def time(fn, *args, compiled=True, **kwargs):
    if compiled:
        fn = torch.compile(fn)
    timer = torch.utils.benchmark.Timer(
        stmt="fn(*args, **kwargs)",
        setup="fn(*args, **kwargs)",
        globals={"fn": fn, "args": args, "kwargs": kwargs},
    ).blocked_autorange()
    print(fn.__name__, timer.mean)
    return timer.mean


dtype = torch.bfloat16

B = 2**16
D = 2048
K = 32
E = 2**17

encoder_flops = 2 * B * D * E
# decoder_flops = 2 * B * K * D
sparse_bytes = 2 * B * (K + 1) * D

gpu_flops = 150e12
gpu_membw = 696e9

seconds_forward = encoder_flops / gpu_flops + sparse_bytes / gpu_membw
seconds_backward = encoder_flops / gpu_flops + 3 * sparse_bytes / gpu_membw

local_rank = os.environ.get("LOCAL_RANK")
distributed = local_rank is not None
rank = int(local_rank) if distributed else 0
torch.cuda.set_device(rank)

# Increase the default timeout in order to account for slow downloads
# and data preprocessing on the main rank
dist.init_process_group(
    "cpu:gloo,cuda:nccl",
    device_id=torch.device(rank),
    timeout=timedelta(weeks=1),
)
dist.barrier()
world_size = dist.get_world_size()


TP = world_size
E *= TP


assert world_size % TP == 0, "world_size must be divisible by tp"
mesh = init_device_mesh(
    "cuda",
    (world_size // TP, TP),
    mesh_dim_names=("dp", "tp"),
)
dp_rank = mesh.get_coordinate()[0]  # type: ignore

if rank == 0:
    print(f"Using DDP/TP across {dist.get_world_size()} GPUs.")

with (nullcontext() if rank == 0 else redirect_stdout(None)):
    print("Creating sparse coder and data")
    sparse_coder = SparseCoder(
        D,
        cfg=SparseCoderConfig(
            dtype="bfloat16",
            # activation="batchtopk",
            activation="topk",
            # activation="groupmax",
            tp_output=True,
            num_latents=E,
            k=K,
            transcode=True,
            n_targets=1,
        ),
        device="cuda",
        mesh=mesh,
    )
    with torch.no_grad():
        data = torch.randn(B, D, device="cuda", dtype=dtype)
        data = DTensor.from_local(data, device_mesh=mesh, placements=(Shard(0), Replicate()))
        out_data = data.clone()
        if sparse_coder.cfg.tp_output:
            out_data = out_data.redistribute(mesh, (Shard(0), Shard(1)))
    torch.distributed.barrier()
    print("Sparse coder created")
    @torch.no_grad
    def forward_pass(coder, x, y):
        return coder(x, y)

    def forward_backward_pass(coder, x, y):
        loss = coder(x)(y, 0).fvu
        loss.backward()

    # forward_time = time(forward_pass, sparse_coder, data, out_data, compiled=True)
    forward_time = time(forward_pass, sparse_coder, data, out_data, compiled=False)
    print(f"Forward time: {forward_time} seconds, expected:", seconds_forward)
    print(f"Forward utilization:", seconds_forward / forward_time)
    # backward_time = time(forward_backward_pass, sparse_coder, data, out_data, compiled=True)
    backward_time = time(forward_backward_pass, sparse_coder, data, out_data, compiled=False)
    print(f"Backward time: {backward_time} seconds, expected:", seconds_backward)
    print(f"Backward utilization:", seconds_backward / backward_time)
    print("Exiting")
    dist.destroy_process_group()
