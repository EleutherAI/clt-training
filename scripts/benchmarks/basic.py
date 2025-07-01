#%%
%load_ext autoreload
%autoreload 2
%env CUDA_VISIBLE_DEVICES=0
# %env TORCH_LOGS=kernel_code

import torch

B = 2**15
D = 2048
K = 32
E = 2**17

dtype = torch.bfloat16
W_enc = torch.randn(E, D, device="cuda", requires_grad=True, dtype=dtype)
W_dec = torch.randn(E, D, device="cuda", requires_grad=True, dtype=dtype)
data = torch.randn(B, D, device="cuda", dtype=dtype)
values = torch.randn(B, K, device="cuda", dtype=dtype, requires_grad=True)
indices = torch.randint(0, E, (B, K), device="cuda", dtype=torch.int32)
#%%
# with torch.no_grad():
#     values_f32, W_dec_f32 = values.float(), W_dec.float()
#     row_indices = torch.arange(indices.shape[0], device=indices.device, dtype=indices.dtype)
#     row_indices = row_indices[:, None].expand(-1, indices.shape[1])
#     # sparse_matrix_bf16 = torch.sparse_coo_tensor(torch.stack([row_indices.flatten(), indices.flatten()], dim=0), values.flatten(), (indices.shape[0], W_dec_f32.shape[0]))
#     # dense_matrix = sparse_matrix_bf16.to_dense()
#     sparse_matrix = torch.sparse_coo_tensor(torch.stack([row_indices.flatten(), indices.flatten()], dim=0), values_f32.flatten(), (indices.shape[0], W_dec_f32.shape[0]))
#     # sparse_matrix = sparse_matrix.to_sparse_csr()
#%%
import torch.nn.functional as F
import torch.utils.benchmark
from sparsify.nanogpt import linear
from sparsify.kernels import COODecoder, triton_sparse_dense_matmul
from sparsify.fused_encoder import fused_encoder, FusedEncoderCOO
from sparsify.utils import decoder_impl

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

@torch.no_grad()
def decode_torch_emb_bag(indices, values, W_dec):
    return torch.nn.functional.embedding_bag(indices, W_dec, per_sample_weights=values, mode="sum")

@torch.no_grad()
def decode_openai(indices, values, W_dec):
    return triton_sparse_dense_matmul(indices, values, W_dec)

@torch.no_grad()
def decode_xformers(indices, values, W_dec):
    return decoder_impl(indices, values, W_dec)

@torch.no_grad()
def decode_torch_sparse(sparse_matrix, W_dec):
    return sparse_matrix @ W_dec

@torch.no_grad()
def decode_torch_dense(dense_matrix, W_dec):
    return dense_matrix @ W_dec

# time(decode_torch_emb_bag, indices, values, W_dec)
# time(decode_openai, indices, values, W_dec)
decode_time = time(decode_xformers, indices, values, W_dec)
# time(decode_torch_sparse, sparse_matrix, W_dec_f32)
# time(decode_torch_dense, dense_matrix, W_dec)

sparse_bytes = 2 * B * (K + 1) * D
gpu_flops = 150e12
gpu_membw = 696e9
seconds_decoder = sparse_bytes / gpu_membw
print(f"Decoder time: {decode_time} seconds, expected: {seconds_decoder}")
print(f"Decoder utilization: {seconds_decoder / decode_time}")
#%%
@torch.no_grad()
def forward_topk(data, W_enc):
    return torch.topk(data @ W_enc.T, k=K, dim=-1)

@torch.no_grad()
def forward_groupmax(data, W_enc):
    bias = torch.zeros(E, device="cuda", dtype=dtype)
    return fused_encoder(
        data,
        W_enc,
        bias,
        k=K,
        activation="groupmax",
    )

@torch.no_grad()
def forward_decoder(values, indices, W_dec):
    return decoder_impl(indices, values, W_dec)

def forward_backward(data, W_enc, W_dec, activation: str = "topk"):
    bias = torch.zeros(E, device="cuda", dtype=dtype)
    encoded = fused_encoder(
        data,
        W_enc,
        bias,
        k=K,
        activation=activation,
    )
    indices, values = encoded.top_indices, encoded.top_acts
    decoded = decoder_impl(indices, values, W_dec)
    loss = (decoded - data).pow(2).mean()
    loss.backward()

@torch.no_grad()
def forward_coo(data, W_enc):
    bias = torch.zeros(E, device="cuda", dtype=dtype)
    values, rows, cols = FusedEncoderCOO.apply(data, W_enc, bias, K)
    return values, rows, cols

def forward_backward_coo(data, W_enc, W_dec):
    bias = torch.zeros(E, device="cuda", dtype=dtype)
    values, rows, cols = FusedEncoderCOO.apply(data, W_enc, bias, K)
    decoded = COODecoder.apply(rows, cols, values, W_dec.T, data.shape[0])
    loss = (decoded - data).pow(2).mean()
    loss.backward()

# time(forward_topk, data, W_enc)
# time(forward_groupmax, data, W_enc)
# time(forward_decoder, values, indices, W_dec)
# time(forward_backward, data, W_enc, W_dec)
# time(forward_backward, data, W_enc, W_dec, activation="groupmax")
# time(forward_coo, data, W_enc)
time(forward_backward_coo, data, W_enc, W_dec, compiled=False)
#%%
