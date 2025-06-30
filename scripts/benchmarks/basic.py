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
import torch.nn.functional as F
import torch.utils.benchmark
from sparsify.nanogpt import linear
from sparsify.kernels import COODecoder
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
