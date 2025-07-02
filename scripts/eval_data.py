#%%
%env CUDA_VISIBLE_DEVICES=2
from sparsify.__main__ import load_artifacts, RunConfig


cfg = RunConfig(
    model="gpt2",
    dataset="EleutherAI/SmolLM2-135M-10B",
    split="train",
    ctx_len=16,
    return_overflowed_tokens=False,
    sae=None,
    loss_fn="kl"
)

model, dataset = load_artifacts(cfg, 0)
#%%
from tqdm import tqdm, trange
import torch
take_samples = 16384
torch.set_grad_enabled(False)
token_counts = torch.zeros(model.config.vocab_size, dtype=torch.int32)
for _, seq in zip(trange(take_samples), dataset):
    input_ids = seq["input_ids"]
    token_counts.scatter_add_(0, input_ids, torch.ones(input_ids.shape, dtype=torch.int32))
token_counts = token_counts.float()
token_counts /= token_counts.sum()
below_threshold = token_counts < 1e-4
#%%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.model)
#%%
import ftfy
target_size = 256
min_len = 10
below_threshold = below_threshold.cuda()
results = []
for _, seq in zip((bar := trange(take_samples)), dataset):
    input_ids = seq["input_ids"].cuda()
    outputs = model(input_ids=input_ids.unsqueeze(0))
    logprobs = torch.log_softmax(outputs.logits[0], dim=-1)
    correct_logprobs = torch.gather(logprobs[:-1], dim=-1, index=input_ids[1:, None])[:, 0]
    mask = outputs.logits[0].argmax(dim=-1)[:-1] == input_ids[1:]
    range_ = torch.arange(len(input_ids) - 1, device="cuda")
    mask &= (range_ > min_len)
    mask &= below_threshold[input_ids[1:]]
    mask &= ~(input_ids[1:, None] == (input_ids[None, :-1] * (range_[:, None] <= range_[None, :]))).any(dim=-1)
    mask &= correct_logprobs < -0.2
    losses = -correct_logprobs.cumsum(dim=0) / (range_ + 1)
    if mask.any():
        mask_tok = mask.tolist().index(True) + 2
        tokens = input_ids.tolist()[:mask_tok]
        text = tokenizer.decode(tokens)
        if any(not c.isascii() for c in text):
            continue
        if "\n" in text:
            continue
        results.append(tokens)
        bar.set_postfix(completion=len(results) / target_size)
        if len(results) >= target_size:
            break
#%%
