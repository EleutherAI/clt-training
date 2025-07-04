#%%
model_type = "llama-1b"
# model_type = "gpt2"
# model_type = "gemma-2-2b"
%env CUDA_VISIBLE_DEVICES=2
from sparsify.__main__ import load_artifacts, RunConfig


cfg = RunConfig(
    model={
        "gpt2": "gpt2",
        "gemma-2-2b": "google/gemma-2-2b",
        "llama-1b": "meta-llama/Llama-3.2-1B",
    }[model_type],
    dataset="EleutherAI/SmolLM2-135M-10B",
    split="train",
    ctx_len=16,
    return_overflowed_tokens=False,
    sae=None,
    loss_fn="kl",
    max_examples=65536
)

model, dataset = load_artifacts(cfg, 0, limit_before_processing=True)
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
        if tokenizer.eos_token_id in tokens:
            continue
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
from pathlib import Path
import json
to_save = dict(
    model_name=cfg.model,
    dataset_name=cfg.dataset,
    ctx_len=cfg.ctx_len,
    min_len=min_len,
    prompts=results,
)
save_path = Path(f"data/{model_type}-eval-data")
save_path.mkdir(parents=True, exist_ok=True)
json.dump(to_save, open(save_path / "data.json", "w"))
#%%
