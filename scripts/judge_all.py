
#%%
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import threading
import os
import subprocess

os.chdir(os.path.dirname(os.path.dirname(__file__)))

# model_type = "llama-1b"
model_type = "gpt2"
# model_type = "gemma2-2b"
judge_ctx = "j1"

eval_data = json.load(open(f"data/{model_type}-eval-data/data.json"))
tokenizer = AutoTokenizer.from_pretrained(eval_data["model_name"])
run_names = {
    "gpt2": [
        "bs16-lr2e-4-nonskip-tied-no-affine-ef128-k16-adam8-bf16",
        "bs16-lr2e-4-nonskip-ef128-k16-adam8-bf16",
        "bs8-lr3e-4-tied-ef128-k16",
        "bs8-lr2e-4-none-ef128-k16",
        "bs16-lr2e-4-btopk-clt-noskip-ef128-k16-adam8",
        "../clt-gpt2-finetune/bs8-lr2e-4-none-ef128-k16",
    ],
    "gemma2-2b": [
        "gemma-mntss-no-skip",
        "gemma-mntss-main",
        "gemmascope-transcoders-sparsify",
    ],
    "llama-1b": [
        "EleutherAI/llama1b-clt-none-ef64-k16",
        "EleutherAI/llama1b-clt-tied-ef64-k16",
        "EleutherAI/Llama-3.2-1B-mntss-skip-transcoder",
        "mntss/skip-transcoder-Llama-3.2-1B-131k-nobos",
        "./checkpoints/llama-sweep/bs32_lr2e-4_none_ef64_k32"
    ]
}[model_type]
extra_args = {
    "gpt2": {
        "bs16-lr2e-4-btopk-clt-noskip-ef128-k16-adam8": "--offload=True",
    },
    "gemma2-2b": {
        "gemma-mntss-no-skip": "--pre_ln_hook=True --post_ln_hook=True --offload=True",
        "gemma-mntss-main": "--pre_ln_hook=True --post_ln_hook=True --offload=True",
        "gemmascope-transcoders-sparsify": "--post_ln_hook=True --offload=True",
    },
    "llama-1b": {
        "EleutherAI/Llama-3.2-1B-mntss-skip-transcoder": "--pre_ln_hook=True"
    }
}[model_type]
script_name = {
    "gpt2": "gpt",
    "gemma2-2b": "gemma",
    "llama-1b": "llama",
}[model_type]
available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
n_can_run_parallel = {
    "gpt2": 4,
    "gemma2-2b": 1,
    "llama-1b": 1,
}[model_type]

n_occupants = {gpu: 0 for gpu in available_gpus}
start_end_mutex = threading.Lock()
wake_up_signal = threading.Condition(start_end_mutex)

def judge_run(run_name, prompt, i):
    with start_end_mutex:
        prompt_name = f"{judge_ctx}-{i}"
        results_path = f"../results/{model_type}-eval/{prompt_name}-{model_type}/results.json"
        if os.path.exists(results_path):
            return
        while all(n_occupants[gpu] >= n_can_run_parallel for gpu in available_gpus):
            wake_up_signal.wait()
        gpu = min(available_gpus, key=lambda gpu: n_occupants[gpu])
        assert n_occupants[gpu] < n_can_run_parallel
        n_occupants[gpu] += 1
        # print(f"Received GPU {gpu} for {prompt_name} ({run_name})")
        # print(n_occupants)

    try:
        args = [f"scripts/judge-{script_name}", run_name, *extra_args.get(run_name, "").split()]
        prompt_text = tokenizer.decode(prompt, skip_special_tokens=True)
        pipes = subprocess.Popen(
            args,
            env=os.environ | dict(
                pn=prompt_name,
                pt=prompt_text,
                CUDA_VISIBLE_DEVICES=str(gpu),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        stdout, stderr = pipes.communicate()
        if pipes.returncode != 0:
            print(f"Error running {run_name} on prompt {i}: {pipes.returncode}")
            print(stdout.decode("utf-8"))
            print(stderr.decode("utf-8"))
            return
    finally:
        with start_end_mutex:
            n_occupants[gpu] -= 1
            wake_up_signal.notify()
            # print(f"GPU {gpu} is free")
            # print(n_occupants)

threads = []
for i, prompt in enumerate(eval_data["prompts"]):
    for run_name in run_names:
        thread = threading.Thread(target=judge_run, args=(run_name, prompt, i))
        threads.append(thread)
        thread.start()
for thread in tqdm(threads, desc=f"Evaluating runs"):
    thread.join()
#%%
