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
print(f"Running {model_type}")

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
        "../clt-gpt2-finetune/bs8-lr2e-4-no-affine-ef128-k16",
        "bs32-lr2e-4-source-tied-ef128-k16-adam8",
        "bs32-lr2e-4-source-target-tied-ef128-k16-adam8",
        "/EleutherAI/gpt2-curt-clt-untied_global_batchtopk_jumprelu",
        "/EleutherAI/gpt2-curt-clt-tied_per_target_skip_global_batchtopk_jumprelu",
        "/EleutherAI/gpt2-curt-clt-untied-layerwise-tokentopk",
        "bs32-lr2e-4-clt-noskip-ef34-k16-adam8-bf16",
        "/EleutherAI/gpt2-mntss-transcoder-clt-relu-sp10-1b",
        "/EleutherAI/gpt2-mntss-transcoder-relu-sp6-skip",
        "/EleutherAI/gpt2-mntss-transcoder-clt-relu-sp8",
        "/EleutherAI/gpt2-curt-clt-untied_layerwise_batchtopk_jumprelu",

        "bs8-lr2e-4-no-affine-ef128-k8",
        "bs8-lr2e-4-no-affine-ef128-k16",
        "bs8-lr2e-4-no-affine-ef128-k24",
        "bs8-lr2e-4-no-affine-ef128-k32",

        "bs8-lr2e-4-nonskip-no-affine-ef128-k8",
        "bs8-lr2e-4-nonskip-no-affine-ef128-k16",
        "bs8-lr2e-4-nonskip-no-affine-ef128-k24",
        "bs8-lr2e-4-nonskip-no-affine-ef128-k32",

        "bs8-lr2e-4-tied-no-affine-ef128-k16",

        "bs8-lr2e-4-clt-noskip-ef128-k16"
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
        "./checkpoints/llama-sweep/bs32_lr2e-4_none_ef64_k32",
        "./checkpoints/llama-sweep/bs16_lr2e-4_no-skip_ef64_k32",
        "./checkpoints/llama-sweep/tied-pre_ef64_k32_bs32_lr2e-4",
        "EleutherAI/Llama-3.2-1B-mntss-transcoder-no-skip-sp10",
        "EleutherAI/Llama-3.2-1B-mntss-transcoder-no-skip-sp20",
    ]
}[model_type]
extra_args = {
    "gpt2": {
        # "bs16-lr2e-4-btopk-clt-noskip-ef128-k16-adam8": "--offload=True",
        "/EleutherAI/gpt2-curt-clt-untied_global_batchtopk_jumprelu": "--pre_ln_hook=True",
        "/EleutherAI/gpt2-curt-clt-tied_per_target_skip_global_batchtopk_jumprelu": "--pre_ln_hook=True",
        "/EleutherAI/gpt2-mntss-transcoder-clt-relu-sp10-1b": "--pre_ln_hook=True",
        "/EleutherAI/gpt2-mntss-transcoder-relu-sp6-skip": "--pre_ln_hook=True",
        "/EleutherAI/gpt2-mntss-transcoder-clt-relu-sp8": "--pre_ln_hook=True",
        "/EleutherAI/gpt2-curt-clt-untied-layerwise-tokentopk": "--pre_ln_hook=True",
        "/EleutherAI/gpt2-curt-clt-untied_layerwise_batchtopk_jumprelu": "--pre_ln_hook=True",
    },
    "gemma2-2b": {
        "gemma-mntss-no-skip": "--pre_ln_hook=True --post_ln_hook=True --offload=True",
        "gemma-mntss-main": "--pre_ln_hook=True --post_ln_hook=True --offload=True",
        "gemmascope-transcoders-sparsify": "--post_ln_hook=True --offload=True",
    },
    "llama-1b": {
        "./checkpoints/llama-sweep/tied-pre_ef64_k32_bs32_lr2e-4": "--pre_ln_hook=True",
        "EleutherAI/Llama-3.2-1B-mntss-skip-transcoder": "--pre_ln_hook=True",
        "EleutherAI/Llama-3.2-1B-mntss-transcoder-no-skip-sp10": "--pre_ln_hook=True",
        "EleutherAI/Llama-3.2-1B-mntss-transcoder-no-skip-sp20": "--pre_ln_hook=True",
    }
}[model_type]
script_name = {
    "gpt2": "gpt",
    "gemma2-2b": "gemma",
    "llama-1b": "llama",
}[model_type]
available_gpus = list(range(8)) if "CUDA_VISIBLE_DEVICES" not in os.environ else [int(gpu) for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
n_can_run_parallel = {
    "gpt2": 1,
    "gemma2-2b": 1,
    "llama-1b": 1,
}[model_type]

n_occupants = {gpu: 0 for gpu in available_gpus}
start_end_mutex = threading.Lock()
wake_up_signal = threading.Condition(start_end_mutex)

def judge_run(run_name, prompt, i):
    prompt_text = tokenizer.decode(prompt[:-1], skip_special_tokens=True)
    args = [f"scripts/judge-{script_name}", run_name, *extra_args.get(run_name, "").split()]
    if model_type == "gpt2":
        prompt_text = "<|endoftext|>" + prompt_text
        args.append("--remove_prefix=1")

    with start_end_mutex:
        prompt_name = f"{judge_ctx}-{i}"
        model_name = run_name.replace("/", "_")
        results_path = f"./results/{model_type}-eval/{model_name}/{prompt_name}-{model_type}/results.json"
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
