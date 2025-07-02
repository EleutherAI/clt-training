
#%%
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import threading
import os
import subprocess
eval_data = json.load(open("data/gpt2-eval-data/data.json"))
tokenizer = AutoTokenizer.from_pretrained(eval_data["model_name"])
run_names = [
    "bs16-lr2e-4-nonskip-tied-no-affine-ef128-k16-adam8-bf16",
    "bs16-lr2e-4-nonskip-ef128-k16-adam8-bf16",
    "bs8-lr3e-4-tied-ef128-k16",
    "bs8-lr2e-4-none-ef128-k16"
]
judge_ctx = "j1"
available_gpus = [1, 2, 3]
n_can_run_parallel = 4

n_occupants = {gpu: 0 for gpu in available_gpus}
start_end_mutex = threading.Lock()
wake_up_signal = threading.Condition(start_end_mutex)

def judge_run(run_name, prompt, i):
    with start_end_mutex:
        prompt_name = f"{judge_ctx}-{i}"
        results_path = f"../results/gpt2-eval/{prompt_name}-gpt2/results.json"
        if os.path.exists(results_path):
            return
        while all(n_occupants[gpu] >= n_can_run_parallel for gpu in available_gpus):
            wake_up_signal.wait()
        gpu = min(available_gpus, key=lambda gpu: n_occupants[gpu])
        assert n_occupants[gpu] < n_can_run_parallel
        n_occupants[gpu] += 1

    try:
        subprocess.check_call(
            ["scripts/judge-gpt", run_name],
            env=os.environ | dict(
                pn=prompt_name,
                pt=tokenizer.decode(prompt),
                CUDA_VISIBLE_DEVICES=str(gpu),
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
    finally:
        with start_end_mutex:
            n_occupants[gpu] -= 1
            wake_up_signal.notify()

threads = []
for i, prompt in enumerate(eval_data["prompts"]):
    for run_name in run_names:
        thread = threading.Thread(target=judge_run, args=(run_name, prompt, i))
        threads.append(thread)
        thread.start()
for thread in tqdm(threads, desc=f"Evaluating runs"):
    thread.join()
#%%
