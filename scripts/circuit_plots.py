#%%
import json
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import os
sns.set_theme()
os.chdir(os.path.dirname(os.path.dirname(__file__)))
# model_type = "gpt2"
# model_type = "gemma2-2b"
model_type = "llama-1b"
eval_path = Path(f"results/{model_type}-eval/")
model_name = {
    "gpt2": "GPT2",
    "gemma2-2b": "Gemma 2 2B",
    "llama-1b": "Llama 3.2 1B",
}[model_type]
model_type_for_path = {
    "gpt2": "gpt2",
    "gemma2-2b": "gemma-2-2b",
    "llama-1b": "llama-3.2-1b",
}
judge_ctx = "j1"
run_names = {
    "gpt2": {
        ".._clt-gpt2-finetune_bs8-lr2e-4-none-ef128-k16": "FT PLT, no skip",
        "bs8-lr2e-4-none-ef128-k16": "PLT, skip",
        "bs16-lr2e-4-nonskip-ef128-k16-adam8-bf16": "PLT, no skip",
        "bs16-lr2e-4-nonskip-tied-no-affine-ef128-k16-adam8-bf16": "Tied CLT, no skip",
        "bs8-lr3e-4-tied-ef128-k16": "Tied CLT, skip",
        "bs16-lr2e-4-btopk-clt-noskip-ef128-k16-adam8": "CLT, no skip",
    },
    "gemma2-2b": {
        "gemma-mntss-no-skip": "PLT, no skip",
        "gemma-mntss-main": "PLT, skip",
        "gemmascope-transcoders-sparsify": "PLT, Gemmascope",
    },
    "llama-1b": {
        "EleutherAI_llama1b-clt-none-ef64-k16": "PLT, skip",
        "EleutherAI_llama1b-clt-tied-ef64-k16": "Tied CLT, skip",
        "._checkpoints_llama-sweep_bs32_lr2e-4_none_ef64_k32": "PLT, skip, k=32",
        "EleutherAI_Llama-3.2-1B-mntss-skip-transcoder": "PLT, skip, ReLU",
        "mntss_skip-transcoder-Llama-3.2-1B-131k-nobos": "PLT, skip, TopK",
    }
}[model_type]
# metric = "replacement_score_unpruned"
metric = "sweep_pruning_results.200.replacement_score"
# metric = "sweep_pruning_results.50.replacement_score"
# metric = "completeness_score_unpruned"
results = defaultdict(dict)
all_json_data = {}
for run_name, run_name_str in run_names.items():
    for results_path in (eval_path / run_name).glob(f"{judge_ctx}-*-{model_type_for_path[model_type]}/results.json"):
        prompt_n = int(results_path.parent.stem.split('-')[1])
        json_data = json.load(open(results_path))
        all_json_data[(run_name, prompt_n)] = json_data
        for k in metric.split('.'):
            json_data = json_data[k]
        metric_value = json_data
        results[run_name_str][prompt_n] = metric_value
result_sets = {k: set(v.keys()) for k, v in results.items()}
result_set = next(iter(result_sets.values()))
for res_set in result_sets.values():
    result_set = result_set.intersection(res_set)
results = {k: [v[k2] for k2 in result_set] for k, v in results.items()}

result_mean_std = {k: (np.mean(v), np.std(v)) for k, v in results.items()}
plt.figure(figsize=(10, 5))
all_results = []
for run_name_str, results_list in results.items():
    all_results.extend([
        {
            "run_name": run_name_str,
            "result": result
        } for result in results_list
    ])
all_results = pd.DataFrame(all_results)
# sns.violinplot(x="run_name", y="result", data=all_results, order=list(run_names.values()))
sns.boxplot(x="run_name", y="result", data=all_results, order=list(run_names.values()))
plt.ylabel(f"{metric}")
plt.xlabel("Model")
plt.title(f"Circuit score ({len(result_set)} prompts), {model_name}")
#%%
sns.displot(x="result", data=all_results, kind="kde", hue="run_name", bw_adjust=0.7)
plt.xlabel(f"{metric}")
plt.title(f"Circuit score ({len(result_set)} prompts), {model_name}")
#%%
for_averages = defaultdict(lambda: defaultdict(list))
for (run_name, prompt_n), json_data in all_json_data.items():
    for k, v in json_data["sweep_pruning_results"].items():
        for_averages[run_name][k].append(v["replacement_score"])
averages = {k: {k2: np.mean(v) for k2, v in v.items()} for k, v in for_averages.items()}
stds = {k: {k2: np.std(v) for k2, v in v.items()} for k, v in for_averages.items()}
plt.figure(figsize=(10, 5))
for run_name, averages in averages.items():
    run_name_str = run_names[run_name]
    plt.plot(averages.keys(), averages.values(), label=run_name_str, marker='o')
    for k in averages.keys():
        std = stds[run_name][k]
        plt.errorbar(k, averages[k], yerr=std, color=plt.gca().lines[-1].get_color(), capsize=3)
plt.legend()
plt.xlabel("Average L0")
plt.ylabel(f"{metric}")
plt.title(f"Circuit score ({len(result_set)} prompts), {model_name}")
#%%
# Generate a scatter plot between all pairs of runs
run_name_list = list(run_names.values())
n_runs = len(run_name_list)

plt.figure(figsize=(4 * n_runs, 4 * n_runs))

comparisons = [
    ("CLT, no skip", "Tied CLT, skip"),
    ("FT PLT, no skip", "PLT, no skip"),
]

for i in range(n_runs):
    for j in range(n_runs):
        if i == j:
            continue
        run_i = run_name_list[i]
        run_j = run_name_list[j]
        if (run_i, run_j) not in comparisons:
            continue
        # Get results for the intersection set, already aligned in 'results'
        x = results[run_i]
        y = results[run_j]
        plt.figure(figsize=(4, 4))
        plt.scatter(x, y, alpha=0.7)
        plt.xlabel(run_i)
        plt.ylabel(run_j)
        plt.title(f"Scatter: {run_j} vs {run_i} ({len(x)} prompts)")
        # Optionally, plot y=x line
        min_val = min(min(x), min(y))
        max_val = max(max(x), max(y))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        plt.tight_layout()
        plt.show()
#%%
