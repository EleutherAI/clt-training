#%%
import json
import glob
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
#%%
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
eval_path = Path("results/gpt2-eval/")
model_name = "GPT2"
judge_ctx = "j1"
run_names = {
    "bs16-lr2e-4-nonskip-tied-no-affine-ef128-k16-adam8-bf16": "Tied CLT, no skip",
    "bs16-lr2e-4-nonskip-ef128-k16-adam8-bf16": "PLT, no skip",
    "bs8-lr3e-4-tied-ef128-k16": "Tied CLT, skip",
    "bs8-lr2e-4-none-ef128-k16": "PLT, skip",
}
metric = "replacement_score_unpruned"
# metric = "completeness_score_unpruned"
results = defaultdict(dict)
for run_name, run_name_str in run_names.items():
    for results_path in eval_path.glob(f"{judge_ctx}-*-gpt2/{run_name}/results.json"):
        prompt_n = int(results_path.parent.parent.stem.split('-')[1])
        results[run_name_str][prompt_n] = json.load(open(results_path))[metric]
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
# Generate a scatter plot between all pairs of runs
run_name_list = list(run_names.values())
n_runs = len(run_name_list)

plt.figure(figsize=(4 * n_runs, 4 * n_runs))
for i in range(n_runs):
    for j in range(n_runs):
        if i == j:
            continue
        run_i = run_name_list[i]
        run_j = run_name_list[j]
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
