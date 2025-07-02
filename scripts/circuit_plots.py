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
