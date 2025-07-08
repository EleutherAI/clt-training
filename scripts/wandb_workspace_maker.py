# %%
import json
import os
from collections import defaultdict
import wandb

os.chdir(os.path.dirname(os.path.dirname(__file__)))

api = wandb.Api()
run_list = list(
    {
        "gdg44koi",
        "4lhl7yqn",
        "aake0mvy",
        "1kooomeu",
        "sji6x2ga",
        "b2ieqm7o",
        "rwlgvm7x",
        "utqpl4fi",
        "7aduxnf2",
        "aqud4jpl",
        "7dxe86cz",
        "1kooomeu",
        "se4mv1pr",
        "f9234bwu",
        "7qzntyj7",
        "b2ieqm7o",
        "rwlgvm7x",
        "gja6qin3",
        "y39ubwbl"
    }
)
run_lists = defaultdict(list)
histories = {}
fvus = {}
accuracies = {}
kls = {}
id_to_run = {}
for run_id in run_list:
    run = api.run(f"eleutherai/sparsify/{run_id}")
    id_to_run[run_id] = run
    keys = [
        f"fvu/{'h' if 'gpt2' in run.name else 'layers'}.{i}.mlp"
        for i in range(12 if "gpt2" in run.name else 16)
    ]
    history = run.history(keys=keys)
    histories[run_id] = history
    fvus[run_id] = history.iloc[-1, 1:].mean()
    if "gpt2" in run.name:
        path_name = run.name.partition("/")[2]
        kl_eval_path = f"results/gpt2-eval-kl/{path_name}/eval_results.json"
        if not os.path.exists(kl_eval_path):
            continue
        results = json.load(open(kl_eval_path))

        accuracies[run_id] = results["accuracy"]["mean"], results["accuracy"]["std"]
        kls[run_id] = results["kl_divergence"]["mean"], results["kl_divergence"]["std"]
    if "gpt2" in run.name:
        run_lists["gpt2"].append(run_id)
    else:
        run_lists["llama"].append(run_id)
# %%
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "eleutherai"
PROJECT = "sparsify"

# workspace: ws.Workspace = ws.Workspace(
#     name="Cross-layer transcoder training",
#     entity=ENTITY,
#     project=PROJECT,
# )
# workspace.save()
# workspace.url
workspace_url = "https://wandb.ai/eleutherai/sparsify?nw=fp05111wlx9"
workspace = ws.Workspace.from_url(workspace_url)
workspace.sections = []
proper_names = dict(
    gpt2="GPT-2 small",
    llama="LLaMA 1B",
)
for name in ("gpt2", "llama"):
    fvu_keys = [
        f"fvu/{'h' if name == 'gpt2' else 'layers'}.{i}.mlp"
        for i in range(12 if name == "gpt2" else 16)
    ]
    fvu_avg = "(" + " + ".join(f"${{{f}}}" for f in fvu_keys) + f") / {len(fvu_keys)}"
    workspace.sections.append(
        ws.Section(
            name=f"{proper_names[name]} FVU",
            layout_settings=ws.SectionLayoutSettings(
                rows=1,
                columns=1,
            ),
            panel_settings=ws.SectionPanelSettings(
                smoothing_type="average",
                smoothing_weight=100,
            ),
            panels=[
                wr.LinePlot(
                    y=[key] if key is not None else [],
                    custom_expressions=[fvu_avg] if key is None else None,
                    log_x=True,
                    log_y=True,
                    # xaxis_expression="${_step}/${summary:_step}*100000",
                    # range_x=(50000.0, 100000.0),
                    range_x=(80000, 205000) if name == "gpt2" else (20_000, 30_000),
                    max_runs_to_show=30,
                )
                for key in [None] + fvu_keys
            ],
            is_open=True,
        ),
    )
    if name in ("gpt2",):
        kl_table = [
            "| Run | KL | Accuracy |",
            "| --- | --- | --- |",
        ]
        for run in run_lists[name]:
            if run not in kls or run not in accuracies:
                continue
            kl, kl_std = kls[run]
            accuracy, accuracy_std = accuracies[run]
            kl_table.append(
                f"| {id_to_run[run].name} | {kl:.4f} ± {kl_std:.4f} | {accuracy:.4f} ± {accuracy_std:.4f} |"
            )
        kl_table = "\n".join(kl_table)
        workspace.sections.append(
            ws.Section(
                name=f"{proper_names[name]} KL",
                layout_settings=ws.SectionLayoutSettings(
                    rows=1,
                    columns=1,
                ),
                panels=[wr.MarkdownPanel(markdown=kl_table)],
            ),
        )
workspace.runset_settings = ws.RunsetSettings(
    filters=[
        ws.Metric("ID").isin(run_list),
    ]
)
workspace.save()
# %%
