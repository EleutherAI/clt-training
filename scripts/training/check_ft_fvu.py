#%%
import wandb


run_id = "9be5btq0"
run = wandb.Api().run(f"eleutherai/sparsify/{run_id}")
fvu_keys = [f"fvu/h.{i}.mlp" for i in range(12)]
history = run.history(keys=fvu_keys)[fvu_keys]
hist = history.mean(axis=1)
hist.iloc[:10].mean()
# %%
