#%%
import wandb


run_id = "fuh07z0g"
run_id = "f4ujn7kf"
run = wandb.Api().run(f"eleutherai/sparsify/{run_id}")
fvu_keys = [f"fvu/h.{i}.mlp" for i in range(12)]
history = run.history(keys=fvu_keys)[fvu_keys]
hist = history.mean(axis=1)
hist.iloc[-10:].mean()
# %%
