#%%
from safetensors.torch import load_file
from pathlib import Path
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
before = Path(f"../attribution_graph/results/gpt2-curt-clt-tied_per_target_skip_global_batchtopk_jumprelu")
after = Path(f"checkpoints/curt-finetune")
module_name = "h.5.mlp"

before_state_dict = load_file(before / module_name / "sae.safetensors")
after_state_dict = load_file(after / module_name / "sae.safetensors")
#%%
for k, a in after_state_dict.items():
    b = before_state_dict[k]
    print(k)
    print(a.abs().mean(), b.abs().mean(), (a - b).abs().mean())
#%%
