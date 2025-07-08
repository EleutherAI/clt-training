#%%
from sparsify import SparseCoder
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_to_convert = "checkpoints/gpt2-sweep/bs16-lr2e-4-btopk-clt-noskip-ef128-k16-adam8"
coders = SparseCoder.load_many(model_to_convert, local=True, device="cuda")
#%%
from sparsify.__main__ import load_artifacts, RunConfig


cfg = RunConfig.load(model_to_convert + "/config.json")
model, dataset = load_artifacts(cfg, 0)
#%%
from sparsify.runner import CrossLayerRunner
import torch
from functools import partial
from collections import OrderedDict

def sae_hook(module, input, output, *, sae, module_name):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    input = input.reshape(-1, input.shape[-1])
    output = output.reshape(-1, output.shape[-1])
    encodings = runner.encode(input, sae)
    print((encodings.latent_acts > 0).float().mean().item())
    decoded = runner.decode(encodings, output, module_name)

torch.set_grad_enabled(False)
runner = CrossLayerRunner()
for m in model.modules():
    m._forward_hooks = OrderedDict()
handles = [
    model.get_submodule(hookpoint).register_forward_hook(partial(sae_hook, module_name=hookpoint, sae=sae))
    for hookpoint, sae in coders.items()
]

for sequence in dataset:
    input_ids = sequence["input_ids"].cuda()
    model(input_ids)
    runner.reset()
#%%
