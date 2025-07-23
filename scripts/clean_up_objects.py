# %%
from huggingface_hub import list_models
from pprint import pprint
from itertools import chain

gpt_models = list_models(author='EleutherAI', search='gpt2-')
llama_models = list_models(author='EleutherAI', search='llama1b-')
models = chain(gpt_models, llama_models)
filtered = [m.modelId for m in models]
pprint(filtered)

# %%
from huggingface_hub import HfApi
api = HfApi()
candidates = ['EleutherAI/gpt2-curt-clt-tied_per_target_skip_global_batchtopk_jumprelu',
 'EleutherAI/gpt2-clt-none-ef128-k16',
 'EleutherAI/gpt2-clt-tied-ef128-k16',
 'EleutherAI/gpt2-plt-noskip-ef128-k16',
 'EleutherAI/gpt2-curt-clt-untied_global_batchtopk_jumprelu',
 'EleutherAI/gpt2-mntss-transcoder-clt-relu-sp10-1b',
 'EleutherAI/gpt2-mntss-transcoder-clt-relu-sp8',
 'EleutherAI/gpt2-mntss-transcoder-relu-sp6-skip',
 'EleutherAI/gpt2-clt-source-tied-ef128-k16',
 'EleutherAI/gpt2-plt-ef128-ksweep',
 'EleutherAI/gpt2-clt-noskip-ef128-k16',
 'EleutherAI/gpt2-curt-clt-untied-layerwise-tokentopk',
 'EleutherAI/gpt2-plt-ef512-k16',
 'EleutherAI/gpt2-curt-clt-untied_layerwise_batchtopk_jumprelu',
 'EleutherAI/llama1b-clt-tied-ef64-k16',
 'EleutherAI/llama1b-clt-none-ef64-k16',
 'EleutherAI/llama1b-plt-no-skip-ef64-k32',
 'EleutherAI/llama1b-plt-skip-ef64-k32']

for candidate in candidates:
    try:
        refs = api.list_repo_refs(candidate)
    except Exception as e:
        print(f"Error listing revisions for {candidate}: {e}")
        continue
    branches = refs.branches
    for branch in branches:
        files = api.list_repo_files(candidate, revision=branch.name)
        for file in files:
            if file.startswith("optimizer_") and file.endswith(".pt"):
                print("Deleting", f"{candidate}/{branch.name}/{file}")
                try:
                    api.delete_file(file, candidate, revision=branch.name)
                    print(f"Deleted {file} in {candidate}")
                except Exception as e:
                    print(f"Failed to delete {file} in {candidate}: {e}")

# %%
