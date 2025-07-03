#%%
from safetensors.numpy import load_file
from pathlib import Path
import numpy as np
import re
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
weight_path = Path("checkpoints/gpt2-sweep/bs16-lr2e-4-btopk-clt-noskip-ef128-k16-adam8")
n_layers = 12

w_dec_from_to = {}
for source_module in weight_path.glob("*.mlp"):
    layer_idx = int(re.match(r"h\.(\d+)\.mlp", source_module.name).group(1))
    weights = load_file(source_module / "sae.safetensors")
    w_dec = weights["W_dec"]
    w_decs = np.split(w_dec, layer_idx + 1, axis=0)
    for source_layer_idx, w_dec in enumerate(w_decs):
        w_dec_from_to[(source_layer_idx, layer_idx)] = w_dec
#%%
source_layer_idx = 3
np.random.seed(0)
selected_features = np.random.randint(0, 768 * 128, size=100)
feats_over_time = []
for target_layer_idx in range(source_layer_idx, n_layers):
    w_dec = w_dec_from_to[(source_layer_idx, target_layer_idx)]
    feats_over_time.append(w_dec[selected_features])
feature_evolution = np.stack(feats_over_time)
#%%
from sklearn.decomposition import PCA
# from matplotlib import pyplot as plt
feats = feature_evolution[:, 9]

# feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)

# print(np.square(np.mean(feats, axis=0)).sum() * len(feats) / np.square(feats).sum())
print(np.square(feats - feats.mean(axis=0)).sum() / np.square(feats).sum())

# pca = PCA()
# pca.fit(feats)
# # print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_[:4].sum())
# s = np.linalg.svd(feats - feats.mean(axis=0)).S ** 2
s = np.linalg.svd(feats - feats.mean(axis=0)).S ** 2
print((s / s.sum())[:1].sum())
# s = np.linalg.svd(feats - feats[0]).S ** 2
from matplotlib import pyplot as plt
pca = PCA(n_components=2)
transed = pca.fit_transform(feats)
# layer_idx = np.arange(len(transed))
colors = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    [1, 1, 1],
])
plt.scatter(*transed.T, c=colors[:len(transed)])
