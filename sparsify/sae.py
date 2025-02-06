# PYTHONPATH=hamm/src/python CUDA_VISIBLE_DEVICES=5 python -m sae EleutherAI/pythia-160m togethercomputer/RedPajama-Data-1T-Sample
import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple
import math

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SaeConfig
from .utils import decoder_impl
from .xformers_decoder import xformers_embedding_bag


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""


class PKMLinear(nn.Module):
    def __init__(self,
                 d_in: int, num_latents: int,
                 device: str | torch.device,
                 dtype: torch.dtype | None = None,
                 *,
                 cfg: SaeConfig
                 ):
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents
        if cfg.pkm_pad:
            self.pkm_base = int(2 ** math.ceil(math.log2(num_latents) / 2))
        else:
            self.pkm_base = int(math.ceil(math.sqrt(num_latents)))
        self.cfg = cfg
        self._weight = nn.Linear(d_in, 2 * self.pkm_base, device=device, dtype=dtype)
        self._weight.weight.data *= cfg.pkm_init_scale / 4
        # Orthogonal matrices have the same FVU  as /4, but produce more dead latents
        # torch.nn.init.orthogonal_(self._weight.weight, gain=0.5 / math.sqrt(self.d_in))
        self._scale = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        if cfg.pkm_bias:
            self.bias = nn.Parameter(torch.zeros(self.pkm_base**2, dtype=dtype, device=device))
        
    def forward(self, x):
        x1, x2 = torch.chunk(self._weight(x), 2, dim=-1)
        y = (x1[..., :, None] + x2[..., None, :]).reshape(
            *x.shape[:-1], self.pkm_base**2
        )[..., : self.num_latents]
        return y

    @torch.compile(mode="max-autotune")
    def topk(self, x, k: int):
        orig_batch_size = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        x1, x2 = torch.chunk(self._weight(x), 2, dim=-1)
        if not self.cfg.topk_separate:
            x = (x1[:, :, None] + x2[:, None, :]).view(-1, self.pkm_base**2)
            if self.cfg.pkm_bias:
                x += self.bias
            return x.topk(k, dim=-1)
        k1, k2 = k, k
        w1, i1 = x1.topk(k1, dim=1)
        w2, i2 = x2.topk(k2, dim=1)
        w = w1[:, :, None] + w2[:, None, :]
        i = i1[:, :, None] * self.pkm_base + i2[:, None, :]
        if self.cfg.pkm_bias:
            w = w + self.bias[i]
        w = w.view(-1, k1 * k2)
        w, i = w.topk(k, dim=-1)
        i1 = torch.gather(i1, 1, i // k2)
        i2 = torch.gather(i2, 1, i % k2)
        i = i1 * self.pkm_base + i2
        return w.view(*orig_batch_size, k), i.reshape(*orig_batch_size, k)

    @property
    def weight(self):
        w1, w2 = torch.chunk(self._weight.weight, 2, dim=0)
        pkm_trim = math.ceil(self.num_latents / self.pkm_base)
        w1 = w1[:pkm_trim]
        w1 = w1[:, None, :]
        w2 = w2[None, :, :]
        w1 = w1.expand(-1, w2.shape[1], -1)
        w2 = w2.expand(w1.shape[0], -1, -1)
        return (w1 + w2).reshape(self.pkm_base * pkm_trim, self.d_in)[:self.num_latents] * torch.exp(self._scale)


class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.device = device
        self.dtype = dtype
        if cfg.encoder_halut:
            from halutmatmul.modules import HalutLinear
            self.encoder = HalutLinear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.halut_active[:] = 1
            # TODO properly set parameters. We should have the data at this point
            # Reference:
            # https://github.com/joennlae/halutmatmul/blob/master/src/python/halutmatmul/model.py#L92
            self.encoder.bias.data.zero_()
        elif cfg.encoder_pkm:
            self.encoder = PKMLinear(d_in, self.num_latents, device=device, dtype=dtype, cfg=cfg)
            self.encoder._weight.bias.data.zero_()
        else:
            self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.clone()) if decoder else None
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

        self.W_skip = nn.Parameter(
            torch.zeros(d_in, d_in, device=device, dtype=dtype)
        ) if cfg.skip_connection else None

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "Sae"]:
        """Load SAEs for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: Sae.load_from_disk(
                    repo_path / layer, device=device, decoder=decoder
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: Sae.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return Sae.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)

        return nn.functional.relu(out)

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        if self.cfg.encoder_pkm:
            return self.encoder.topk(x, self.cfg.k)
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        if self.cfg.decoder_xformers:
            y = xformers_embedding_bag(top_indices, self.W_dec, top_acts.to(torch.bfloat16))
        else:
            y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        pre_acts = self.pre_acts(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Decode
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = self.decode(top_acts, top_indices)
        if self.W_skip is not None:
            sae_out += x.to(self.dtype) @ self.W_skip.mT

        # Compute the residual
        e = sae_out - y

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (y - y.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = y.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        if self.cfg.multi_topk:
            top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out = self.decode(top_acts, top_indices)

            multi_topk_fvu = (sae_out - y).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
