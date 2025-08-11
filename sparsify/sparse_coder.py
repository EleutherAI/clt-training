import json
import os
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.distributed import tensor as dtensor
from torch.distributed.tensor.device_mesh import DeviceMesh

from .config import SparseCoderConfig
from .fused_encoder import NO_COMPILE, EncoderOutput, fused_encoder
from .utils import decoder_impl, load_sharded, save_sharded


@dataclass
class ForwardOutput:
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    is_last: bool = False
    """Whether this is the last target in a multi-target setup."""


class MidDecoder:
    def __init__(
        self,
        sparse_coder: "SparseCoder",
        x: Tensor,
        activations: Tensor,
        indices: Tensor,
        dead_mask: Optional[Tensor] = None,
    ):
        self.sparse_coder = sparse_coder
        self.x = x
        self.latent_acts = activations
        self.latent_indices = indices
        self.dead_mask = dead_mask
        self.index = 0

    def copy(
        self,
        x: Tensor | None = None,
        activations: Tensor | None = None,
        indices: Tensor | None = None,
        dead_mask: Tensor | None = None,
    ):
        if x is None:
            x = self.x
        if activations is None:
            activations = self.latent_acts
        if indices is None:
            indices = self.latent_indices
        if dead_mask is None:
            dead_mask = self.dead_mask
        return MidDecoder(self.sparse_coder, x, activations, indices, dead_mask)

    def detach(self):
        if not hasattr(self, "original_activations"):
            self.original_activations = self.latent_acts
            self.latent_acts = self.latent_acts.detach()
            self.latent_acts.requires_grad = True

    def restore(self, is_last: bool = False):
        grad = self.latent_acts.grad
        assert grad is not None, "Activations have no gradient."
        self.latent_acts = self.original_activations
        self.latent_acts.backward(grad, retain_graph=not is_last)
        del self.original_activations

    def next(self):
        self.index += 1

    def prev(self):
        self.index = max(0, self.index - 1)

    @property
    def will_be_last(self):
        return self.index + 1 >= self.sparse_coder.cfg.n_targets

    @property
    def current_w_dec(self):
        return (
            self.sparse_coder.W_decs[self.index]
            if self.sparse_coder.multi_target
            else self.sparse_coder.W_dec
        )

    @property
    def current_latent_acts(self):
        post_enc = (
            self.sparse_coder.post_encs[self.index]
            if self.sparse_coder.multi_target
            else self.sparse_coder.post_enc
        )

        if self.sparse_coder.cfg.post_encoder_scale:
            post_enc_scale = (
                self.sparse_coder.post_enc_scales[self.index]
                if self.sparse_coder.multi_target
                else self.sparse_coder.post_enc_scale
            )
            if isinstance(post_enc_scale, dtensor.DTensor):
                post_enc_scale = post_enc_scale.to_local()
        else:
            post_enc_scale = None

        latent_acts = self.latent_acts
        if isinstance(latent_acts, dtensor.DTensor):
            latent_acts = latent_acts.to_local()
            latent_indices = self.latent_indices.to_local()
            post_enc = post_enc.to_local()
            latent_acts = latent_acts + post_enc[latent_indices] * (latent_acts > 0)
            if post_enc_scale is not None:
                latent_acts = latent_acts * post_enc_scale[latent_indices]
            latent_acts = dtensor.DTensor.from_local(
                latent_acts,
                self.latent_acts.device_mesh,
                placements=self.latent_acts.placements,
            )
        else:
            latent_acts = latent_acts + post_enc[self.latent_indices] * (
                latent_acts > 0
            )
            if post_enc_scale is not None:
                latent_acts = latent_acts * post_enc_scale[self.latent_indices]
        return latent_acts

    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def __call__(
        self,
        y: Tensor | None,
        index: Optional[int] = None,
        addition: float | Tensor = 0,
        no_extras: bool = False,
        denormalize: bool = True,
        add_post_enc: bool = True,
        loss_mask: Tensor | None = None,
    ) -> ForwardOutput:
        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = self.x
            if isinstance(y, dtensor.DTensor):
                y = y.redistribute(
                    y.device_mesh, (dtensor.Replicate(), dtensor.Shard(1))
                )

        assert isinstance(y, Tensor), "y must be a tensor."
        if add_post_enc:
            latent_acts = self.current_latent_acts
        else:
            latent_acts = self.latent_acts
        if index is None:
            index = self.index
            self.next()
        elif self.sparse_coder.cfg.n_targets > 0:
            assert 0 <= index < self.sparse_coder.cfg.n_targets, "Index out of bounds."
        is_last = self.index >= self.sparse_coder.cfg.n_targets

        # Decode
        if latent_acts is None and self.latent_indices is None:
            sae_out = torch.zeros_like(self.x)
        else:
            latent_indices = self.latent_indices
            sae_out = self.sparse_coder.decode(latent_acts, latent_indices, index)
        W_skip = (
            self.sparse_coder.W_skips[index]
            if hasattr(self.sparse_coder, "W_skips")
            else self.sparse_coder.W_skip
        )
        if W_skip is not None:
            sae_out += self.x.to(self.sparse_coder.dtype) @ W_skip.mT
        sae_out += addition

        if denormalize:
            sae_out = self.sparse_coder.denormalize_output(sae_out)

        if no_extras:
            return ForwardOutput(
                sae_out,
                self.latent_acts,
                self.latent_indices,
                sae_out.new_tensor(0.0),
                sae_out.new_tensor(0.0),
                sae_out.new_tensor(0.0),
                is_last,
            )
        else:
            # Compute the residual
            e = y - sae_out
            if loss_mask is not None:
                e = e * loss_mask[..., None]

            # Used as a denominator for putting everything on a reasonable scale
            if loss_mask is None:
                total_variance = (y - y.mean(0)).pow(2).sum()
            else:
                lm = loss_mask[..., None]
                y_mean = (y * lm).sum(0) / lm.sum(0)
                total_variance = (y - y_mean).pow(2).mul(lm).sum()

            l2_loss = e.pow(2).sum()
            fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            self.latent_acts,
            self.latent_indices,
            fvu,
            is_last,
        )


class SparseCoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
        mesh: Optional[DeviceMesh] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.multi_target = cfg.n_targets > 0 and cfg.transcode
        self.mesh = mesh
        weight_dtype = cfg.torch_dtype
        if weight_dtype is None:
            weight_dtype = dtype
        encoder_dtype = weight_dtype
        decoder_dtype = weight_dtype
        self.encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=encoder_dtype
        )
        self.encoder.bias.data.zero_()
        if mesh is None:
            self.encoder = nn.Linear(
                d_in, self.num_latents, device=device, dtype=encoder_dtype
            )
            self.encoder.bias.data.zero_()
        else:
            self.encoder = nn.Linear(
                d_in,
                self.num_latents // mesh.shape[1],
                device=device,
                dtype=encoder_dtype,
            )
            self.encoder.bias.data.zero_()
            scaling = 1 / self.encoder.weight.shape[1] ** 0.5
            self.encoder.register_parameter(
                "weight",
                nn.Parameter(
                    # default torch initialization
                    dtensor.rand(
                        (self.num_latents, d_in),
                        dtype=encoder_dtype,
                        device_mesh=mesh,
                        placements=[dtensor.Replicate(), dtensor.Shard(0)],
                    )
                    * (2.0 * scaling)
                    - scaling
                ),
            )
            self.encoder.register_parameter(
                "bias",
                nn.Parameter(
                    dtensor.DTensor.from_local(
                        self.encoder.bias.data,
                        mesh,
                        placements=[
                            dtensor.Replicate(),
                            dtensor.Shard(0),
                        ],
                    )
                ),
            )

        if decoder:
            # Transcoder initialization: use zeros
            if cfg.transcode:

                def create_W_dec():
                    num_latents = self.num_latents
                    if self.cfg.coalesce_topk == "per-layer":
                        num_latents *= max(1, cfg.n_sources)
                    if mesh is not None:
                        result = dtensor.zeros(
                            (num_latents, d_in),
                            dtype=decoder_dtype,
                            device_mesh=mesh,
                            placements=[
                                dtensor.Replicate(),
                                dtensor.Shard(1) if cfg.tp_output else dtensor.Shard(0),
                            ],
                        )
                    else:
                        result = torch.zeros(
                            num_latents, d_in, device=device, dtype=decoder_dtype
                        )
                    return nn.Parameter(result)

                if (
                    self.multi_target
                    and self.cfg.coalesce_topk
                    not in (
                        "concat",
                        "per-layer",
                    )
                    and not cfg.per_source_tied
                ):
                    self.W_decs = nn.ParameterList()
                    for i in range(cfg.n_targets):
                        self.W_decs.append(create_W_dec())
                    self.W_dec = self.W_decs[0]
                else:
                    self.W_dec = create_W_dec()

            # Sparse autoencoder initialization: use the transpose of encoder weights
            else:
                self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
                if self.cfg.normalize_decoder:
                    self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        def create_bias():
            if mesh is not None:
                result = dtensor.zeros(
                    (self.d_in,),
                    dtype=dtype,
                    device_mesh=mesh,
                    placements=[
                        dtensor.Replicate(),
                        dtensor.Shard(0) if cfg.tp_output else dtensor.Replicate(),
                    ],
                )
            else:
                result = torch.zeros(self.d_in, device=device, dtype=dtype)
            return nn.Parameter(result)

        def create_W_skip():
            if not cfg.skip_connection:
                return None
            if mesh is not None:
                result = dtensor.zeros(
                    (self.d_in, self.d_in),
                    dtype=dtype,
                    device_mesh=mesh,
                    placements=[dtensor.Replicate(), dtensor.Shard(0)],
                )
            else:
                result = torch.zeros(self.d_in, self.d_in, device=device, dtype=dtype)
            return nn.Parameter(result)

        if self.multi_target and self.cfg.coalesce_topk not in ("concat", "per-layer"):
            self.b_decs = nn.ParameterList()
            self.W_skips = nn.ParameterList()
            for _ in range(cfg.n_targets):
                self.b_decs.append(create_bias())
                self.W_skips.append(create_W_skip())
            self.W_skip = self.W_skips[0]
            self.b_dec = self.b_decs[0]
        else:
            self.b_dec = create_bias()
            self.W_skip = create_W_skip()

        if cfg.normalize_io:
            if mesh is not None:
                self.register_buffer(
                    "in_norm",
                    dtensor.ones(
                        1,
                        dtype=dtype,
                        device_mesh=mesh,
                        placements=[dtensor.Replicate(), dtensor.Replicate()],
                    ),
                )
                self.register_buffer(
                    "out_norm",
                    dtensor.ones(
                        1,
                        dtype=dtype,
                        device_mesh=mesh,
                        placements=[dtensor.Replicate(), dtensor.Replicate()],
                    ),
                )
            else:
                self.register_buffer(
                    "in_norm", torch.ones(1, device=device, dtype=dtype)
                )
                self.register_buffer(
                    "out_norm", torch.ones(1, device=device, dtype=dtype)
                )

        def make_post_enc(is_zeros: bool = True):
            if mesh is not None:
                post_enc = (dtensor.zeros if is_zeros else dtensor.ones)(
                    (self.num_latents,),
                    dtype=dtype,
                    device_mesh=mesh,
                    placements=[dtensor.Replicate(), dtensor.Replicate()],
                )
            else:
                post_enc = (torch.zeros if is_zeros else torch.ones)(
                    self.num_latents, device=device, dtype=dtype
                )
            post_enc = nn.Parameter(post_enc, requires_grad=cfg.train_post_encoder)
            return post_enc

        if self.multi_target:
            self.post_encs = nn.ParameterList()
            for _ in range(cfg.n_targets):
                self.post_encs.append(make_post_enc())
        else:
            self.post_enc = make_post_enc()

        if self.cfg.post_encoder_scale:
            if self.multi_target:
                self.post_enc_scales = nn.ParameterList()
                for i in range(cfg.n_targets):
                    self.post_enc_scales.append(make_post_enc(is_zeros=i > 0))
            else:
                self.post_enc_scale = make_post_enc(is_zeros=False)

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "SparseCoder"]:
        """Load sparse coders for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: SparseCoder.load_from_disk(
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
            f.name: SparseCoder.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
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

        return SparseCoder.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        mesh: Optional[DeviceMesh] = None,
    ) -> "SparseCoder":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = SparseCoder(d_in, cfg, device=device, decoder=decoder, mesh=mesh)
        sae.load_state(path)
        return sae

    def load_state(self, path: os.PathLike, strict: bool = False):
        current_state_dict = self.state_dict()

        filename = str(Path(path) / "sae.safetensors")
        if self.mesh is None:
            state_dict = load_file(
                filename,
                device=str(self.device),
            )
        else:
            state_dict = load_sharded(
                filename,
                current_state_dict,
                self.mesh,
            )
        if hasattr(self, "post_enc") and "post_enc" not in state_dict:
            print("Imputing post_enc")
            state_dict["post_enc"] = self.post_enc.clone()
        if hasattr(self, "post_encs") and not any(
            f"post_encs.{i}" in state_dict for i in range(len(self.post_encs))
        ):
            print("Imputing post_encs")
            for i, post_enc in enumerate(self.post_encs):
                state_dict[f"post_encs.{i}"] = post_enc.clone()
        if hasattr(self, "post_enc_scale") and "post_enc_scale" not in state_dict:
            print("Imputing post_enc_scale")
            state_dict["post_enc_scale"] = self.post_enc_scale.clone()
        if hasattr(self, "post_enc_scales") and not any(
            f"post_enc_scales.{i}" in state_dict
            for i in range(len(self.post_enc_scales))
        ):
            print("Imputing post_enc_scales")
            for i, post_enc_scale in enumerate(self.post_enc_scales):
                if f"post_enc_scales.{i}" not in state_dict:
                    state_dict[f"post_enc_scales.{i}"] = post_enc_scale.clone()
        if hasattr(self, "W_decs") and not any(
            f"W_decs.{i}" in state_dict for i in range(len(self.W_decs))
        ):
            print("Imputing W_decs")
            state_dict["W_decs.0"] = state_dict.pop("W_dec")
            for i, W_dec in enumerate(self.W_decs):
                if i > 0:
                    state_dict[f"W_decs.{i}"] = W_dec.clone()
        if (
            self.cfg.skip_connection
            and hasattr(self, "W_skips")
            and not any(f"W_skips.{i}" in state_dict for i in range(len(self.W_skips)))
        ):
            state_dict["W_skips.0"] = state_dict.pop("W_skip")
            for i, W_skip in enumerate(self.W_skips):
                if i > 0:
                    state_dict[f"W_skips.{i}"] = W_skip.clone()
        if hasattr(self, "b_decs") and not any(
            f"b_decs.{i}" in state_dict for i in range(len(self.b_decs))
        ):
            print("Imputing b_decs")
            state_dict["b_decs.0"] = state_dict.pop("b_dec")
            for i, b_dec in enumerate(self.b_decs):
                if i > 0:
                    state_dict[f"b_decs.{i}"] = b_dec.clone()

        items = [(k, v) for k, v in state_dict.items() if k in current_state_dict]
        current_keys = list(current_state_dict.keys())
        items.sort(key=lambda x: current_keys.index(x[0]))
        state_dict = dict()
        for k, v in items:
            state_dict[k] = v
        self.load_state_dict(state_dict, strict=strict)

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        if (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0:
            path.mkdir(parents=True, exist_ok=True)

        filename = str(path / "sae.safetensors")
        current_state_dict = self.state_dict()
        is_main = save_sharded(current_state_dict, filename, mesh=self.mesh)

        if is_main:
            with open(path / "cfg.json", "w") as f:
                json.dump(
                    {
                        **self.cfg.to_dict(),
                        "d_in": self.d_in,
                    },
                    f,
                )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def normalize_input(self, x: Tensor) -> Tensor:
        if self.cfg.normalize_io:
            return x * (x.shape[-1] ** 0.5 / self.in_norm)
        return x

    def denormalize_output(self, x: Tensor) -> Tensor:
        if self.cfg.normalize_io:
            return x * (self.out_norm / (x.shape[-1] ** 0.5))
        return x

    @torch.compile(disable=NO_COMPILE)
    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        x = self.normalize_input(x)

        if not self.cfg.transcode:
            x = x - self.b_dec

        return fused_encoder(
            x,
            self.encoder.weight,
            self.encoder.bias,
            self.cfg.k,
            self.cfg.activation,
            self.cfg.use_fp8,
        )

    def decode(
        self,
        top_acts: Tensor | dtensor.DTensor,
        top_indices: Tensor | dtensor.DTensor,
        index: int = 0,
    ) -> Tensor:
        W_dec = self.W_decs[index] if hasattr(self, "W_decs") else self.W_dec
        b_dec = self.b_decs[index] if hasattr(self, "b_decs") else self.b_dec

        assert W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), W_dec)
        return y + b_dec

    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        *,
        dead_mask: Tensor | None = None,
        return_mid_decoder: bool = False,
        loss_mask: Tensor | None = None,
    ) -> ForwardOutput | MidDecoder:
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16 if self.cfg.dtype != "float16" else torch.float16,
            enabled=torch.cuda.is_bf16_supported(),
        ):
            top_acts, top_indices = self.encode(x)

            x = self.normalize_input(x)

            mid_decoder = MidDecoder(self, x, top_acts, top_indices, dead_mask)
            if self.multi_target or return_mid_decoder:
                return mid_decoder
            else:
                return mid_decoder(y, 0, loss_mask=loss_mask)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        for W_dec in self.W_decs if self.multi_target else (self.W_dec,):
            assert W_dec is not None, "Decoder weight was not initialized."

            eps = torch.finfo(W_dec.dtype).eps
            norm = torch.norm(W_dec.data, dim=1, keepdim=True)
            W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        for W_dec in self.W_decs if self.multi_target else (self.W_dec,):
            assert W_dec is not None, "Decoder weight was not initialized."
            assert W_dec.grad is not None  # keep pyright happy

            parallel_component = einops.einsum(
                W_dec.grad,
                W_dec.data,
                "d_sae d_in, d_sae d_in -> d_sae",
            )
            W_dec.grad -= einops.einsum(
                parallel_component,
                W_dec.data,
                "d_sae, d_sae d_in -> d_sae d_in",
            )


# Allow for alternate naming conventions
Sae = SparseCoder
