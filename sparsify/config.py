from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
from simple_parsing import Serializable, list_field


@dataclass
class SparseCoderConfig(Serializable):
    """
    Configuration for training a sparse coder on a language model.
    """

    dtype: Literal["none", "float32", "float16", "bfloat16"] = "none"
    """Data type for sparse coder weights."""

    @property
    def torch_dtype(self):
        if self.dtype == "none":
            return None
        return dict(
            float32=torch.float32,
            float16=torch.float16,
            bfloat16=torch.bfloat16,
        )[self.dtype]

    activation: Literal["groupmax", "topk", "batchtopk"] = "topk"
    """Activation function to use."""

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the sparse coder dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    skip_connection: bool = False
    """Include a linear skip connection."""

    transcode: bool = False
    """Whether we want to predict the output of a module given its input."""

    tp_output: bool = True
    """Whether to shard across the output dimension for the decoder."""

    n_targets: int = 0
    """Number of targets to predict. Only used if `transcode` is True."""

    n_sources: int = 0
    """Number of cross-layer sources writing to this layer. Only used if
    `transcode` is True and `cross_layer` is not 0."""

    normalize_io: bool = False
    """Normalize the input and output of the sparse coder."""

    divide_cross_layer: bool = False
    """Divide the preceding layer skip connections by the number of layers."""

    train_post_encoder: bool = True
    """Train the post-encoder bias."""

    post_encoder_scale: bool = False
    """Train a scale for post-encoder layers."""

    per_source_tied: bool = False
    """Tie decoders for each source layer."""

    secondary_target_tied: bool = False
    """In addition to per-source tying, decode with per-target tying."""

    coalesce_topk: Literal["none", "concat", "per-layer", "group"] = "none"
    """How to combine values and indices across layers."""

    topk_coalesced: bool = False
    """Whether to actually apply topk to the coalesced values."""

    @property
    def do_coalesce_topk(self):
        return self.coalesce_topk != "none"

    use_fp8: bool = False
    """Use FP8 for the sparse coder."""


# Support different naming conventions for the same configuration
SaeConfig = SparseCoderConfig
TranscoderConfig = partial(SparseCoderConfig, transcode=True)


@dataclass
class TrainConfig(Serializable):
    sae: SparseCoderConfig

    batch_size: int = 32
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for training."""

    loss_fn: Literal["ce", "fvu", "kl", "kl-fvu"] = "fvu"
    """Loss function to use for training the sparse coders.

    - `ce`: Cross-entropy loss of the final model logits.
    - `fvu`: Fraction of variance explained.
    - `kl`: KL divergence of the final model logits w.r.t. the original logits.
    - `kl-fvu`: KL divergence of the final model logits w.r.t. the original logits
      plus FVU loss.
    """

    kl_coeff: float = 1.0
    """Coefficient for the KL divergence loss for KL+FVU. 0.0 -> automatic scaling."""

    filter_bos: bool = False
    """Filter out BOS tokens from the dataset for KL loss."""

    remove_first_token: bool = False
    """Remove the first token from each sequence."""

    remove_transcoded_modules: bool = False
    """Don't run modules that are replaced for transcoders with CE loss."""

    optimizer: Literal["adam", "adam8", "muon", "signum"] = "signum"
    """Optimizer to use."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000
    """Number of steps over which to warm up the learning rate. Only used if
    `optimizer` is `adam`."""

    b1: float = 0.9
    """Beta1 for Adam."""

    b2: float = 0.999
    """Beta2 for Adam."""

    force_lr_warmup: bool = False
    """Force the learning rate warmup even if `optimizer` is not `adam`."""

    k_decay_steps: int = 0
    """Number of steps over which to decay the number of active latents. Starts at
    input width * 10 and decays to k. Experimental feature."""

    k_anneal_mul: int = 10
    """How much to increase k by for k-annealing."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    dead_latent_penalty: float = 0.0
    """Regularization penalty for dead latents."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train sparse coders on."""

    hookpoints_in: list[str] = list_field()
    """List of input hookpoints for sparse coders."""

    init_seeds: list[int] = list_field(0)
    """List of random seeds to use for initialization. If more than one, train a sparse
    coder for each seed."""

    layers: list[int] = list_field()
    """List of layer indices to train sparse coders on."""

    per_layer_k: list[int] = list_field()
    """List of k values to use for each layer."""

    layer_stride: int = 1
    """Stride between layers to train sparse coders on."""

    cross_layer: int = 0
    """How many layers ahead to train the sparse coder on.
    If 0, train only on the same layer."""

    tp: int = 1
    """Number of tensor parallel ranks to use."""

    @property
    def distribute_modules(self) -> bool:
        """Whether to distribute the modules across ranks."""
        return self.tp > 1

    save_every: int = 1000
    """Save sparse coders every `save_every` steps."""

    save_best: bool = False
    """Save the best checkpoint found for each hookpoint."""

    finetune: str | None = None
    """Finetune the sparse coders from a pretrained checkpoint."""

    restart_epoch: bool = False
    """Start loading the dataset from the beginning after loading a checkpoint."""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1

    save_dir: str = "checkpoints"

    def __post_init__(self):
        """Validate the configuration."""
        if self.layers and self.layer_stride != 1:
            raise ValueError("Cannot specify both `layers` and `layer_stride`.")

        if not self.init_seeds:
            raise ValueError("Must specify at least one random seed.")
