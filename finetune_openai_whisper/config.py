"""
Centralised configuration for Whisper fine-tuning.

All training parameters are defined in one place so users have a single
dataclass to customise. Pass a Config instance to prepare_trainer_from_config()
to start training.
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Config:

    # ── Model ──────────────────────────────────────────────────────────────────
    model_name: str = "turbo"
    """Whisper variant: tiny | base | small | medium | large | large-v2 | large-v3 | turbo."""

    lang: str = "ar"
    """
    Language code, e.g. 'ar' for Arabic, 'en' for English. Under task='transcribe'
    this is the transcription language; under task='translate' it is the source
    audio language (the output is always English).
    """

    task: str = "transcribe"
    """
    Whisper task:
      - 'transcribe' → audio to text in the same language (`lang`).
      - 'translate'  → audio to English text. `lang` is the source audio language,
                       and each training example's `text` must be its English
                       translation.

    Translation requires a multilingual model (the *.en variants do not support it).
    """

    # ── Encoder freezing ───────────────────────────────────────────────────────
    # Encoder layout: conv1 → conv2 → blocks[0..N-1] → ln_post
    #
    #  freeze_encoder=False                                → encoder fully trainable
    #  freeze_encoder=True,  num_frozen_encoder_layers=None → freeze entire encoder
    #  freeze_encoder=True,  num_frozen_encoder_layers=0    → freeze conv1, conv2, ln_post only
    #  freeze_encoder=True,  num_frozen_encoder_layers=N    → freeze conv1, conv2, ln_post + first N blocks

    freeze_encoder: bool = True
    """Freeze the encoder during training. True preserves audio representations (recommended)."""

    num_frozen_encoder_layers: Optional[int] = None
    """
    Number of encoder transformer blocks to freeze. Only used when freeze_encoder=True.

    - None  → freeze the entire encoder (conv layers, ln_post, and all blocks)
    - 0     → freeze only conv1, conv2, and ln_post; all transformer blocks remain trainable
    - N > 0 → freeze conv1, conv2, ln_post, and the first N transformer blocks
    """

    # ── Decoder freezing ───────────────────────────────────────────────────────
    # Decoder layout: token_embedding → blocks[0..N-1] → ln
    #
    #  freeze_decoder=False                                → decoder fully trainable
    #  freeze_decoder=True,  num_frozen_decoder_layers=None → freeze entire decoder
    #  freeze_decoder=True,  num_frozen_decoder_layers=0    → freeze token_embedding and ln only
    #  freeze_decoder=True,  num_frozen_decoder_layers=N    → freeze token_embedding, ln + first N blocks

    freeze_decoder: bool = False
    """Freeze the decoder during training. False keeps the full decoder trainable (recommended)."""

    num_frozen_decoder_layers: Optional[int] = None
    """
    Number of decoder transformer blocks to freeze. Only used when freeze_decoder=True.

    - None  → freeze the entire decoder (token_embedding, ln, and all blocks)
    - 0     → freeze only token_embedding and ln; all transformer blocks remain trainable
    - N > 0 → freeze token_embedding, ln, and the first N transformer blocks
    """

    # ── Weight untying ─────────────────────────────────────────────────────────
    untie_weights: bool = False
    """
    Decouple the decoder's output projection from its token embedding weights.

    By default Whisper ties the output logit projection to token_embedding (the
    same weight matrix is used for both). Setting this to True creates an
    independent lm_head layer initialised from a copy of those weights.

    This is most useful when freeze_decoder=True and num_frozen_decoder_layers=0:
    token_embedding stays frozen (preserving vocabulary representations) while
    lm_head continues to adapt during fine-tuning.
    """

    # ── Data ───────────────────────────────────────────────────────────────────
    train_data: str = "YOUR_TRAIN_DATA.jsonl"
    """Path to training JSONL file."""

    val_data: str = "YOUR_VAL_DATA.jsonl"
    """Path to validation JSONL file."""

    min_duration: float = 5.0
    """Minimum audio duration in seconds; shorter clips are skipped."""

    max_duration: float = 30.0
    """Maximum audio duration in seconds; longer clips are skipped."""

    sample_rate: int = 16000
    """Audio sample rate expected by Whisper (do not change)."""

    tmp_folder: Optional[str] = None
    """
    Directory for caching mel spectrograms to disk. Set to a path (e.g.
    'tmp/spectrograms') to enable caching, which speeds up training after
    the first epoch. None disables caching.
    """

    storage_threshold_gb: float = 100.0
    """
    Minimum free disk space in GB required before a spectrogram is written
    to the cache. Caching is silently skipped when free space falls below
    this threshold so training is never interrupted by a full disk.
    """

    # ── Optimizer ──────────────────────────────────────────────────────────────
    learning_rate: float = 1e-5
    """Peak learning rate for AdamW."""

    weight_decay: float = 0.01
    """L2 regularisation applied to all parameters except biases and LayerNorm weights."""

    adam_epsilon: float = 1e-8
    """Epsilon for numerical stability in AdamW."""

    warmup_steps: int = 2000
    """Number of linear warmup steps before the learning rate reaches its peak."""

    gradient_accumulation_steps: int = 1
    """
    Accumulate gradients over this many batches before updating weights.
    Effective batch size = train_batch_size × gradient_accumulation_steps.
    """

    # ── Training loop ──────────────────────────────────────────────────────────
    num_train_epochs: int = 10
    """Maximum number of training epochs."""

    train_dataset_len: int = 0
    """
    Number of training samples, used to compute the LR scheduler's total number
    of steps in WhisperModelModule.setup(). Auto-populated by
    prepare_trainer_from_config() from len(train_dataset) before trainer.fit();
    you normally do not set this manually.
    """

    seed: int = 1415
    """Global random seed for reproducibility."""

    # ── Batch / workers ────────────────────────────────────────────────────────
    train_batch_size: int = 32
    """Number of samples per training batch."""

    val_batch_size: int = 16
    """Number of samples per validation batch."""

    train_num_workers: int = 32
    """CPU worker processes for the training DataLoader."""

    val_num_workers: int = 16
    """CPU worker processes for the validation DataLoader."""

    # ── Hardware ───────────────────────────────────────────────────────────────
    accelerator: str = "auto"
    """
    PyTorch Lightning accelerator. 'auto' selects a GPU if available, else CPU.
    Other options: 'gpu', 'cpu', 'tpu'.
    """

    precision: Union[int, str] = 16
    """
    Training precision. 16 uses 16-bit mixed precision (recommended for speed
    and memory). Other options: 32 for full precision, 'bf16-mixed' on Ampere+ GPUs.

    Accepts both ints (16, 32) and strings ('bf16-mixed', '16-mixed', ...); the
    value is passed straight through to pytorch_lightning.Trainer(precision=...).
    """

    # ── Logging ────────────────────────────────────────────────────────────────
    log_dir: str = "logs/"
    """Root directory for TensorBoard logs and checkpoints."""

    log_every_n_steps: int = 1
    """How often (in optimiser steps) to write metrics to TensorBoard."""

    logger_name: str = "whisper_turbo_v1"
    """Experiment name used as a sub-directory inside log_dir. Change per run."""

    lr_monitor_logging_interval: str = "epoch"
    """How often to log the learning rate: 'step' or 'epoch'."""

    # ── Checkpointing ──────────────────────────────────────────────────────────
    checkpoint_dirpath: str = "logs/checkpoint"
    """Directory where checkpoint files are saved."""

    checkpoint_filename: str = (
        "whisper-finetuned-{epoch:04d}-{val_loss:.5f}-{val_wer:.5f}-{val_cer:.5f}"
    )
    """Filename template for saved checkpoints. Supports PyTorch Lightning metric placeholders."""

    checkpoint_monitor: str = "val_wer"
    """Metric used to rank and keep the best checkpoints."""

    checkpoint_monitor_mode: str = "min"
    """'min' if lower is better (WER/CER/loss), 'max' if higher is better (accuracy)."""

    save_top_k: int = 5
    """Number of best checkpoints to keep on disk."""

    def __post_init__(self):
        valid_tasks = {"transcribe", "translate"}
        if self.task not in valid_tasks:
            raise ValueError(
                f"task must be one of {sorted(valid_tasks)}, got {self.task!r}."
            )
