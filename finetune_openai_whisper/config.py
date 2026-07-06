"""
Centralised configuration for Whisper fine-tuning.

All training parameters are defined in one place so users have a single
dataclass to customise. Pass a Config instance to prepare_trainer_from_config()
to start training.
"""

import json
import textwrap
from dataclasses import dataclass, asdict, fields
from typing import Optional, Union


# Border glyph sets. ascii=True is for logs/CI/terminals that mangle Unicode
# box-drawing characters.
_STYLES = {
    False: dict(tl="╔", tr="╗", bl="╚", br="╝", h="═", v="║", ml="╠", mr="╣"),
    True:  dict(tl="+", tr="+", bl="+", br="+", h="-", v="|", ml="+", mr="+"),
}


def _render_box(title, blocks, width=None, ascii=False, max_width=78, min_width=44):
    """
    Render a monospaced, box-drawn panel from a list of blocks.

    Each block is one of:
      ("section", name)   → a labeled horizontal divider
      ("row", head, val)  → an aligned "head value" line; ``val`` wraps under
                            the value column when it is too long to fit
      ("text", string)    → a full-width free line (wraps)

    Every line is padded to the same width so the right border always aligns.
    When ``width`` is None the box auto-sizes to its content, clamped between
    ``min_width`` and ``max_width`` (long values then wrap instead of forcing a
    very wide box).

    Note: assumes a monospaced font and single-width characters. Double-width
    glyphs (emoji, CJK) count as one in len() but render two columns wide, which
    would push the right border out of alignment.
    """
    s = _STYLES[bool(ascii)]

    # Auto-size from the widest natural line, capped so long values wrap.
    natural = [len(title)]
    for b in blocks:
        if b[0] == "row":
            natural.append(len(b[1]) + len(str(b[2])))
        elif b[0] == "text":
            natural.append(len(b[1]))
        elif b[0] == "section":
            natural.append(len(b[1]) + 4)
    if width is None:
        width = max(min_width, min(max(natural) + 4, max_width))

    inner  = width - 2       # columns between the vertical borders
    text_w = inner - 2       # usable width, one-space margin each side

    top    = s["tl"] + s["h"] * inner + s["tr"]
    bottom = s["bl"] + s["h"] * inner + s["br"]

    def content(line):
        return s["v"] + " " + line.ljust(text_w)[:text_w] + " " + s["v"]

    def section(name):
        seg = s["h"] + " " + name + " "
        return s["ml"] + seg + s["h"] * max(0, inner - len(seg)) + s["mr"]

    out = [top, content(title.center(text_w))]
    if blocks and blocks[0][0] != "section":
        out.append(s["ml"] + s["h"] * inner + s["mr"])
    for b in blocks:
        if b[0] == "section":
            out.append(section(b[1]))
        elif b[0] == "text":
            for ln in textwrap.wrap(b[1], text_w, break_long_words=True,
                                    break_on_hyphens=False) or [""]:
                out.append(content(ln))
        elif b[0] == "row":
            head, val = b[1], str(b[2])
            for ln in textwrap.wrap(
                val, text_w, initial_indent=head,
                subsequent_indent=" " * len(head),
                break_long_words=True, break_on_hyphens=False,
            ) or [head]:
                out.append(content(ln))
    out.append(bottom)
    return "\n".join(out)


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

    def to_dict(self) -> dict:
        """Return this configuration as a plain dict (all fields)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return this configuration as a JSON string, e.g. to log alongside a run."""
        return json.dumps(asdict(self), indent=indent, default=str)

    def summary(self, width=None, ascii=False, only_changed=False, max_width=78) -> str:
        """
        Return a box-drawn summary of this configuration.

        Fields are grouped into labeled sections with their colons vertically
        aligned; a leading ``*`` marks any value that differs from the dataclass
        default. A few derived quantities (effective batch size, steps/epoch,
        total steps) are shown too.

        Args:
            width:        Fixed outer width. None (default) auto-sizes to content.
            ascii:        Use +/-/| borders instead of Unicode (safer in logs/CI).
            only_changed: Show only fields that differ from their defaults.
            max_width:    Cap for auto-sizing; longer values wrap instead.

        Call as ``print(cfg.summary())``.
        """
        defaults = {f.name: f.default for f in fields(self)}

        def is_changed(*names):
            return any(getattr(self, n) != defaults.get(n) for n in names)

        def fmt(v):
            if isinstance(v, bool):
                return "True" if v else "False"   # kept greppable (not ✓/✗)
            if isinstance(v, float):
                return f"{v:g}"
            return str(v)

        eff = self.train_batch_size * self.gradient_accumulation_steps
        if self.train_dataset_len:
            per_epoch = (self.train_dataset_len // self.train_batch_size
                         // self.gradient_accumulation_steps)
            per_epoch_s = str(per_epoch)
            total_s = str(int(per_epoch * self.num_train_epochs))
        else:
            per_epoch_s = total_s = "n/a (set at fit time)"

        # (label, value, field-names-backing-it or None for derived rows)
        sections = [
            ("Model & data", [
                ("Model", self.model_name, ("model_name",)),
                ("Language", self.lang, ("lang",)),
                ("Task", self.task, ("task",)),
                ("Train data", self.train_data, ("train_data",)),
                ("Val data", self.val_data, ("val_data",)),
                ("Duration filter", f"{fmt(self.min_duration)}-{fmt(self.max_duration)} s",
                 ("min_duration", "max_duration")),
                ("Cache folder", self.tmp_folder, ("tmp_folder",)),
            ]),
            ("Freezing", [
                ("Freeze encoder", self.freeze_encoder, ("freeze_encoder",)),
                ("Frozen enc layers", self.num_frozen_encoder_layers, ("num_frozen_encoder_layers",)),
                ("Freeze decoder", self.freeze_decoder, ("freeze_decoder",)),
                ("Frozen dec layers", self.num_frozen_decoder_layers, ("num_frozen_decoder_layers",)),
                ("Untie weights", self.untie_weights, ("untie_weights",)),
            ]),
            ("Optimization", [
                ("Epochs", self.num_train_epochs, ("num_train_epochs",)),
                ("Learning rate", self.learning_rate, ("learning_rate",)),
                ("Warmup steps", self.warmup_steps, ("warmup_steps",)),
                ("Weight decay", self.weight_decay, ("weight_decay",)),
                ("Batch / GPU", self.train_batch_size, ("train_batch_size",)),
                ("Grad accum", self.gradient_accumulation_steps, ("gradient_accumulation_steps",)),
                ("Effective batch", eff, None),
                ("Steps / epoch", per_epoch_s, None),
                ("Total steps", total_s, None),
                ("Precision", self.precision, ("precision",)),
            ]),
            ("Hardware & logging", [
                ("Accelerator", self.accelerator, ("accelerator",)),
                ("Log dir", self.log_dir, ("log_dir",)),
                ("Run name", self.logger_name, ("logger_name",)),
                ("Checkpoint dir", self.checkpoint_dirpath, ("checkpoint_dirpath",)),
                ("Monitor", f"{self.checkpoint_monitor} "
                            f"({self.checkpoint_monitor_mode}, top {self.save_top_k})",
                 ("checkpoint_monitor", "checkpoint_monitor_mode", "save_top_k")),
            ]),
        ]

        # Align all colons to the widest visible label.
        label_w = max(len(lbl) for _, rows in sections for lbl, _, _ in rows)

        blocks, shown_any = [], False
        for name, rows in sections:
            sec_blocks = []
            for lbl, val, names in rows:
                changed = bool(names) and is_changed(*names)
                if only_changed and not changed:
                    continue
                mark = "*" if changed else " "
                head = f"{mark} {lbl.ljust(label_w)} : "
                sec_blocks.append(("row", head, fmt(val)))
            if sec_blocks:
                blocks.append(("section", name))
                blocks.extend(sec_blocks)
                shown_any = True

        if only_changed and not shown_any:
            blocks.append(("text", "(all settings at defaults)"))

        box = _render_box("Whisper Finetuning — Configuration", blocks,
                          width=width, ascii=ascii, max_width=max_width)
        return box + "\n  * = changed from default"
