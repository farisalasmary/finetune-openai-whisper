# finetune-openai-whisper

A complete, production-ready pipeline for fine-tuning [OpenAI's Whisper](https://github.com/openai/whisper) ASR model on custom datasets using [PyTorch Lightning](https://lightning.ai/). Supports both **transcription** (audio → text in the same language) and **translation** (audio → English text).

> **Version 0.1.1**

## What's New in 0.1.1

- **Translation fine-tuning** — a new `task` option lets you fine-tune for `"translate"` (audio → English) in addition to `"transcribe"`. See [Transcription vs. Translation](#transcription-vs-translation).
- **Config introspection** — `Config.summary()` prints an aligned, sectioned overview of a run (with a `*` next to any non-default value), and `Config.to_dict()` / `Config.to_json()` export the resolved config for logging and diffing. See [Inspecting & Logging the Config](#inspecting--logging-the-config).
- **CPU-safe validation** — validation now decodes autoregressively and picks fp16 only on CUDA (fp32 elsewhere), so validation no longer errors on CPU.


## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Command-Line Training](#command-line-training)
- [Transcription vs. Translation](#transcription-vs-translation)
- [Inspecting & Logging the Config](#inspecting--logging-the-config)
- [Configuration Reference](#configuration-reference)
  - [Model](#model)
  - [Freezing Strategy](#freezing-strategy)
  - [Weight Untying](#weight-untying)
  - [Data](#data)
  - [Optimizer](#optimizer)
  - [Training Loop](#training-loop)
  - [Hardware](#hardware)
  - [Logging](#logging)
  - [Checkpointing](#checkpointing)
- [Freezing Strategies In Depth](#freezing-strategies-in-depth)
- [Weight Untying In Depth](#weight-untying-in-depth)
- [Spectrogram Caching](#spectrogram-caching)
- [Monitoring Training](#monitoring-training)
- [Loading a Checkpoint for Inference](#loading-a-checkpoint-for-inference)
- [Converting to Official Whisper Format](#converting-to-official-whisper-format)
- [Converting to Hugging Face Format](#converting-to-hugging-face-format)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Features

- **All Whisper variants supported** — `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `turbo`
- **Transcribe or translate** — fine-tune for same-language transcription or audio → English translation via a single `task` switch
- **Configurable freezing** — freeze the full encoder, a specific number of encoder/decoder transformer blocks, or nothing at all
- **Weight untying** — optionally decouple the decoder's output projection from its token embedding so `lm_head` can adapt while `token_embedding` stays frozen
- **Spectrogram caching** — optionally cache mel spectrograms to disk so subsequent epochs load instantly
- **WER & CER evaluation** — word and character error rates computed and logged at every validation step
- **Config introspection** — a printable, sectioned run summary plus JSON export for reproducible logging
- **TensorBoard integration** — all metrics streamed live during training
- **Checkpoint management** — automatically keeps the top-K best checkpoints ranked by validation WER
- **Multi-GPU / DDP ready** — includes the required fix for Whisper's sparse `alignment_heads` buffer
- **Single config object** — every tunable parameter lives in one `Config` dataclass; no scattered hard-coded values

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- A CUDA-capable GPU is strongly recommended

Translation (`task="translate"`) requires a **multilingual** model — the `*.en` variants are English-only and are not offered here, but note that translation is undefined for English-only checkpoints in general.

All Python dependencies are installed automatically (see [Installation](#installation)).

---

## Installation

Install the latest release from PyPI:

```bash
pip install finetune-openai-whisper
```

Or install directly from source for the latest development version:

```bash
git clone https://github.com/farisalasmary/finetune-openai-whisper
cd finetune-openai-whisper
pip install -e .
```

---

## Data Preparation

Your training and validation data must be in **JSONL** format — one JSON object per line, where each object describes a single audio segment:

```jsonl
{"utt": "spk01_utt001", "audio_filepath": "/data/audio/spk01_utt001.wav", "text": "hello world", "duration": 3.2, "offset": 0.0}
{"utt": "spk01_utt002", "audio_filepath": "/data/audio/spk01_utt002.wav", "text": "how are you", "duration": 4.7, "offset": 0.0}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `utt` | `str` | Unique utterance ID. Used as the cache filename when spectrogram caching is enabled. |
| `audio_filepath` | `str` | Absolute or relative path to the audio file (WAV, MP3, FLAC, etc.). |
| `text` | `str` | Target text for this segment. For `task="transcribe"` this is the transcription in `lang`; for `task="translate"` this is the **English** translation. |
| `duration` | `float` | Duration of the audio segment in seconds. Used for duration filtering. |
| `offset` | `float` | Start time offset within the file in seconds. Use `0.0` if the file contains only this segment. |

### Duration Filtering

Segments shorter than `min_duration` or longer than `max_duration` are automatically skipped before training. The total sample count and hours are printed before and after filtering so you can verify your dataset.

---

## Quick Start

```python
from finetune_openai_whisper import Config
from finetune_openai_whisper.helpers import prepare_trainer_from_config

cfg = Config(
    model_name="turbo",
    lang="ar",
    task="transcribe",           # or "translate" for audio -> English
    train_data="data/train.jsonl",
    val_data="data/val.jsonl",
)

trainer, model, train_dl, val_dl = prepare_trainer_from_config(cfg)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
```

That's it. `prepare_trainer_from_config` will:
1. Load and filter your datasets
2. Initialise the Whisper model with the freezing strategy defined in `cfg`
3. Apply the DDP sparse tensor fix automatically
4. Return a configured `Trainer`, `LightningModule`, and the two dataloaders, ready for `trainer.fit(...)`

---

## Command-Line Training

A complete, ready-to-run CLI wrapper lives at [`examples/train.py`](https://github.com/farisalasmary/finetune-openai-whisper/blob/main/examples/train.py). It exposes every common option as a flag, groups all outputs of a run under one directory, prints a config summary, and archives the resolved config as JSON.

```bash
python examples/train.py \
    --train_data data/train.jsonl \
    --val_data   data/val.jsonl \
    --model turbo --lang ar \
    --output_dir runs/whisper_ar_v1 \
    --run_name ar_v1
```

Every run produces a single self-contained directory:

```
runs/whisper_ar_v1/
├── logs/<run_name>/      TensorBoard event files
├── checkpoints/          best .ckpt files (ranked by val_wer)
├── config.json           resolved configuration (machine-readable)
└── config_summary.txt    resolved configuration (human-readable)
```

Tip: run it with `python -i examples/train.py ...` to drop into an interactive shell after training, with `cfg`, `trainer`, `model`, `train_dl`, and `val_dl` still available for inspection. See all options with `python examples/train.py --help`.

A convenience wrapper with a full invocation is provided at [`examples/run_train.sh`](https://github.com/farisalasmary/finetune-openai-whisper/blob/main/examples/run_train.sh).

<details>
<summary><b>Show the full <code>examples/train.py</code></b></summary>

```python
"""
train.py
────────
Finetune OpenAI Whisper on a custom speech dataset using
finetune-openai-whisper (PyTorch Lightning backend).

Prepare two JSON Lines files first — one for training and one for validation —
where each line describes a single utterance (audio path, text, duration, ...).
See the finetune-openai-whisper docs for the exact JSONL schema.

Quick start:
    pip install finetune-openai-whisper
    python train.py --train_data train.jsonl --val_data val.jsonl
    python train.py --model large-v3 ...         # larger model
    python train.py --lang ar ...                # pick the language
    python train.py --no_freeze_encoder ...      # full model finetuning
    python train.py --task translate ...         # audio -> English text
    python train.py --help                       # all options

All outputs for a run live under a single root directory (--output_dir):
    <output_dir>/
    ├── logs/<run_name>/      TensorBoard event files
    ├── checkpoints/          best .ckpt files (ranked by val_wer)
    ├── config.json           resolved configuration (machine-readable)
    └── config_summary.txt    resolved configuration (human-readable)

Tip: run with `python -i train.py ...` to be dropped into an interactive shell
after training, with `cfg`, `trainer`, `model`, `train_dl`, and `val_dl` still
available for inspection.

After training, convert the best checkpoint to standard Whisper format:
    python -m finetune_openai_whisper.convert_ckpt_to_official_whisper_format \
        <model> <checkpoint.ckpt> whisper_finetuned.pt
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Finetune OpenAI Whisper on a custom dataset "
                    "using finetune-openai-whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("Data")
    data.add_argument("--train_data", default="data/train.jsonl",
                      help="Path to the training JSONL")
    data.add_argument("--val_data", default="data/val.jsonl",
                      help="Path to the validation JSONL")
    data.add_argument("--min_duration", type=float, default=1.0,
                      help="Minimum audio duration in seconds")
    data.add_argument("--max_duration", type=float, default=30.0,
                      help="Maximum audio duration in seconds")
    data.add_argument("--cache_dir", default=None,
                      help="Directory for the spectrogram cache. None = disabled. "
                           "Speeds up epoch 2+ significantly")

    model = p.add_argument_group("Model")
    model.add_argument("--model", default="turbo",
                       choices=["tiny", "base", "small", "medium",
                                "large", "large-v2", "large-v3", "turbo"],
                       help="Whisper model variant")
    model.add_argument("--task", default="transcribe",
                       choices=["transcribe", "translate"],
                       help="transcribe = audio -> text in --lang; "
                            "translate = audio -> English text (needs a "
                            "multilingual model)")
    model.add_argument("--lang", default="en",
                       help="Language code. Under --task transcribe this is the "
                            "transcription language; under --task translate it is "
                            "the source audio language (output is always English)")

    freeze = p.add_argument_group("Freezing")
    freeze.add_argument("--no_freeze_encoder", action="store_true",
                        help="Unfreeze the entire encoder (default: encoder frozen)")
    freeze.add_argument("--num_frozen_encoder_layers", type=int, default=None,
                        help="Freeze only the first N encoder blocks. "
                             "None = freeze all. Requires the encoder to be frozen")
    freeze.add_argument("--freeze_decoder", action="store_true", default=False,
                        help="Freeze the decoder (default: decoder trainable)")
    freeze.add_argument("--num_frozen_decoder_layers", type=int, default=None,
                        help="Freeze only the first N decoder blocks. "
                             "None = freeze all. Requires --freeze_decoder")
    freeze.add_argument("--untie_weights", action="store_true", default=False,
                        help="Decouple lm_head from token_embedding so the output "
                             "projection can adapt independently")

    opt = p.add_argument_group("Optimiser")
    opt.add_argument("--epochs", type=int, default=50)
    opt.add_argument("--train_batch", type=int, default=16)
    opt.add_argument("--val_batch", type=int, default=8)
    opt.add_argument("--lr", type=float, default=1e-5)
    opt.add_argument("--warmup_steps", type=int, default=500)
    opt.add_argument("--grad_accum", type=int, default=2,
                     help="Gradient accumulation steps. "
                          "Effective batch = train_batch x grad_accum")
    opt.add_argument("--weight_decay", type=float, default=0.01)
    opt.add_argument("--seed", type=int, default=1415,
                     help="Random seed for reproducibility")

    hw = p.add_argument_group("Hardware")
    hw.add_argument("--accelerator", default="auto",
                    help="PyTorch Lightning accelerator: auto | gpu | cpu")
    hw.add_argument("--precision", default="16",
                    help="Training precision: 16 | 32 | bf16-mixed")

    logg = p.add_argument_group("Logging & checkpointing")
    logg.add_argument("--output_dir", default="whisper_finetuned_v1",
                      help="Root run directory; holds logs/ and checkpoints/")
    logg.add_argument("--run_name", default="finetune_v1",
                      help="Experiment name — TensorBoard sub-directory under logs/")
    logg.add_argument("--save_top_k", type=int, default=3,
                      help="Keep the top-K checkpoints by val_wer")
    logg.add_argument("--train_workers", type=int, default=4)
    logg.add_argument("--val_workers", type=int, default=2)

    return p.parse_args()


def build_config(args):
    """Translate CLI args into a finetune_openai_whisper.Config object."""
    from finetune_openai_whisper import Config

    # The library accepts precision as an int (16, 32) or a str ('bf16-mixed').
    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    # Everything this run produces lives under a single root directory.
    log_dir        = os.path.join(args.output_dir, "logs")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    cfg = Config(
        # ── Model ─────────────────────────────────────────────────────────────
        model_name = args.model,
        lang       = args.lang,
        task       = args.task,

        # ── Data ──────────────────────────────────────────────────────────────
        train_data   = args.train_data,
        val_data     = args.val_data,
        min_duration = args.min_duration,
        max_duration = args.max_duration,
        tmp_folder   = args.cache_dir,

        # ── Freezing ──────────────────────────────────────────────────────────
        freeze_encoder            = not args.no_freeze_encoder,
        num_frozen_encoder_layers = args.num_frozen_encoder_layers,
        freeze_decoder            = args.freeze_decoder,
        num_frozen_decoder_layers = args.num_frozen_decoder_layers,
        untie_weights             = args.untie_weights,

        # ── Training ──────────────────────────────────────────────────────────
        num_train_epochs            = args.epochs,
        train_batch_size            = args.train_batch,
        val_batch_size              = args.val_batch,
        learning_rate               = args.lr,
        warmup_steps                = args.warmup_steps,
        gradient_accumulation_steps = args.grad_accum,
        weight_decay                = args.weight_decay,
        train_num_workers           = args.train_workers,
        val_num_workers             = args.val_workers,
        seed                        = args.seed,

        # ── Hardware ──────────────────────────────────────────────────────────
        accelerator = args.accelerator,
        precision   = precision,

        # ── Logging & checkpointing ───────────────────────────────────────────
        log_dir                 = log_dir,
        logger_name             = args.run_name,
        save_top_k              = args.save_top_k,
        checkpoint_dirpath      = checkpoint_dir,
        checkpoint_monitor      = "val_wer",
        checkpoint_monitor_mode = "min",
        storage_threshold_gb    = 30.0,
    )
    return cfg


def warn_freeze_conflicts(args):
    """Flag freeze options that would be silently ignored, before training starts."""
    if args.no_freeze_encoder and args.num_frozen_encoder_layers is not None:
        print("[WARN] --num_frozen_encoder_layers is ignored because "
              "--no_freeze_encoder unfreezes the whole encoder.")
    if not args.freeze_decoder and args.num_frozen_decoder_layers is not None:
        print("[WARN] --num_frozen_decoder_layers is ignored because the decoder "
              "is trainable; pass --freeze_decoder to apply it.")


# ── Run ────────────────────────────────────────────────────────────────────────
# Kept at module scope (not inside a main()) so that `python -i train.py ...`
# leaves cfg / trainer / model / train_dl / val_dl available at the prompt.
if __name__ == "__main__":
    args = parse_args()

    # Import here so `--help` works even without the library installed.
    try:
        from finetune_openai_whisper.helpers import prepare_trainer_from_config
    except ImportError:
        raise SystemExit(
            "\n[ERROR] finetune-openai-whisper is not installed.\n"
            "  Run:  pip install finetune-openai-whisper\n"
        )

    # Sanity-check the data files exist before doing any heavy work.
    for _path, _label in [(args.train_data, "train_data"), (args.val_data, "val_data")]:
        if not os.path.isfile(_path):
            raise SystemExit(
                f"\n[ERROR] {_label} not found: {_path}\n"
                f"  Generate your train/val JSONL files first.\n"
            )

    warn_freeze_conflicts(args)

    cfg = build_config(args)

    # Create the run directory tree up front.
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dirpath, exist_ok=True)

    # Build trainer, model, and dataloaders. This also populates
    # cfg.train_dataset_len, so the summary can show real step counts.
    print("Building trainer, model, and dataloaders …")
    trainer, model, train_dl, val_dl = prepare_trainer_from_config(cfg)

    # Show the resolved configuration and archive it next to the run
    # (JSON for tooling/diffing, plain text for humans / log files).
    print(cfg.summary())
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())
    with open(os.path.join(args.output_dir, "config_summary.txt"), "w") as f:
        f.write(cfg.summary(ascii=True))

    # Train.
    print(f"\nStarting training …  (TensorBoard: tensorboard --logdir {cfg.log_dir})\n")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print("\n✓ Training complete.")
    print(f"  Checkpoints saved in: {cfg.checkpoint_dirpath}")
    print(
        "\n  To convert the best checkpoint to standard Whisper format:\n"
        "    python -m finetune_openai_whisper.convert_ckpt_to_official_whisper_format \\\n"
        f"        {args.model} <checkpoint.ckpt> whisper_finetuned.pt\n"
    )
```

</details>

---

## Transcription vs. Translation

Whisper supports two tasks, selected by the `task` field:

| `task` | Input → Output | `lang` means | `text` in your JSONL |
|---|---|---|---|
| `"transcribe"` (default) | audio → text in the same language | the transcription language | the transcription |
| `"translate"` | audio → **English** text | the **source** audio language | the **English** translation |

Whisper's `translate` task only ever produces English, so translation fine-tuning is audio-in-any-language → English-text-out. To fine-tune for translation:

```python
cfg = Config(
    model_name="turbo",   # must be a multilingual model
    lang="ar",            # source audio language
    task="translate",     # output is English
    train_data="data/train.jsonl",  # each "text" is the English translation
    val_data="data/val.jsonl",
)
```

Or from the command line:

```bash
python examples/train.py --task translate --lang ar \
    --train_data data/train.jsonl --val_data data/val.jsonl
```

Requesting `task="translate"` on an English-only model raises a clear error at construction time.

---

## Inspecting & Logging the Config

`Config` can print itself as an aligned, sectioned summary. A leading `*` marks any value that differs from the dataclass default, and derived quantities (effective batch size, steps/epoch, total steps) are shown too:

```python
print(cfg.summary())              # full, auto-width box
print(cfg.summary(only_changed=True))  # only fields you changed this run
print(cfg.summary(ascii=True))    # +/-/| borders, safe for log files / CI
```

```text
╔══════════════════════════════════════════════════════════╗
║              Whisper Finetuning — Configuration          ║
╠═ Model & data ═══════════════════════════════════════════╣
║   Model             : turbo                              ║
║   Language          : ar                                 ║
║   Task              : transcribe                         ║
║ * Train data        : data/train.jsonl                   ║
╠═ Optimization ═══════════════════════════════════════════╣
║ * Epochs            : 50                                 ║
║   Effective batch   : 64                                 ║
║   Steps / epoch     : n/a (set at fit time)              ║
║   Precision         : 16                                 ║
╚══════════════════════════════════════════════════════════╝
  * = changed from default
```

> The `Steps / epoch` and `Total steps` rows read `n/a` until `prepare_trainer_from_config(cfg)` has run (that call populates `train_dataset_len`). Print the summary after building the trainer to see real numbers.

Export the resolved config for reproducible logging or later diffing:

```python
cfg.to_dict()                 # plain dict of every field
cfg.to_json()                 # JSON string

# Archive it next to a run:
from pathlib import Path
Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.log_dir, "config.json").write_text(cfg.to_json())
```

---

## Configuration Reference

All configuration is done through a single `Config` dataclass. Every field has a default value; only override what you need.

### Model

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"turbo"` | Whisper model variant: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `turbo`. |
| `lang` | `str` | `"ar"` | Language code (e.g. `"en"`, `"ar"`, `"fr"`, `"zh"`). Under `task="transcribe"` this is the transcription language; under `task="translate"` it is the source audio language. |
| `task` | `str` | `"transcribe"` | `"transcribe"` (audio → text in `lang`) or `"translate"` (audio → English text). Validated at construction. |

### Freezing Strategy

| Parameter | Type | Default | Description |
|---|---|---|---|
| `freeze_encoder` | `bool` | `True` | Freeze the encoder during training. |
| `num_frozen_encoder_layers` | `Optional[int]` | `None` | Encoder transformer blocks to freeze. See [Freezing Strategies In Depth](#freezing-strategies-in-depth). |
| `freeze_decoder` | `bool` | `False` | Freeze the decoder during training. |
| `num_frozen_decoder_layers` | `Optional[int]` | `None` | Decoder transformer blocks to freeze. See [Freezing Strategies In Depth](#freezing-strategies-in-depth). |

### Weight Untying

| Parameter | Type | Default | Description |
|---|---|---|---|
| `untie_weights` | `bool` | `False` | Decouple the decoder's output projection (`lm_head`) from `token_embedding`. See [Weight Untying In Depth](#weight-untying-in-depth). |

### Data

| Parameter | Type | Default | Description |
|---|---|---|---|
| `train_data` | `str` | `"YOUR_TRAIN_DATA.jsonl"` | Path to the training JSONL file. |
| `val_data` | `str` | `"YOUR_VAL_DATA.jsonl"` | Path to the validation JSONL file. |
| `min_duration` | `float` | `5.0` | Segments shorter than this (seconds) are skipped. |
| `max_duration` | `float` | `30.0` | Segments longer than this (seconds) are skipped. |
| `sample_rate` | `int` | `16000` | Audio sample rate. Whisper always expects 16 kHz — do not change this. |
| `tmp_folder` | `Optional[str]` | `None` | Directory for caching mel spectrograms. `None` disables caching. |
| `storage_threshold_gb` | `float` | `100.0` | Minimum free disk space (GB) required before a spectrogram is cached. |

### Optimizer

| Parameter | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | `1e-5` | Peak learning rate for AdamW. |
| `weight_decay` | `float` | `0.01` | L2 regularisation applied to all parameters except biases and LayerNorm weights. |
| `adam_epsilon` | `float` | `1e-8` | Epsilon for numerical stability in AdamW. |
| `warmup_steps` | `int` | `2000` | Linear warmup steps before the LR reaches its peak. |
| `gradient_accumulation_steps` | `int` | `1` | Effective batch size = `train_batch_size × gradient_accumulation_steps`. |

### Training Loop

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_train_epochs` | `int` | `10` | Maximum number of training epochs. |
| `seed` | `int` | `1415` | Global random seed for reproducibility. |
| `train_batch_size` | `int` | `32` | Samples per training batch. |
| `val_batch_size` | `int` | `16` | Samples per validation batch. |
| `train_num_workers` | `int` | `32` | DataLoader worker processes for training. |
| `val_num_workers` | `int` | `16` | DataLoader worker processes for validation. |
| `train_dataset_len` | `int` | `0` | Auto-populated by `prepare_trainer_from_config` from the training set size; used to size the LR schedule. You normally don't set this manually. |

### Hardware

| Parameter | Type | Default | Description |
|---|---|---|---|
| `accelerator` | `str` | `"auto"` | PyTorch Lightning accelerator. `"auto"` picks a GPU if available, else CPU. |
| `precision` | `int \| str` | `16` | `16` for mixed precision (recommended), `32` for full, or `"bf16-mixed"` on Ampere+ GPUs. Passed straight to `Trainer(precision=...)`. |

### Logging

| Parameter | Type | Default | Description |
|---|---|---|---|
| `log_dir` | `str` | `"logs/"` | Root directory for TensorBoard logs. |
| `logger_name` | `str` | `"whisper_turbo_v1"` | Experiment sub-directory inside `log_dir`. Change per run. |
| `log_every_n_steps` | `int` | `1` | How often (in optimiser steps) to write metrics to TensorBoard. |
| `lr_monitor_logging_interval` | `str` | `"epoch"` | LR logging frequency: `"step"` or `"epoch"`. |

### Checkpointing

| Parameter | Type | Default | Description |
|---|---|---|---|
| `checkpoint_dirpath` | `str` | `"logs/checkpoint"` | Directory for `.ckpt` files. |
| `checkpoint_filename` | `str` | `"whisper-finetuned-{epoch:04d}-{val_loss:.5f}-{val_wer:.5f}-{val_cer:.5f}"` | Filename template with metric placeholders. |
| `checkpoint_monitor` | `str` | `"val_wer"` | Metric used to rank and keep the best checkpoints. |
| `checkpoint_monitor_mode` | `str` | `"min"` | `"min"` for WER/CER/loss, `"max"` for accuracy. |
| `save_top_k` | `int` | `5` | Number of best checkpoints to retain on disk. |

---

## Freezing Strategies In Depth

Whisper is an encoder-decoder model. The encoder converts audio into dense representations; the decoder generates text tokens from those representations.

**Encoder layout:** `conv1 → conv2 → blocks[0 … N-1] → ln_post`

**Decoder layout:** `token_embedding → blocks[0 … N-1] → ln`

The four freezing parameters interact as follows:

### Encoder

```python
# Freeze the entire encoder (recommended starting point)
cfg = Config(freeze_encoder=True, num_frozen_encoder_layers=None)

# Freeze only the convolutional front-end (conv1, conv2) and ln_post;
# all transformer blocks remain trainable
cfg = Config(freeze_encoder=True, num_frozen_encoder_layers=0)

# Freeze the front-end + the first 4 transformer blocks
cfg = Config(freeze_encoder=True, num_frozen_encoder_layers=4)

# Full encoder fine-tuning — nothing is frozen
cfg = Config(freeze_encoder=False)
```

### Decoder

```python
# Decoder fully trainable (default)
cfg = Config(freeze_decoder=False)

# Freeze the entire decoder
cfg = Config(freeze_decoder=True, num_frozen_decoder_layers=None)

# Freeze only token_embedding and ln; all transformer blocks trainable
cfg = Config(freeze_decoder=True, num_frozen_decoder_layers=0)

# Freeze token_embedding, ln, and the first 2 decoder blocks
cfg = Config(freeze_decoder=True, num_frozen_decoder_layers=2)
```

### Common Recipes

| Goal | Settings |
|---|---|
| Fast fine-tuning with minimal memory (default) | `freeze_encoder=True, num_frozen_encoder_layers=None, freeze_decoder=False` |
| Fine-tune the top encoder layers only | `freeze_encoder=True, num_frozen_encoder_layers=N` (bottom N frozen, rest trainable) |
| Full model fine-tuning (most data required) | `freeze_encoder=False, freeze_decoder=False` |
| Frozen encoder + frozen lower decoder layers | `freeze_encoder=True, freeze_decoder=True, num_frozen_decoder_layers=N` |

> **Note:** The number of transformer blocks varies by model. For example, `turbo` has 32 encoder blocks and 4 decoder blocks; `base` has 6 and 6. Passing a value larger than the actual block count freezes all blocks without error.

---

## Weight Untying In Depth

By default Whisper ties its decoder output projection to `token_embedding` — the same weight matrix is used for both input embedding lookup and output logit computation.

Setting `untie_weights=True` creates an independent `lm_head` Linear layer initialised from a copy of those weights, then patches the decoder's forward method to use `lm_head` for logit computation. This is most useful in combination with decoder freezing:

```python
cfg = Config(
    freeze_decoder=True,
    num_frozen_decoder_layers=0,  # freeze token_embedding + ln; blocks trainable
    untie_weights=True,           # lm_head is now an independent trainable projection
)
```

With this setup `token_embedding` stays frozen (preserving the model's vocabulary representations from pre-training) while `lm_head` can still adapt to the fine-tuning domain.

---

## Spectrogram Caching

Computing mel spectrograms on the fly is CPU-intensive. Enabling caching saves significant time after the first epoch.

```python
cfg = Config(
    tmp_folder="tmp/spectrograms",   # cache directory
    storage_threshold_gb=50.0,       # only cache if > 50 GB free
)
```

- On the **first epoch**, spectrograms are computed from audio and saved to `tmp_folder` as `.pt` files (one per utterance, keyed by `utt` ID).
- On **subsequent epochs**, the cached `.pt` files are loaded directly, bypassing audio decoding and FFT computation entirely.
- If free disk space falls below `storage_threshold_gb`, the spectrogram is computed on the fly and **not** cached — training continues safely even if the disk fills up.
- Set `tmp_folder=None` (default) to always compute spectrograms on the fly.

---

## Monitoring Training

Start TensorBoard in a separate terminal to watch metrics live:

```bash
tensorboard --logdir=logs/
```

The following metrics are logged:

| Metric | When | Description |
|---|---|---|
| `train_loss` | Every step | Cross-entropy loss on the training batch |
| `val_loss` | Every validation step | Cross-entropy loss on the validation batch |
| `val_wer` | Every validation step | Word Error Rate across the validation batch |
| `val_cer` | Every validation step | Character Error Rate across the validation batch |
| `lr` | Every epoch (default) | Current learning rate from the scheduler |

---

## Loading a Checkpoint for Inference

After training, load any saved `.ckpt` file for transcription:

```python
from finetune_openai_whisper import Config, WhisperModelModule

cfg = Config(model_name="turbo", lang="ar", task="transcribe")

model = WhisperModelModule.load_from_checkpoint(
    "logs/checkpoint/whisper-finetuned-epoch=0010-....ckpt",
    cfg=cfg,
    model_name=cfg.model_name,
    lang=cfg.lang,
    task=cfg.task,
)
model.eval()

result = model.model.transcribe("path/to/audio.wav")
print(result["text"])
```

---

## Converting to Official Whisper Format

The PyTorch Lightning `.ckpt` format wraps model weights with training metadata. To use your fine-tuned model with the standard `whisper.load_model()` API, convert it first:

```bash
python -m finetune_openai_whisper.convert_ckpt_to_official_whisper_format \
    turbo \
    logs/checkpoint/your_checkpoint.ckpt \
    whisper_turbo_finetuned.pt
```

After conversion, load as a standard Whisper model:

```python
import whisper
model = whisper.load_model("whisper_turbo_finetuned.pt")
result = model.transcribe("audio.wav")
print(result["text"])
```

If you finetuned the model with untied weights, load the checkpoint as follows:

```python
import torch
import whisper
from finetune_openai_whisper.helpers import untie_embed_n_output_weights

model = whisper.load_model('turbo')
untie_embed_n_output_weights(model)

finetuned_model_path = 'whisper_turbo_finetuned.pt'
state_dict = torch.load(finetuned_model_path)['model_state_dict']
model.load_state_dict(state_dict)

result = model.transcribe("audio.wav")
print(result["text"])
```

---

## Converting to Hugging Face Format

To use your fine-tuned model with the 🤗 Transformers library:

1. First convert to the official Whisper format as described above.
2. Then use the [Whisper checkpoint converter](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/convert_openai_to_hf.py) provided by Hugging Face Transformers:

> **Note:** This script works only on finetuned models **with** tied weights.

```bash
python convert_openai_to_hf.py \
    --checkpoint_path whisper_finetuned.pt \
    --pytorch_dump_folder_path ./whisper-hf \
    --convert_preprocessor True
```

---

## Troubleshooting

### CUDA Out of Memory

```python
cfg = Config(
    train_batch_size=8,
    gradient_accumulation_steps=4,   # maintains effective batch size of 32
    precision=16,
    freeze_encoder=True,
)
```

### Slow Data Loading

```python
cfg = Config(
    train_num_workers=8,             # tune to your CPU core count
    tmp_folder="tmp/spectrograms",   # enable caching to skip repeated FFT work
)
```

### Poor Convergence

```python
cfg = Config(
    learning_rate=5e-6,
    warmup_steps=500,                # shorter warmup for smaller datasets
    freeze_encoder=True,
)
```

### `task="translate"` Raises on an English-Only Model

Translation is only defined for multilingual checkpoints. Use a multilingual variant (e.g. `small`, `medium`, `large-v3`, `turbo`) rather than an `*.en` model.

### Sparse Tensor Error with DDP

This is handled automatically by `prepare_trainer_from_config()`. The `alignment_heads` buffer in Whisper is sparse and incompatible with PyTorch DDP; it is converted to a dense tensor before training begins.

### Checkpoint Not Found for Inference

Checkpoint filenames include metric values rendered at save time. Use a glob pattern to find the right file:

```bash
ls logs/checkpoint/whisper-finetuned-*.ckpt
```

---

## Acknowledgments

- Training pipeline inspired by [this Colab notebook](https://colab.research.google.com/drive/1P4ClLkPmfsaKn2tBbRp0nVjGMRKR-EWz?usp=sharing)
- WER/CER evaluation adapted from [abjadai/catt](https://github.com/abjadai/catt)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
