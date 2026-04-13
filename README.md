# finetune-openai-whisper

A complete, production-ready pipeline for fine-tuning [OpenAI's Whisper](https://github.com/openai/whisper) ASR model on custom datasets using [PyTorch Lightning](https://lightning.ai/).

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
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
- **Configurable freezing** — freeze the full encoder, a specific number of encoder/decoder transformer blocks, or nothing at all
- **Weight untying** — optionally decouple the decoder's output projection from its token embedding so `lm_head` can adapt while `token_embedding` stays frozen
- **Spectrogram caching** — optionally cache mel spectrograms to disk so subsequent epochs load instantly
- **WER & CER evaluation** — word and character error rates computed and logged at every validation step
- **TensorBoard integration** — all metrics streamed live during training
- **Checkpoint management** — automatically keeps the top-K best checkpoints ranked by validation WER
- **Multi-GPU / DDP ready** — includes the required fix for Whisper's sparse `alignment_heads` buffer
- **Single config object** — every tunable parameter lives in one `Config` dataclass; no scattered hard-coded values

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- A CUDA-capable GPU is strongly recommended

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
| `text` | `str` | Ground truth transcription for this segment. |
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
    train_data="data/train.jsonl",
    val_data="data/val.jsonl",
)

trainer, model, train_dl, val_dl = prepare_trainer_from_config(cfg)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
```

That's it. The trainer will:
1. Load and filter your datasets
2. Initialise the Whisper model with the freezing strategy defined in `cfg`
3. Apply the DDP sparse tensor fix automatically
4. Run training with checkpointing, TensorBoard logging, and LR monitoring

### Full Example Script

Below is a complete `train.py` you can copy, adapt, and run directly:

```python
from finetune_openai_whisper import Config
from finetune_openai_whisper.helpers import prepare_trainer_from_config

cfg = Config(
    # ── Model ─────────────────────────────────────────────────────────────
    model_name="large-v3",
    lang="ar",

    # ── Data ──────────────────────────────────────────────────────────────
    train_data="data/train.jsonl",
    val_data="data/val.jsonl",
    min_duration=1.0,
    max_duration=30.0,
    tmp_folder=None,              # Set to a path to enable spectrogram caching

    # ── Freezing ──────────────────────────────────────────────────────────
    freeze_encoder=True,          # Freeze the entire encoder (recommended)
    num_frozen_encoder_layers=None,
    freeze_decoder=True,
    num_frozen_decoder_layers=20,

    # ── Weight untying ────────────────────────────────────────────────────
    untie_weights=True,           # Decouple lm_head from token_embedding

    # ── Training ──────────────────────────────────────────────────────────
    num_train_epochs=50,
    train_batch_size=16,
    val_batch_size=8,
    learning_rate=1e-5,
    warmup_steps=500,
    gradient_accumulation_steps=2,  # Effective batch size = 16 × 2 = 32

    # ── Hardware ──────────────────────────────────────────────────────────
    accelerator="auto",
    precision=16,

    # ── Logging & checkpointing ───────────────────────────────────────────
    log_dir="logs/",
    logger_name="arabic_large_v1",   # Change per experiment
    save_top_k=3,
    checkpoint_monitor="val_wer",
)

trainer, model, train_dl, val_dl = prepare_trainer_from_config(cfg)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
```

---

## Configuration Reference

All configuration is done through a single `Config` dataclass. Every field has a default value; only override what you need.

### Model

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"turbo"` | Whisper model variant: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `turbo`. |
| `lang` | `str` | `"ar"` | Target language code (e.g. `"en"`, `"ar"`, `"fr"`, `"zh"`). |

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
| `untie_weights` | `bool` | `False` | Decouple the decoder's output projection (`lm_head`) from `token_embedding`. See [Weight Untying](#weight-untying-in-depth). |

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
| `num_train_epochs` | `int` | `200` | Maximum number of training epochs. |
| `seed` | `int` | `1415` | Global random seed for reproducibility. |
| `train_batch_size` | `int` | `32` | Samples per training batch. |
| `val_batch_size` | `int` | `16` | Samples per validation batch. |
| `train_num_workers` | `int` | `32` | DataLoader worker processes for training. |
| `val_num_workers` | `int` | `16` | DataLoader worker processes for validation. |

### Hardware

| Parameter | Type | Default | Description |
|---|---|---|---|
| `accelerator` | `str` | `"auto"` | PyTorch Lightning accelerator. `"auto"` picks a GPU if available, else CPU. |
| `precision` | `int` | `16` | `16` for mixed precision (recommended), `32` for full, `"bf16-mixed"` on Ampere+ GPUs. |

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

## Weight Untying

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

cfg = Config(model_name="turbo", lang="ar")

model = WhisperModelModule.load_from_checkpoint(
    "logs/checkpoint/whisper-finetuned-epoch=0010-....ckpt",
    cfg=cfg,
    model_name=cfg.model_name,
    lang=cfg.lang,
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
