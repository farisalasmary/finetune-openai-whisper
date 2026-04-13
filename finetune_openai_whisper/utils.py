"""
Utility functions shared across the fine-tuning pipeline.

Includes the linear warmup/decay LR scheduler (ported from Hugging Face
Transformers), text post-processing helpers, and generic encoder/decoder
freezing logic.
"""

import re
import shutil
import librosa
from functools import partial
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


# ── Learning-rate scheduler ────────────────────────────────────────────────────

def _linear_warmup_decay_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int) -> float:
    """
    Per-step LR multiplier for the linear warmup + linear decay schedule.

    During warmup (step < num_warmup_steps) the multiplier rises linearly
    from 0 to 1. After warmup it decays linearly back to 0 at num_training_steps.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Linear warmup then linear decay LR schedule (ported from Hugging Face Transformers).

    The learning rate increases linearly from 0 to the optimizer's initial LR
    over num_warmup_steps, then decreases linearly back to 0 over the remaining
    training steps.

    Args:
        optimizer:           The optimizer whose LR will be scheduled.
        num_warmup_steps:    Steps over which the LR warms up to its peak.
        num_training_steps:  Total number of training steps (warmup + decay).
        last_epoch:          Index of the last epoch when resuming training.

    Returns:
        A LambdaLR scheduler instance.
    """
    lr_lambda = partial(
        _linear_warmup_decay_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ── Text helpers ───────────────────────────────────────────────────────────────

def remove_special_tokens(text: str) -> str:
    """
    Strip Whisper's special tokens and collapse extra whitespace.

    Special tokens follow the pattern ``<|...|>`` (e.g. ``<|startoftranscript|>``,
    ``<|ar|>``, ``<|notimestamps|>``). After removal, runs of whitespace are
    collapsed to a single space and leading/trailing whitespace is stripped.

    Args:
        text: Raw decoded text that may contain special tokens.

    Returns:
        Cleaned transcript string.
    """
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Freezing ───────────────────────────────────────────────────────────────────

def apply_freezing(
    model,
    freeze_component: bool,
    num_frozen_layers: Optional[int],
    blocks: list,
    extra_params: list,
) -> None:
    """
    Freeze part or all of an encoder or decoder component.

    The ``extra_params`` list should contain all non-block parameters for the
    component (e.g. conv layers and ln_post for the encoder, or token_embedding
    and ln for the decoder).

    Freezing behaviour based on the combination of arguments:

    +-------------------+--------------------+------------------------------------------+
    | freeze_component  | num_frozen_layers  | Effect                                   |
    +===================+====================+==========================================+
    | False             | (any)              | Nothing frozen — component fully trains. |
    +-------------------+--------------------+------------------------------------------+
    | True              | None               | Freeze everything: all blocks + extras.  |
    +-------------------+--------------------+------------------------------------------+
    | True              | 0                  | Freeze only extra_params; blocks train.  |
    +-------------------+--------------------+------------------------------------------+
    | True              | N > 0              | Freeze extra_params + first N blocks.    |
    +-------------------+--------------------+------------------------------------------+

    Args:
        model:             The full Whisper model (unused directly; kept for clarity).
        freeze_component:  Whether to apply any freezing at all.
        num_frozen_layers: How many transformer blocks to freeze (see table above).
        blocks:            List of transformer block modules to potentially freeze.
        extra_params:      Non-block parameters (conv layers, embeddings, layer norms).
    """
    if not freeze_component:
        return

    # Always freeze the non-block parameters when freezing is active.
    for p in extra_params:
        p.requires_grad = False

    if num_frozen_layers is None:
        # Freeze the entire component — all blocks and all extra params.
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False
    elif num_frozen_layers > 0:
        # Freeze only the first N transformer blocks.
        for block in blocks[:num_frozen_layers]:
            for p in block.parameters():
                p.requires_grad = False
    # num_frozen_layers == 0: extra_params already frozen above; no blocks frozen.


# ── Audio & disk helpers ───────────────────────────────────────────────────────

def check_disk_space(threshold_gb: float) -> bool:
    """
    Return True if free disk space on the current drive exceeds the threshold.

    Used to guard spectrogram caching: if the disk is nearly full, caching is
    skipped silently so training is never interrupted by a disk-full error.

    Args:
        threshold_gb: Minimum required free space in gigabytes.

    Returns:
        True if free space > threshold_gb, False otherwise.
    """
    _, _, free = shutil.disk_usage('.')
    return (free / (1024 ** 3)) > threshold_gb


def load_audio(audio_file_path: str, offset: float, duration: float, sample_rate: int = 16000):
    """
    Load a single audio segment from a file.

    Args:
        audio_file_path: Path to the audio file (WAV, MP3, FLAC, etc.).
        offset:          Start time within the file in seconds.
        duration:        Length of the segment to load in seconds.
        sample_rate:     Target sample rate in Hz (Whisper expects 16 000 Hz).

    Returns:
        NumPy array of shape (num_samples,) containing the mono audio signal.
    """
    audio_signal, _ = librosa.load(
        audio_file_path,
        sr=sample_rate,
        mono=True,
        offset=offset,
        duration=duration,
    )
    return audio_signal
