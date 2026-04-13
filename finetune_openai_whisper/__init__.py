"""
finetune-openai-whisper
=======================
A complete pipeline for fine-tuning OpenAI's Whisper ASR model
using PyTorch Lightning.

Quickstart
----------
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
"""

from finetune_openai_whisper.config import Config
from finetune_openai_whisper.whisper_model_pl import WhisperModelModule
from finetune_openai_whisper.whisper_dataset import WhisperDataset, WhisperDataCollatorWhithPadding
from finetune_openai_whisper import xer

__version__ = "0.1.0"

__all__ = [
    "Config",
    "WhisperModelModule",
    "WhisperDataset",
    "WhisperDataCollatorWhithPadding",
    "xer",
]
