"""
Core training helpers for Whisper fine-tuning.

prepare_trainer_from_config() is the main entry point: it builds the model,
datasets, data loaders, callbacks, and PyTorch Lightning Trainer from a single
Config object, then returns everything ready for trainer.fit().

untie_embed_n_output_weights() is an optional advanced patch that decouples
the decoder's output projection from its token embedding weights. It is called
automatically by prepare_trainer_from_config() when cfg.untie_weights=True.

Typical usage
-------------
    from finetune_openai_whisper.config import Config
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

import types
from typing import Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from finetune_openai_whisper.config import Config
from finetune_openai_whisper.whisper_model_pl import WhisperModelModule
from finetune_openai_whisper.whisper_dataset import WhisperDataset, WhisperDataCollatorWhithPadding


def prepare_trainer_from_config(cfg: Config):
    """
    Build all training components from a Config object.

    Steps performed:
      1. Seed everything for reproducibility.
      2. Initialise WhisperModelModule with the requested freezing strategy.
      3. Convert the sparse alignment_heads buffer to dense (required for DDP).
      4. Optionally decouple embedding and output-projection weights (cfg.untie_weights).
      5. Build train and validation WhisperDataset instances with duration filtering.
      6. Wrap both datasets in DataLoaders with the padding collator.
      7. Configure ModelCheckpoint, LearningRateMonitor, and TensorBoardLogger.
      8. Construct and return the PyTorch Lightning Trainer.

    Args:
        cfg: Fully populated Config instance.

    Returns:
        tuple: (trainer, model, train_dataloader, val_dataloader)
               Pass these directly to trainer.fit().
    """
    pl.seed_everything(cfg.seed, workers=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = WhisperModelModule(cfg, cfg.model_name, cfg.lang)

    # Whisper's alignment_heads buffer is a sparse tensor, which is incompatible
    # with PyTorch DDP. Convert it to dense before training begins.
    # See: https://discuss.pytorch.org/t/ddp-no-support-for-sparse-tensor/190375/1
    alignment_heads_dense = model.model.get_buffer("alignment_heads").to_dense()
    model.model.register_buffer("alignment_heads", alignment_heads_dense, persistent=False)

    # Optionally decouple the output projection from token_embedding so that
    # lm_head can adapt even when token_embedding is frozen.
    if cfg.untie_weights:
        untie_embed_n_output_weights(model)

    # ── Datasets ───────────────────────────────────────────────────────────────
    ignored_tokens = model.tokenizer.sot_sequence_including_notimestamps

    train_dataset = WhisperDataset(
        cfg.train_data,
        model.tokenizer,
        n_mels=model.model.dims.n_mels,
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
        tmp_folder=cfg.tmp_folder,
        storage_threshold_gb=cfg.storage_threshold_gb,
    )

    val_dataset = WhisperDataset(
        cfg.val_data,
        model.tokenizer,
        n_mels=model.model.dims.n_mels,
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
        tmp_folder=cfg.tmp_folder,
        storage_threshold_gb=cfg.storage_threshold_gb,
    )

    # ── Data loaders ───────────────────────────────────────────────────────────
    collator = WhisperDataCollatorWhithPadding(ignored_tokens)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.train_num_workers,
        shuffle=True,
        collate_fn=collator,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.val_num_workers,
        shuffle=False,
        collate_fn=collator,
    )

    # Pass dataset length to the model so the LR scheduler can compute t_total
    # in WhisperModelModule.setup(). Must be set before trainer.fit() is called.
    cfg.train_dataset_len = len(train_dataset)

    # ── Callbacks & logger ─────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dirpath,
        filename=cfg.checkpoint_filename,
        save_top_k=cfg.save_top_k,
        monitor=cfg.checkpoint_monitor,
        mode=cfg.checkpoint_monitor_mode,
    )

    lr_monitor = LearningRateMonitor(
        logging_interval=cfg.lr_monitor_logging_interval,
    )

    logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=cfg.logger_name,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        precision=cfg.precision,
        accelerator=cfg.accelerator,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.log_every_n_steps,
    )

    return trainer, model, train_dataloader, val_dataloader


def untie_embed_n_output_weights(model: Union[WhisperModelModule, "whisper.Whisper"], add_bias: bool = False) -> None:
    """
    Decouple the decoder's output projection from its token embedding weights.

    By default Whisper ties the output logit projection to token_embedding (the
    same weight matrix serves both purposes). This function creates an independent
    lm_head Linear layer initialised from a copy of those weights, then patches
    the decoder's forward method to use lm_head instead of token_embedding.weight.

    This is most useful when freeze_decoder=True and num_frozen_decoder_layers=0:
    token_embedding stays frozen (preserving the model's vocabulary representations
    from pre-training) while lm_head continues to adapt during fine-tuning.

    Called automatically by prepare_trainer_from_config() when cfg.untie_weights=True.
    Can also be called manually before trainer.fit() for full control.

    Args:
        model:    Either a WhisperModelModule (PL wrapper) or a raw whisper.Whisper
                  model. Must be called before trainer.fit().
        add_bias: If True, adds a trainable bias term to the new lm_head layer.
    """
    # Support both the PL wrapper (model.model is the Whisper instance)
    # and a raw whisper.Whisper model passed directly.
    whisper_model = model.model if isinstance(model, WhisperModelModule) else model

    def _forward(self, x: torch.Tensor, xa: torch.Tensor, kv_cache: Optional[dict] = None):
        """
        Decoder forward pass using the decoupled lm_head projection.

        Args:
            x:        Text token IDs, shape (batch_size, <= n_ctx).
            xa:       Encoded audio features, shape (batch_size, n_audio_ctx, n_audio_state).
            kv_cache: Optional key-value cache for autoregressive decoding.
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset: offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        return self.lm_head(x)   # decoupled projection instead of token_embedding.weight

    vocab_size, d_model = whisper_model.decoder.token_embedding.weight.shape
    lm_head = nn.Linear(d_model, vocab_size, bias=add_bias)

    # Initialise from the current embedding weights so the model starts from
    # the same point as the original tied-weight configuration.
    with torch.no_grad():
        lm_head.weight.copy_(whisper_model.decoder.token_embedding.weight)

    whisper_model.decoder.lm_head = lm_head
    whisper_model.decoder.forward = types.MethodType(_forward, whisper_model.decoder)
