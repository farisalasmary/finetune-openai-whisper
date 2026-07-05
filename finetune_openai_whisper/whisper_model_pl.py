"""
PyTorch Lightning module wrapping OpenAI's Whisper model for fine-tuning.

Key features:
  - Configurable encoder and decoder layer freezing via Config
  - Cross-entropy training loss with padding-token masking
  - Per-batch and aggregate WER / CER computed during validation
  - Linear warmup + linear decay LR schedule (AdamW)
"""

import torch
import torch.nn as nn
import whisper
from pytorch_lightning import LightningModule
from torch.optim import AdamW

from finetune_openai_whisper import xer
from finetune_openai_whisper.utils import (
    apply_freezing,
    remove_special_tokens,
    get_linear_schedule_with_warmup,
)


class WhisperModelModule(LightningModule):
    """
    PyTorch Lightning module wrapping OpenAI's Whisper for fine-tuning.

    Freezing is fully controlled by the Config object:
      - cfg.freeze_encoder / cfg.num_frozen_encoder_layers
      - cfg.freeze_decoder / cfg.num_frozen_decoder_layers

    See Config for the full description of each freezing option.
    """

    def __init__(self, cfg, model_name: str = "turbo", lang: str = "ar", task: str = "transcribe") -> None:
        super().__init__()

        # Decoding parameters. The fp16 flag is intentionally NOT fixed here:
        # it is resolved per-device in validation_step (fp16 only on CUDA),
        # because the module is still on CPU at construction time and Whisper's
        # fp16 decode path is unimplemented for several ops on CPU.
        #
        # task is 'transcribe' (audio → same-language text) or 'translate'
        # (audio → English text); it selects the task token in the decoder's
        # SOT prefix and the decoding options used at validation time.
        self.lang = lang
        self.task = task

        self.model = whisper.load_model(model_name)

        # Whisper's translate task (audio → English) is only defined for the
        # multilingual checkpoints; the *.en models have no language/task tokens.
        if self.task == "translate" and not self.model.is_multilingual:
            raise ValueError(
                f"task='translate' requires a multilingual model, but '{model_name}' "
                "is English-only. Use a multilingual checkpoint such as 'small', "
                "'medium', 'large-v3', or 'turbo'."
            )

        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language=lang,
            task=self.task,
        )

        self.model.train()

        # ── Encoder freezing ───────────────────────────────────────────────
        encoder = self.model.encoder
        apply_freezing(
            self.model,
            freeze_component=cfg.freeze_encoder,
            num_frozen_layers=cfg.num_frozen_encoder_layers,
            blocks=list(encoder.blocks),
            extra_params=(
                list(encoder.conv1.parameters())
                + list(encoder.conv2.parameters())
                + list(encoder.ln_post.parameters())
            ),
        )

        # ── Decoder freezing ───────────────────────────────────────────────
        decoder = self.model.decoder
        apply_freezing(
            self.model,
            freeze_component=cfg.freeze_decoder,
            num_frozen_layers=cfg.num_frozen_decoder_layers,
            blocks=list(decoder.blocks),
            extra_params=(
                list(decoder.token_embedding.parameters())
                + list(decoder.ln.parameters())
                + [decoder.positional_embedding]
            ),
        )

        # -100 is the standard PyTorch ignore index: padding positions in labels
        # are set to -100 by the collator so they don't contribute to the loss.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        mel_spects    = batch["mel_spects"]
        labels        = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(mel_spects)
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        mel_spects    = batch["mel_spects"]
        labels        = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(mel_spects)
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        # Whisper's fp16 decode path casts the mel to half precision, which is
        # unimplemented for several ops on CPU and mismatches the fp32 model
        # weights there. Mirror whisper.transcribe()'s own behaviour: use fp16
        # only on CUDA, and fall back to fp32 everywhere else.
        decode_options = whisper.DecodingOptions(
            language=self.lang,
            task=self.task,
            without_timestamps=True,
            fp16=(self.device.type == "cuda"),
        )
        with torch.no_grad():
            decoded_texts = self.model.decode(mel_spects, options=decode_options)
            hyp_texts = [t.text for t in decoded_texts]
        # Replace padding markers with the end-of-transcript token so the
        # tokenizer can decode predictions and references cleanly.
        labels[labels == -100] = self.tokenizer.eot

        # Accumulate edit-distance numerators and denominators separately so
        # the final error rate is corpus-level (not an average of per-sample rates).
        total_wer_distance   = 0
        total_wer_ref_length = 0
        total_cer_distance   = 0
        total_cer_ref_length = 0

        for hyp_text, ref_input_ids in zip(hyp_texts, labels):
            ref_text = remove_special_tokens(self.tokenizer.decode(ref_input_ids))

            wer_info = xer.wer(ref_text, hyp_text)
            cer_info = xer.cer(ref_text, hyp_text)

            # Per-sample rates for console logging.
            sample_wer = wer_info['distance'] / wer_info['ref_length']
            sample_cer = cer_info['distance'] / cer_info['ref_length']

            total_wer_distance   += wer_info['distance']
            total_wer_ref_length += wer_info['ref_length']
            total_cer_distance   += cer_info['distance']
            total_cer_ref_length += cer_info['ref_length']

            print('Hyp:', hyp_text)
            print('Ref:', ref_text)
            print('WER:', sample_wer)
            print('CER:', sample_cer)
            print('-' * 89)

        total_wer = total_wer_distance / total_wer_ref_length
        total_cer = total_cer_distance / total_cer_ref_length

        print('Total WER:', total_wer)
        print('Total CER:', total_cer)
        print('-' * 89)

        self.log("val_loss", loss,      on_step=True, prog_bar=True, logger=True)
        self.log("val_cer",  total_cer, on_step=True, prog_bar=True, logger=True)
        self.log("val_wer",  total_wer, on_step=True, prog_bar=True, logger=True)

        return {"cer": total_cer, "wer": total_wer, "loss": loss}

    def configure_optimizers(self):
        # Apply weight decay to all parameters except biases and LayerNorm weights,
        # which are commonly excluded following the original BERT/GPT conventions.
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon,
        )
        self.optimizer = optimizer

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.t_total,
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Compute total optimiser steps for the LR scheduler.
            # This must be done in setup() (not __init__) because train_dataset_len
            # is only available after the datasets have been built by the trainer.
            self.t_total = (
                (self.cfg.train_dataset_len // self.cfg.train_batch_size)
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )
