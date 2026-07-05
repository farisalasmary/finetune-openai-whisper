"""
Dataset classes for loading and preprocessing audio-text pairs for Whisper fine-tuning.

WhisperDataset handles audio loading, mel-spectrogram computation, optional
on-disk caching, duration filtering, and tokenization.

WhisperDataCollatorWithPadding collates individual samples into padded batches
and masks special-token positions so they don't contribute to the training loss.
"""

import os
import numpy as np
import pandas as pd

import torch
import whisper
from torch.utils.data import Dataset

from finetune_openai_whisper.utils import check_disk_space, load_audio


class WhisperDataset(Dataset):
    """
    PyTorch Dataset for audio-text pairs used in Whisper fine-tuning.

    Each sample in the backing JSONL file describes one audio segment.
    On first access the dataset computes the mel spectrogram from audio;
    on subsequent accesses (when caching is enabled) it loads the saved
    .pt file directly, bypassing audio decoding and FFT computation.

    JSONL format (one JSON object per line)::

        {"utt": "utt001", "audio_filepath": "/data/utt001.wav",
         "text": "hello world", "duration": 3.2, "offset": 0.0}
    """

    def __init__(
        self,
        json_file: str,
        tokenizer,
        n_mels: int,
        min_duration: float = 5.0,
        max_duration: float = 30.0,
        tmp_folder: str = None,
        storage_threshold_gb: float = 20.0,
    ):
        """
        Args:
            json_file:            Path to the JSONL metadata file.
            tokenizer:            Whisper tokenizer instance.
            n_mels:               Number of mel frequency bins (taken from model.dims.n_mels).
            min_duration:         Segments shorter than this (seconds) are skipped.
            max_duration:         Segments longer than this (seconds) are skipped.
            tmp_folder:           Directory for caching spectrograms. None disables caching.
            storage_threshold_gb: Minimum free disk space (GB) required before a spectrogram
                                  is written to the cache.
        """
        self.data = pd.read_json(json_file, lines=True)

        print(f'Total samples BEFORE filtering: {len(self.data)}')
        print(f'Total duration BEFORE filtering: {self.data["duration"].sum() / 3600:.2f} hours')

        self.data = self.data[
            (self.data['duration'] >= min_duration) &
            (self.data['duration'] <= max_duration)
        ].reset_index(drop=True)

        print(f'Total samples AFTER filtering:  {len(self.data)}')
        print(f'Total duration AFTER filtering:  {self.data["duration"].sum() / 3600:.2f} hours')

        self.tokenizer            = tokenizer
        self.n_mels               = n_mels
        self.tmp_folder           = tmp_folder
        self.storage_threshold_gb = storage_threshold_gb

        if self.tmp_folder:
            os.makedirs(self.tmp_folder, exist_ok=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one training sample.

        Returns:
            dict with keys:
              - ``mel_spects``    – log-mel spectrogram tensor, shape (n_mels, T).
              - ``labels``        – target token IDs (decoder input shifted right by one,
                                   with <|endoftext|> appended).
              - ``dec_input_ids`` – full decoder input token IDs including the SOT prefix.
        """
        row          = self.data.iloc[idx]
        segment_id   = row['utt']
        text         = row['text']

        # ── Spectrogram (cached or computed) ──────────────────────────────
        cache_path = os.path.join(self.tmp_folder, f'{segment_id}.pt') if self.tmp_folder else None

        if cache_path and os.path.exists(cache_path):
            mel = torch.load(cache_path)
        else:
            audio_signal = load_audio(row['audio_filepath'], row['offset'], row['duration'])
            audio_signal = whisper.pad_or_trim(audio_signal)
            mel          = whisper.log_mel_spectrogram(audio_signal, n_mels=self.n_mels)

            if cache_path and check_disk_space(self.storage_threshold_gb, self.tmp_folder):
                torch.save(mel, cache_path)

        # ── Tokenization ──────────────────────────────────────────────────
        # Decoder input: [<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>, ...tokens...]
        # Labels:        decoder input shifted left by one position, with <|endoftext|> appended.
        sot_tokens = list(self.tokenizer.sot_sequence_including_notimestamps)
        text_tokens = self.tokenizer.encode(text)
        dec_input_ids = sot_tokens + text_tokens
        labels        = dec_input_ids[1:] + [self.tokenizer.eot]

        return {
            "mel_spects":    mel,
            "labels":        labels,
            "dec_input_ids": dec_input_ids,
        }


class WhisperDataCollatorWithPadding:
    """
    Collator that pads variable-length token sequences into fixed-size batches.

    Special-token positions (the SOT prefix) are masked with -100 in the labels
    tensor so the loss is only computed on the actual transcription tokens. This
    preserves the model's original language-identification representations.

    The class name retains the original spelling for backward compatibility.
    """

    def __init__(self, ignored_tokens):
        """
        Args:
            ignored_tokens: Token IDs corresponding to the SOT prefix
                            (tokenizer.sot_sequence_including_notimestamps).
                            These positions are masked out in the labels.
        """
        self.ignored_tokens = torch.tensor(ignored_tokens)

    def __call__(self, features: list) -> dict:
        """
        Collate a list of dataset samples into a padded batch.

        Args:
            features: List of dicts returned by WhisperDataset.__getitem__.

        Returns:
            Batched dict with keys ``mel_spects``, ``labels``, ``dec_input_ids``.
        """
        mel_spects, labels, dec_input_ids = [], [], []

        for f in features:
            mel_spects.append(f["mel_spects"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        # Stack spectrograms along a new batch dimension.
        mel_spects = torch.stack(mel_spects)

        # Pad all sequences to the longest one in the batch.
        # Labels use -100 as the pad value (ignored by CrossEntropyLoss).
        # Decoder input IDs use the <|endoftext|> token (50257) as padding.
        max_len = max(
            max(len(l) for l in labels),
            max(len(e) for e in dec_input_ids),
        )

        labels = [
            np.pad(l, (0, max_len - len(l)), constant_values=-100)
            for l in labels
        ]
        dec_input_ids = [
            np.pad(e, (0, max_len - len(e)), constant_values=50257)
            for e in dec_input_ids
        ]

        batch = {k: torch.tensor(np.array(v), requires_grad=False)
                 for k, v in [("labels", labels), ("dec_input_ids", dec_input_ids)]}

        # Mask SOT prefix tokens in the labels so they don't contribute to the loss.
        # This keeps the model's language-identification behaviour intact.
        batch['labels'][torch.isin(batch['labels'], self.ignored_tokens)] = -100
        batch["mel_spects"] = mel_spects

        return batch
