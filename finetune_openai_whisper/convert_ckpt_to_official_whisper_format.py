#!/usr/bin/env python3
"""
Convert a PyTorch Lightning Whisper checkpoint to the official Whisper .pt format.

PyTorch Lightning wraps model weights with training metadata and prefixes all
parameter names with ``model.``. This script strips that prefix and bundles
the weights with the model's architecture dimensions so the result can be
loaded directly with ``whisper.load_model()``.

Usage
-----
    python -m finetune_openai_whisper.convert_ckpt_to_official_whisper_format \\
        <whisper_model_name> <pl_checkpoint.ckpt> <output.pt>

Example
-------
    python -m finetune_openai_whisper.convert_ckpt_to_official_whisper_format \\
        turbo logs/checkpoint/whisper-finetuned.ckpt whisper_finetuned.pt

After conversion::

    import whisper
    model = whisper.load_model("whisper_finetuned.pt")
    result = model.transcribe("audio.wav")
"""

import sys
import torch
import whisper

if len(sys.argv) < 4:
    print(
        f'Usage: python {sys.argv[0]} '
        '<whisper_model_name> <pl_checkpoint.ckpt> <output.pt>'
    )
    sys.exit(1)

whisper_model_name  = sys.argv[1]   # e.g. 'turbo', 'large-v3'
pl_ckpt_path        = sys.argv[2]   # PyTorch Lightning .ckpt file
output_pt_path      = sys.argv[3]   # destination .pt file

# Load the reference model to capture its architecture dimensions.
reference_model = whisper.load_model(whisper_model_name, device='cpu')

# Load the PL checkpoint and extract the model weights.
state_dict = torch.load(pl_ckpt_path, weights_only=False, map_location='cpu')['state_dict']

# PyTorch Lightning prefixes every parameter with 'model.' — remove it.
converted_state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

# Bundle the architecture dimensions with the weights, matching the format
# expected by whisper.load_model() when given a path to a .pt file.
output = {
    'dims':             reference_model.dims.__dict__,
    'model_state_dict': converted_state_dict,
}

torch.save(output, output_pt_path)

print(f'Converted:  {pl_ckpt_path}')
print(f'Saved to:   {output_pt_path}')
print(f'Load with:  whisper.load_model("{output_pt_path}")')
