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
