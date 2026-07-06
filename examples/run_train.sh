
python -i train.py \
         --model turbo \
         --lang ar \
         --train_data /content/train.jsonl \
         --val_data /content/dev.jsonl \
         --train_batch 64 \
         --val_batch 32 \
         --grad_accum 1 \
         --train_workers 8 \
         --val_workers 4 \
         --cache_dir /mnt/local-scratch/tmp/spectrograms \
         --output_dir /content/whisper_turbo_v3_mgb2_untie_unfreeze_2_v2 \
         --run_name turbo_untie_unfreeze2 \
         --save_top_k 3 \
         --freeze_decoder \
         --num_frozen_decoder_layers 2 \
         --untie_weights
