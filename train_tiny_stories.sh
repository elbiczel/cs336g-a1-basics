uv run cs336_basics/trainer.py \
  --project='tiny-stories-v2' \
  --train_path='data/TinyStoriesV2-GPT4-train-trie-tokenized.dat' \
  --val_path='data/TinyStoriesV2-GPT4-valid-trie-tokenized.dat' \
  --models_base_path='data/models' \
  --device='mps' \
  --group='init_runs' \
  --run_name='debug_1_06' \
  --lr=5e-3 \
  --final_lr=5e-4 \
  --warmup_t=200 \
  --batch_size=32 \
  --max_grad_l2_norm=3.0 \
  --z_loss_weight=5e-5 \
  $@
