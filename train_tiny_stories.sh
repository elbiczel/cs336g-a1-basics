uv run cs336_basics/trainer.py \
  --project='tiny-stories-v2' \
  --train_path='data/TinyStoriesV2-GPT4-train-trie-tokenized.dat' \
  --val_path='data/TinyStoriesV2-GPT4-valid-trie-tokenized.dat' \
  --models_base_path='data/models' \
  --device='mps' \
  --group='bug_fix_1' \
  --run_name='bug_fix_1_00' \
  --lr=5e-4 \
  --final_lr=1e-6 \
  --warmup_t=150 \
  --batch_size=32 \
  --max_grad_l2_norm=3.0 \
  --z_loss_weight=0.0 \
  --adam_beta_1=0.9 \
  --adam_beta_2=0.95 \
  --adam_eps=1e-8 \
  $@
