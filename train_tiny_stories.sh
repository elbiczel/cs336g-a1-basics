uv run cs336_basics/trainer.py \
  --project='tiny-stories-v2' \
  --train_path='data/TinyStoriesV2-GPT4-train-trie-tokenized.dat' \
  --val_path='data/TinyStoriesV2-GPT4-valid-trie-tokenized.dat' \
  --models_base_path='data/models' \
  --run_name='debug_04' \
  --device='cpu' \
  --group='cpu_debug' \
  --max_steps_per_epoch=5_000 \
  --max_epochs=20 \

