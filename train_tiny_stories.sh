uv run cs336_basics/trainer.py \
  --project='tiny-stories-v2' \
  --train_path='data/TinyStoriesV2-GPT4-train-trie-tokenized.dat' \
  --val_path='data/TinyStoriesV2-GPT4-valid-trie-tokenized.dat' \
  --models_base_path='data/models' \
  --run_name='debug_01' \
  --device='mps' \
  --group='mps_debug' \
  --batch_size=32 \
  --max_steps_per_epoch=250 \
  --max_epochs=20 \
  --compile=True \
  $@
