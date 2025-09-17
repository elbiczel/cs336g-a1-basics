uv run cs336_basics/generator.py \
  --project='tiny-stories-v2' \
  --trainer_run='afx1f4bg' \
  --models_base_path='data/models' \
  --device='mps' \
  --tokenizer_vocab='data/TinyStoriesV2-GPT4-train-vocab.json' \
  --tokenizer_merges='data/TinyStoriesV2-GPT4-train-merges.txt' \
  --prompt=' ' \
  $@

