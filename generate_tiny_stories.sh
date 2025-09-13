uv run cs336_basics/generator.py \
  --project='tiny-stories-v2' \
  --trainer_run='uh4z0lkn' \
  --models_base_path='data/models' \
  --device='cpu' \
  --tokenizer_vocab='data/TinyStoriesV2-GPT4-train-vocab.json' \
  --tokenizer_merges='data/TinyStoriesV2-GPT4-train-merges.txt' \
  --prompt='Once upon a time, there was a little ' \
  $@

