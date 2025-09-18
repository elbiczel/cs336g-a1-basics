# Example usage:
# uv run cs336_tokenizer/run_decode.py --prefix='data/TinyStoriesV2-GPT4-valid'  --out_suffix='trie-decoded' --tokenized_suffix='trie-tokenized' --use_trie=True

import argparse
from cs336_basics import io, utils
from cs336_tokenizer.bpe import Tokenizer, TrieTokenizer, MergeTokenizer
from types import SimpleNamespace

import os 

special_token = b"<|endoftext|>"
special_token_str = special_token.decode("utf-8")


@utils.stopwatch
def run_decode(tokenizer: Tokenizer, in_path: str | os.PathLike, out_path: str | os.PathLike):
    print("Decoding: ", in_path)
    tokens = io.read_tokens(in_path)
    with open(out_path, "wb") as f:
        buf = bytearray()
        for b in tokenizer.decode_iterable(tokens):
            buf.extend(b)
            if len(buf) > io._mini_chunk_size:
                f.write(buf)
                buf.clear()
        if b:
            f.write(buf)
            buf.clear()


def parse_params():
    parser = argparse.ArgumentParser(description="Decoder configuration")
    # Project Data
    parser.add_argument("--prefix", type=str, default="", help="Prefix for the paths.")
    parser.add_argument("--tokenized_suffix", type=str, default="tokenized", help="Suffix for the tokenized dataset.")
    parser.add_argument("--out_suffix", type=str, default="decoded", help="Suffix for the decoded dataset.")
    parser.add_argument("--use_trie", type=bool, default=False, help="Whether to use Trie or Merge tokenizer.")
    return parser.parse_args()


def main():
    params = parse_params()
    cfg = vars(params)
    cfg = SimpleNamespace(cfg)
    vocab_path = f"{cfg.prefix}-vocab.json"
    merges_path = f"{cfg.prefix}-merges.txt"
    in_path = f"{cfg.prefix}-{cfg.tokenized_suffix}.dat"
    out_path = f"{cfg.prefix}-{cfg.out_suffix}.txt"

    TokenizerCls = TrieTokenizer if cfg.use_trie else MergeTokenizer
    tokenizer = TokenizerCls.from_files(vocab_path, merges_path, special_tokens=None)
    run_decode(tokenizer, in_path, out_path)


if __name__ == "__main__":
    main()
