from cs336_basics import io, utils
from cs336_tokenizer.bpe import Tokenizer, TrieTokenizer, MergeTokenizer

import os 

prefix = 'data/TinyStoriesV2-GPT4-valid'
prefix = 'data/owt_valid'
prefix = 'data/TinyStoriesV2-GPT4-train'
prefix = 'data/owt_train'

vocab_path = f"{prefix}-vocab.json"
merges_path = f"{prefix}-merges.txt"
in_path = f"{prefix}-tokenized.dat"
out_path = f"{prefix}-decoded.txt"

TokenizerCls = MergeTokenizer

special_token = b"<|endoftext|>"
special_token_str = special_token.decode("utf-8")

def run_decode(tokenizer: Tokenizer, path: str | os.PathLike):
    print("Decoding: ", path)
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

run_decode = utils.stopwatch(run_decode)

if __name__ == "__main__":
    tokenizer = TokenizerCls.from_files(vocab_path, merges_path, special_tokens=None)
    run_decode(tokenizer, out_path)
