from cs336_basics import io, utils
from cs336_tokenizer.bpe import Tokenizer, TrieTokenizer, MergeTokenizer

from concurrent.futures import ProcessPoolExecutor, as_completed

prefix = 'data/TinyStoriesV2-GPT4-valid'
prefix = 'data/TinyStoriesV2-GPT4-train'
prefix = 'data/owt_valid'
prefix = 'data/owt_train'

vocab_path = f"{prefix}-vocab.json"
merges_path = f"{prefix}-merges.txt"
in_path = f"{prefix}-trie-tokenized.npy"
out_path = f"{prefix}-trie-decoded.txt"

special_token = b"<|endoftext|>"
special_token_str = special_token.decode("utf-8")

if __name__ == "__main__":
    tokenizer = TrieTokenizer.from_files(vocab_path, merges_path, special_tokens=None)
    run_decode = utils.stopwatch(tokenizer.decode)
    tokens = io.read_tokens(in_path)
    with open(out_path, "wb") as f:
        chunk = run_decode(tokens)
        f.write(chunk.encode("utf-8"))
