from cs336_basics import io, utils
from cs336_tokenizer.bpe import Tokenizer, TrieTokenizer, MergeTokenizer
from typing import BinaryIO, Iterable, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

prefix = 'data/TinyStoriesV2-GPT4-valid'
prefix = 'data/TinyStoriesV2-GPT4-train'
prefix = 'data/owt_valid'
prefix = 'data/owt_train'

data_path = f"{prefix}.txt"
vocab_path = f"{prefix}-vocab.json"
merges_path = f"{prefix}-merges.txt"
out_path = f"{prefix}-trie-tokenized"

special_token = b"<|endoftext|>"
special_token_str = special_token.decode("utf-8")

def iterate_docs(f: BinaryIO, start: int, end: int) -> Iterable[str]:
    for doc in io.docs_in_range(f, special_token, start, end):
        yield doc.decode("utf-8", errors="ignore")

def tokenize_chunk(tokenizer: Tokenizer, path: str | os.PathLike, start: int, end: int) -> Tuple[int, list[int]]:
    out = []
    with open(path, "rb") as f:
        out.extend(tokenizer.encode_iterable(iterate_docs(f, start, end)))
    return (start, out)

def run_tokenize(tokenizer: Tokenizer, path: str | os.PathLike, num_processes=1) -> list[int]:
    with open(path, "rb") as f:
        boundaries = io.find_chunk_boundaries(f, num_processes, special_token)
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        chunk_tokens_futures = [pool.submit(tokenize_chunk, tokenizer, path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    data = {}
    for fut in as_completed(chunk_tokens_futures):
        start, tokens = fut.result()
        data[start] = tokens
    out = []
    for k in sorted(data.keys()):
        out.extend(data[k])
    return out

run_tokenize = utils.stopwatch(run_tokenize)

if __name__ == "__main__":
    tokenizer = TrieTokenizer.from_files(vocab_path, merges_path, special_tokens=[special_token])
    encoded = run_tokenize(tokenizer, data_path, 10)
    print("Number of tokens: ", len(encoded))
    io.write_tokens(out_path, encoded)
