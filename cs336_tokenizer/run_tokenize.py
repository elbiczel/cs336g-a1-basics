from cs336_basics import io, utils
from cs336_tokenizer.bpe import Tokenizer, TrieTokenizer, MergeTokenizer
from typing import Iterable
import shutil
from tempfile import TemporaryDirectory

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

prefix = 'data/TinyStoriesV2-GPT4-valid'
prefix = 'data/owt_valid'
prefix = 'data/TinyStoriesV2-GPT4-train'
prefix = 'data/owt_train'

data_path = f"{prefix}.txt"
vocab_path = f"{prefix}-vocab.json"
merges_path = f"{prefix}-merges.txt"
out_path = f"{prefix}-tokenized.dat"

TokenizerCls = MergeTokenizer

special_token = b"<|endoftext|>"
special_token_str = special_token.decode("utf-8")

def iterate_docs(path: str | os.PathLike, start: int, end: int) -> Iterable[str]:
    with open(path, "rb") as f:
        for doc in io.docs_in_range(f, special_token, start, end):
            yield doc.decode("utf-8", errors="ignore")

def tokenize_chunk(tokenizer: Tokenizer, tmpdir: str, path: str | os.PathLike, i: int, start: int, end: int) -> int:
    tmp_path = os.path.join(tmpdir, f"out_{i:03d}.dat")
    num_tokens = 0
    with io.TokenWriter(tmp_path) as out:
        for tok in tokenizer.encode_iterable(iterate_docs(path, start, end)):
            num_tokens += 1
            out.write(tok)
    return num_tokens

def run_tokenize(tokenizer: Tokenizer, path: str | os.PathLike, num_processes=1) -> int:
    with open(path, "rb") as f:
        boundaries = io.find_chunk_boundaries(f, num_processes, special_token)
    with TemporaryDirectory() as tmpdir:
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            chunk_tokens_futures = [pool.submit(tokenize_chunk, tokenizer, tmpdir, path, i, start, end) for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
        num_tokens = 0
        for fut in as_completed(chunk_tokens_futures):
            num_tokens += fut.result()
        tmp_files = sorted(os.listdir(tmpdir))
        with open(out_path, "wb") as wfd:
            for fname in tmp_files:
                fpath = os.path.join(tmpdir, fname)
                with open(fpath, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)
    return num_tokens

run_tokenize = utils.stopwatch(run_tokenize)

if __name__ == "__main__":
    tokenizer = TokenizerCls.from_files(vocab_path, merges_path, special_tokens=[special_token])
    print("Tokenizing: ", data_path)
    total_tokens = run_tokenize(tokenizer, data_path, 10)
    print("Number of tokens: ", total_tokens)
