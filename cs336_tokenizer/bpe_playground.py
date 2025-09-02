from bpe import Tokenizer
from cs336_basics import io
import time

tiny_stories_tokenizer = Tokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/data/TinyStoriesV2-GPT4-train-vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/data/TinyStoriesV2-GPT4-train-merges.txt",
    special_tokens=[b"<|endoftext|>"],
)
owt_tokenizer = Tokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/data/owt_train-vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/data/owt_train-merges.txt",
    special_tokens=[b"<|endoftext|>"],
)

tiny_stories_sample = io.sample_docs(
    "/Users/tomek/Coding/cs336g-a1-basics/data/TinyStoriesV2-GPT4-train.txt",
    10, b"<|endoftext|>")
owt_sample = io.sample_docs(
    "/Users/tomek/Coding/cs336g-a1-basics/data/owt_train.txt",
    10, b"<|endoftext|>")

print("Loaded data...")

def get_stats(docs, tokenizer):
    token_num = 0
    byte_length = 0
    duration = 0
    for doc in docs:
        start_time = time.time()
        tokens = tokenizer.encode(doc)
        end_time = time.time()
        duration += end_time - start_time
        token_num += len(tokens)
        byte_length += len(doc.encode("utf-8"))
    return byte_length / token_num, (byte_length * 1e-6) / duration


print("TS Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(tiny_stories_sample, tiny_stories_tokenizer))
print("OWT Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(owt_sample, owt_tokenizer))

print("TS-Tokenizer on OWT data Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(owt_sample, tiny_stories_tokenizer))
print("OWT-Tokenizer on TS Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(tiny_stories_sample, owt_tokenizer))

### Outputs:
# TS Compression ratio: 4.099 Bytes per Token. 0.022 MB/Sec
# OWT Compression ratio: 4.628 Bytes per Token. 0.007 MB/Sec
# TS-Tokenizer on OWT data Compression ratio: 3.196 Bytes per Token. 0.012 MB/Sec
# OWT-Tokenizer on TS Compression ratio: 3.903 Bytes per Token. 0.006 MB/Sec

print(tiny_stories_sample[0])
print(tiny_stories_tokenizer.encode(tiny_stories_sample[0]))