from bpe import TrieTokenizer, MergeTokenizer
from cs336_basics import io
import time

tiny_stories_tokenizer = TrieTokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/data/TinyStoriesV2-GPT4-train-vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/data/TinyStoriesV2-GPT4-train-merges.txt",
    special_tokens=[b"<|endoftext|>"],
)
owt_tokenizer = TrieTokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/data/owt_train-vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/data/owt_train-merges.txt",
    special_tokens=[b"<|endoftext|>"],
)
tiny_stories_merge_tokenizer = MergeTokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/data/TinyStoriesV2-GPT4-train-vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/data/TinyStoriesV2-GPT4-train-merges.txt",
    special_tokens=[b"<|endoftext|>"],
)
owt_merge_tokenizer = MergeTokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/data/owt_train-vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/data/owt_train-merges.txt",
    special_tokens=[b"<|endoftext|>"],
)

tiny_stories_sample = io.sample_docs(
    "/Users/tomek/Coding/cs336g-a1-basics/data/TinyStoriesV2-GPT4-train.txt",
    2000, b"<|endoftext|>")
owt_sample = io.sample_docs(
    "/Users/tomek/Coding/cs336g-a1-basics/data/owt_train.txt",
    2000, b"<|endoftext|>")

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


print("Trie tokenizer: ")
print("TS Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(tiny_stories_sample, tiny_stories_tokenizer))
print("OWT Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(owt_sample, owt_tokenizer))
print("TS-Tokenizer on OWT data Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(owt_sample, tiny_stories_tokenizer))
print("OWT-Tokenizer on TS Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(tiny_stories_sample, owt_tokenizer))

print("Merge tokenizer: ")
print("TS Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(tiny_stories_sample, tiny_stories_merge_tokenizer))
print("OWT Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(owt_sample, owt_merge_tokenizer))
print("TS-Tokenizer on OWT data Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(owt_sample, tiny_stories_merge_tokenizer))
print("OWT-Tokenizer on TS Compression ratio: %.3f Bytes per Token. %.3f MB/Sec" % get_stats(tiny_stories_sample, owt_merge_tokenizer))

### Outputs:
# Trie tokenizer: 
# TS Compression ratio: 4.145 Bytes per Token. 6.906 MB/Sec
# OWT Compression ratio: 4.326 Bytes per Token. 7.398 MB/Sec
# TS-Tokenizer on OWT data Compression ratio: 3.249 Bytes per Token. 7.011 MB/Sec
# OWT-Tokenizer on TS Compression ratio: 4.019 Bytes per Token. 7.319 MB/Sec

# Merge tokenizer:
# TS Compression ratio: 4.073 Bytes per Token. 0.672 MB/Sec
# OWT Compression ratio: 4.381 Bytes per Token. 0.599 MB/Sec
# TS-Tokenizer on OWT data Compression ratio: 3.190 Bytes per Token. 0.688 MB/Sec
# OWT-Tokenizer on TS Compression ratio: 3.961 Bytes per Token. 0.627 MB/Sec

#print(tiny_stories_sample[0])
#print(tiny_stories_tokenizer.encode(tiny_stories_sample[0]))

test_tokenizer = MergeTokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/tests/fixtures/gpt2_vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/tests/fixtures/gpt2_merges.txt",
    special_tokens=None,
)
test_string = """Die Leland"""
encoded_ids = test_tokenizer.encode(test_string)
print("Encoded ids: ", encoded_ids)
decoded_string = test_tokenizer.decode(encoded_ids)

import tiktoken
reference_tokenizer = tiktoken.get_encoding("gpt2")
reference_ids = reference_tokenizer.encode(test_string)
print("Ref ids: ", reference_ids)
print("Decoded ref: ", test_tokenizer.decode(reference_ids))
