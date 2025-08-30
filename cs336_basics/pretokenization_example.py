from cs336_tokenizer import bpe
from token_utils import gpt2_bytes_to_unicode


pre_tokenizer_pattern = r"""(?:\w+)"""
## Usage
vocab, merge_list = bpe.train_bpe(
    input_path="data/bpe_sample2.txt",
    vocab_size=256 + 1 + 6,
    special_tokens=["<|endoftext|>"],
    num_processes=1,
    pre_tokenizer_pattern = pre_tokenizer_pattern
)

byte_to_unicode = gpt2_bytes_to_unicode()

string_vocab = {
    "".join([byte_to_unicode[b] for b in byte_token]): k
    for k, byte_token in vocab.items()
}

print(string_vocab)
print("----")

string_merges = [
    f"{''.join([byte_to_unicode[b] for b in merge[0]])} "
    f"{''.join([byte_to_unicode[b] for b in merge[1]])}"
    for merge in merge_list
]

print(merge_list)

