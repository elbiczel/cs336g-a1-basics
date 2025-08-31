from cs336_tokenizer import bpe
from token_utils import gpt2_bytes_to_unicode, save_vocab_and_merges
from utils import stopwatch


pre_tokenizer_pattern = None # r"""(?:\w+)"""

should_print = False
vocab_size = 10_000
prefix = 'data/bpe_sample'
prefix = 'data/TinyStoriesV2-GPT4-valid'
#prefix = 'data/TinyStoriesV2-GPT4-train'
#vocab_size = 32_000
#prefix = 'data/owt_valid'
#prefix = 'data/owt-train'

if __name__ == "__main__":
    vocab, merge_list = stopwatch(bpe.train_bpe)(
        input_path=f"{prefix}.txt",
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        num_processes=16,
        pre_tokenizer_pattern = pre_tokenizer_pattern
    )

    if should_print:
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
    else:
        save_vocab_and_merges(vocab, merge_list, vocab_path=f'{prefix}-vocab.json', merges_path=f'{prefix}-merges.txt')

