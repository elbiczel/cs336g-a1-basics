from bpe import TrieTokenizer, MergeTokenizer
from cs336_basics import io
import time
import os


prefixes = ['data/TinyStoriesV2-GPT4-valid', 'data/TinyStoriesV2-GPT4-train', 'data/owt_valid', 'data/owt_train']
for prefix in prefixes:
    original_path = f"{prefix}.txt"
    decoded_path = f"{prefix}-decoded.txt"
    trie_decoded_path = f"{prefix}-trie-decoded.txt"
    with open(original_path, "rb") as f:
        orig_content = f.read()
    if os.path.exists(decoded_path):
        with open(decoded_path, "rb") as f:
            decoded_content = f.read()
        print("Prefix: ", prefix, " matches: ", orig_content == decoded_content)
    else:
        print("Missing decoded data")
    if os.path.exists(trie_decoded_path):
        with open(trie_decoded_path, "rb") as f:
            trie_decoded_content = f.read()
        print("Prefix: ", prefix, " trie-matches: ", orig_content == trie_decoded_content)
    else:
        print("Missing trie decoded data")

