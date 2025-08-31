from bpe import Tokenizer

tokenizer = Tokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/tests/fixtures/gpt2_vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/tests/fixtures/gpt2_merges.txt",
    special_tokens=[b"<|endoftext|>", b"<|endoftext|><|endoftext|>"],
)
test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
tokens = tokenizer.encode(test_string)
print(tokens)

tokenized_string = [tokenizer.decode([x]) for x in tokens]
print(tokenized_string)


tokenizer = Tokenizer.from_files(
    "/Users/tomek/Coding/cs336g-a1-basics/tests/fixtures/gpt2_vocab.json",
    "/Users/tomek/Coding/cs336g-a1-basics/tests/fixtures/gpt2_merges.txt",
    special_tokens=[b"<|endoftext|>"],
)
test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
tokens = tokenizer.encode(test_string)
print(tokens)

tokenized_string = [tokenizer.decode([x]) for x in tokens]
print(tokenized_string)
