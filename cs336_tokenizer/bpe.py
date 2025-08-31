import os
import regex as re
from collections import defaultdict
from typing import BinaryIO, Dict, Tuple, Iterator, Optional

from cs336_basics.common_types import BytePair, MergeList, Vocab

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _pretoken_to_key(text: str) -> Tuple[bytes, ...]:
    return tuple([bytes([x]) for x in text.encode("utf-8")])

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _get_chunk_freqs(text: str, specials_re, pre_tokenization_re) -> Dict[Tuple[bytes, ...], int]:
    freqs = defaultdict(int)
    parts = re.split(specials_re, text)
    for part in parts:
        for pre_token in re.finditer(pre_tokenization_re, part):
            freqs[_pretoken_to_key(pre_token.group())] += 1
    return freqs

def _merge_freqs(master: Dict[Tuple[bytes, ...], int], additional: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
    for k, v in additional.items():
        master[k] += v
    return master

# 1 call - 4.5s on TinyStories valid dataset.
# 1 call - 464s on TinyStories train dataset. Becomes ~30% of program execution.
def _pre_tokenize_input(input_path: str | os.PathLike, special_token: bytes, specials_re, pre_tokenization_re, num_processes) -> Dict[Tuple[bytes, ...], int]:
    freqs = defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token)
        # TODO: Parallelize.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_freqs = _get_chunk_freqs(chunk, specials_re, pre_tokenization_re)
            _merge_freqs(freqs, chunk_freqs)
    return freqs

def _byte_pairs(b: Tuple[bytes, ...]) -> Iterator[BytePair]:
    # Only for len(b) > 2
    # Zero-copy adjacent pairs (no slicing)
    it = iter(b)
    prev = next(it)
    for cur in it:
        yield (prev, cur)
        prev = cur

# Total time: 77s with 10k calls, each 0.0079s on TinyStories Valid dataset.
# Total time: 686s with 10k calls, each 0.079s on TinyStories Train dataset. ~60% of total execution time.
def _find_merge(freqs: Dict[Tuple[bytes, ...], int]) -> BytePair | None:
    counts: Dict[BytePair, int] = {}
    best_pair: Optional[BytePair] = None
    best_count = 0
    for seq, freq in freqs.items():
        if freq <= 0 or len(seq) < 2:
            continue
        # Inline the tight loop; use locals for speed
        get = counts.get
        setitem = counts.__setitem__

        for bp in _byte_pairs(seq):
            c = get(bp, 0) + freq
            setitem(bp, c)

            if c > best_count or (c == best_count and (best_pair is None or bp > best_pair)):
                best_count = c
                best_pair = bp
    
    if best_count <= 1:
        return None
    return best_pair

def _apply_merge_to_bytes(b: Tuple[bytes, ...], merge: BytePair) -> Tuple[bytes, ...]:
    ret_value = []
    prev_merge = 0
    i = 0
    while i < len(b) - 1:
        if b[i] == merge[0] and b[i+1] == merge[1]:
            ret_value += b[prev_merge:i]
            ret_value.append(merge[0] + merge[1])
            i += 1
            prev_merge = i + 1
        i += 1
    if not prev_merge:
        return ()
    ret_value += b[prev_merge:]
    return tuple(ret_value)

# Total time: 43s with 10k calls, each 0.0044s on TinyStories Valid dataset.
# Total time: 256s with 10k calls, each 0.026s on TinyStories Train dataset. <30% of execution time.
def _apply_merge(freqs: Dict[Tuple[bytes, ...], int], merge: BytePair) -> Dict[Tuple[bytes, ...], int]:
    new_freqs = defaultdict(int, freqs)
    for b, freq in freqs.items():
        new_b = _apply_merge_to_bytes(b, merge)
        if new_b:
            del new_freqs[b]
            new_freqs[new_b] += freq
    return new_freqs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[Vocab, MergeList]:
    num_processes = kwargs["num_processes"]
    pre_tokenization_re = kwargs["pre_tokenizer_pattern"] if "pre_tokenizer_pattern" in kwargs else PAT
    pre_tokenization_re = re.compile(pre_tokenization_re)
    special_token = b"<|endoftext|>" if "<|endoftext|>" in special_tokens else special_tokens[0].encode("utf-8")
    specials_re = re.compile("(?:" + "|".join(re.escape(t) for t in special_tokens) + ")")

    # Basic vocab.
    vocab = {i: t.encode("utf-8") for i, t in enumerate(special_tokens)}
    vocab |= {i+len(vocab): bytes([i]) for i in range(256)}
    
    # Pre tokenize the input.
    freqs = _pre_tokenize_input(input_path, special_token, specials_re, pre_tokenization_re, num_processes)

    # Find merges
    merge_list = []
    for _ in range(vocab_size - len(vocab)):
        merge = _find_merge(freqs)
        if not merge:
            break
        merge_list.append(merge)
        vocab[len(vocab)] = merge[0] + merge[1]
        freqs = _apply_merge(freqs, merge)

    return (vocab, merge_list)
