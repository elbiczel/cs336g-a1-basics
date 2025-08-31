import os
import regex as re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import BinaryIO, Dict, Tuple, Iterator, Optional, Iterable, Iterable

from cs336_basics.common_types import BytePair, MergeList, Vocab
from cs336_basics import token_utils

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

def _get_pretokenized_sequence(text: str, specials_re, pre_tokenization_re, yield_specials: bool = True) -> Iterator[str]:
    special_iter = specials_re.finditer(text)
    pos = 0
    for sm in special_iter:
        # normal chunk before the special
        for tm in pre_tokenization_re.finditer(text, pos, sm.start()):
            yield tm.group(0)
        # the special itself
        if yield_specials:
            yield sm.group(0)
        pos = sm.end()
    # tail after the last special
    for tm in pre_tokenization_re.finditer(text, pos):
        yield tm.group(0)

def _get_chunk_freqs(input_path: str | os.PathLike, start: int, end: int, specials_re, pre_tokenization_re) -> Dict[Tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        freqs = defaultdict(int)
        for pre_token in _get_pretokenized_sequence(chunk, specials_re, pre_tokenization_re, False):
            freqs[_pretoken_to_key(pre_token)] += 1
        return freqs

def _merge_freqs(master: Dict[Tuple[bytes, ...], int], additional: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
    for k, v in additional.items():
        master[k] += v
    return master

# 1 call - 48s with 16 processes on TinyStories train dataset.
def _pre_tokenize_input(input_path: str | os.PathLike, special_token: bytes, specials_re, pre_tokenization_re, num_processes) -> Dict[Tuple[bytes, ...], int]:
    freqs = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token)
    
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        chunk_freqs_futures = [pool.submit(_get_chunk_freqs, input_path, start, end, specials_re, pre_tokenization_re) for start, end in zip(boundaries[:-1], boundaries[1:])]
    for fut in as_completed(chunk_freqs_futures):
        _merge_freqs(freqs, fut.result())
    return freqs

def _byte_pairs(b: Tuple[bytes, ...]) -> Iterator[BytePair]:
    # Only for len(b) > 2
    # Zero-copy adjacent pairs (no slicing)
    it = iter(b)
    prev = next(it)
    for cur in it:
        yield (prev, cur)
        prev = cur

# Irrelevant.
def _init_byte_pair_freqs(freqs: Dict[Tuple[bytes, ...], int]) -> Dict[BytePair, int]:
    counts: Dict[BytePair, int] = {}
    for seq, freq in freqs.items():
        if len(seq) < 2: continue
        # Inline the tight loop; use locals for speed
        get = counts.get
        setitem = counts.__setitem__
        for bp in _byte_pairs(seq):
            setitem(bp, get(bp, 0) + freq)
    return counts


# Irrelevant.
def _find_merge(byte_pair_freqs: Dict[BytePair, int]) -> BytePair | None:
    best_pair: Optional[BytePair] = None
    best_count = 2
    for bp, c in byte_pair_freqs.items():
        if c > best_count or (c == best_count and (best_pair is None or bp > best_pair)):
            best_count = c
            best_pair = bp
    return best_pair

# Total time: 30.7s with 127.7M calls on TinyStories Valid dataset.
# Total time: 160s with 58.4M calls on TinyStories Train dataset.
def _apply_merge_to_seq(seq: Tuple[bytes, ...], merge: BytePair) -> tuple[Tuple[bytes, ...], Dict[BytePair, int]]:
    ret_value = []
    prev_merge = 0
    i = 0
    deltas = defaultdict(int)
    seq_len = len(seq)
    while i < seq_len - 1:
        if seq[i] == merge[0] and seq[i+1] == merge[1]:
            ret_value += seq[prev_merge:i]
            new_value = merge[0] + merge[1]
            ret_value.append(new_value)
            if i > 0:
                deltas[(seq[i-1], seq[i])] -= 1
                deltas[(seq[i-1], new_value)] += 1
            if i < seq_len - 2:
                deltas[(seq[i+1], seq[i+2])] -= 1
                deltas[(new_value, seq[i+2])] += 1
            i += 1
            prev_merge = i + 1
        i += 1
    if not prev_merge:
        return (), {}
    ret_value += seq[prev_merge:]
    return tuple(ret_value), deltas

# Total time: 60.2s with 10k calls, each 0.0075s on TinyStories Valid dataset.
# Total time: 302s with 10k calls, each 0.031s on TinyStories Train dataset. ~84% of execution time.
def _apply_merge(
        freqs: Dict[Tuple[bytes, ...], int],
        byte_pair_freqs: Dict[BytePair, int],
        merge: BytePair) -> tuple[Dict[Tuple[bytes, ...], int], Dict[BytePair, int]]:
    new_freqs = defaultdict(int, freqs)
    del byte_pair_freqs[merge]
    # TODO: This can be optimized further by keeping track of bp occurances.
    for seq, freq in freqs.items():
        new_seq, deltas = _apply_merge_to_seq(seq, merge)
        if new_seq:
            del new_freqs[seq]
            new_freqs[new_seq] += freq
        for bp, delta in deltas.items():
            byte_pair_freqs[bp] = byte_pair_freqs.get(bp, 0) + delta * freq
            if byte_pair_freqs[bp] == 0:
                del byte_pair_freqs[bp]
    return new_freqs, byte_pair_freqs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[Vocab, MergeList]:
    num_processes = kwargs["num_processes"]
    pre_tokenization_re = kwargs["pre_tokenizer_pattern"] if "pre_tokenizer_pattern" in kwargs else None
    pre_tokenization_re = pre_tokenization_re or PAT
    pre_tokenization_re = re.compile(pre_tokenization_re)
    special_token = b"<|endoftext|>" if "<|endoftext|>" in special_tokens else special_tokens[0].encode("utf-8")
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    specials_re = re.compile("(" + "|".join(re.escape(t) for t in special_tokens) + ")")

    # Basic vocab.
    vocab = {i: t.encode("utf-8") for i, t in enumerate(special_tokens)}
    vocab |= {i+len(vocab): bytes([i]) for i in range(256)}
    
    # Pre tokenize the input.
    freqs = _pre_tokenize_input(input_path, special_token, specials_re, pre_tokenization_re, num_processes)

    # Find merges
    merge_list = []
    byte_pair_freqs = _init_byte_pair_freqs(freqs)
    for _ in range(vocab_size - len(vocab)):
        merge = _find_merge(byte_pair_freqs)
        if not merge:
            break
        merge_list.append(merge)
        vocab[len(vocab)] = merge[0] + merge[1]
        freqs, byte_pair_freqs = _apply_merge(freqs, byte_pair_freqs, merge)

    return (vocab, merge_list)

class Tokenizer:
    MALFORMED_CHAR_BYTES = "�".encode("utf-8")

    def __init__(self, vocab: Vocab, merges: MergeList, special_tokens: list[bytes] | None = None):
        self._vocab = dict(vocab)
        self._merges = list(merges)
        self._sepcial_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self.init()

    @classmethod
    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[bytes] | None = None):
        vocab, merge_list = token_utils.load_vocab_and_merges(vocab_filepath, merges_filepath)
        return Tokenizer(vocab, merge_list, special_tokens)

    def init(self):
        self._pre_tokenization_re = re.compile(PAT)
        if self._sepcial_tokens:
            self._specials_re = re.compile("(" + "|".join(re.escape(t.decode("utf-8")) for t in self._sepcial_tokens) + ")")
        else:
            self._specials_re = re.compile(r'a^')
        self._reverse = {seq: token for token, seq in self._vocab.items()}
        if self._sepcial_tokens:
            for t in self._sepcial_tokens:
                if t not in self._reverse:
                    self._reverse[t] = len(self._vocab)
                    self._vocab[len(self._vocab)] = t
        # TODO: Drop _merges and construct a Radix Tree out of vocab for encode optimization.

    def encode(self, text: str) -> list[int]:
        out = []

        for pre_token in _get_pretokenized_sequence(text, self._specials_re, self._pre_tokenization_re):
            if re.match(self._specials_re, pre_token):
                out.append(self._reverse[pre_token.encode("utf-8")])
                continue
            seq = list(_pretoken_to_key(pre_token))
            for merge in self._merges:
                i = 0
                while i < len(seq) - 1:
                    if seq[i] == merge[0] and seq[i+1] == merge[1]:
                        seq = seq[:i] + [merge[0] + merge[1]] + seq[i+2:]
                    i += 1
                        
            for b in seq:
                out.append(self._reverse[b])

        return out
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token
    
    def decode(self, ids: list[int]) -> str:
        return b"".join([self._vocab.get(id, self.MALFORMED_CHAR_BYTES) for id in ids]).decode("utf-8", errors="replace")
