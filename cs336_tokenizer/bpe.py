import os
import regex as re
import heapq
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, Iterator, Iterable, Iterable, List

from cs336_basics import io, token_utils
from cs336_basics.utils import stopwatch
from cs336_basics.common_types import BytePair, MergeList, Vocab

def _pretoken_to_key(text: str) -> Tuple[bytes, ...]:
    return tuple([bytes([x]) for x in text.encode("utf-8")])

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _get_pretokenized_sequence(text: str, specials_re, pre_tokenization_re, yield_specials: bool = True) -> Iterator[Tuple[str, bool]]:
    pos = 0
    for sm in specials_re.finditer(text) if specials_re else []:
        # normal chunk before the special
        for tm in pre_tokenization_re.finditer(text, pos, sm.start()):
            yield (tm.group(0), False)
        # the special itself
        if yield_specials:
            yield (sm.group(0), True)
        pos = sm.end()
    # tail after the last special
    for tm in pre_tokenization_re.finditer(text, pos):
        yield (tm.group(0), False)

def _get_chunk_freqs(
        input_path: str | os.PathLike,
        start: int,
        end: int,
        special_token: bytes,
        specials_re,
        pre_tokenization_re) -> Dict[Tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        freqs = defaultdict(int)
        for doc in chunk.split(special_token):
            doc = doc.decode("utf-8", errors="ignore")
            for pre_token, _ in _get_pretokenized_sequence(doc, specials_re, pre_tokenization_re, False):
                freqs[_pretoken_to_key(pre_token)] += 1
        return freqs

def _merge_freqs(master: Dict[Tuple[bytes, ...], int], additional: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
    for k, v in additional.items():
        master[k] += v
    return master

# 1 call - ~50s with 16 processes on TinyStories train dataset.
def _pre_tokenize_input(
        input_path: str | os.PathLike,
        special_token: bytes,
        specials_re,
        pre_tokenization_re,
        num_processes) -> List[Tuple[Tuple[bytes, ...], int]]:
    # Note: Interenstingly usage of Counter was slower here.
    freqs = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = io.find_chunk_boundaries(f, num_processes, special_token)
    
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        chunk_freqs_futures = [pool.submit(_get_chunk_freqs, input_path, start, end, special_token, specials_re, pre_tokenization_re) for start, end in zip(boundaries[:-1], boundaries[1:])]
    for fut in as_completed(chunk_freqs_futures):
        _merge_freqs(freqs, fut.result())
    return [(k, v) for k, v in freqs.items()]

def _byte_pairs(b: Tuple[bytes, ...]) -> Iterator[BytePair]:
    # Only for len(b) > 2
    # Zero-copy adjacent pairs (no slicing)
    it = iter(b)
    prev = next(it)
    for cur in it:
        yield (prev, cur)
        prev = cur


class NegBytePair:
    def __init__(self, data: BytePair): self.data = data
    def __lt__(self, other):  # reverse lexicographic for max tie-breaks
        return self.data > other.data
    def __repr__(self): return repr(self.data)


class BytePairQueue:
    def __init__(self, bp_counts: Dict[BytePair, int]):
        self.heap = []  # entries: (-freq, NegTuple(data), version)
        self.entry = {}  # bp -> (freq, current_version)
        for bp, freq in bp_counts.items():
            self.entry[bp] = (freq, 0)
            self.heap.append((-freq, NegBytePair(bp), 0))
        heapq.heapify(self.heap)

    def _discard_stale(self):
        # pop stale entries until top is current
        while self.heap:
            neg_freq, neg_bp, ver = self.heap[0]
            cur_freq, cur_ver = self.entry[neg_bp.data]
            if cur_ver == ver and -neg_freq == cur_freq:
                return                 # top is fresh
            heapq.heappop(self.heap)

    def pop(self) -> BytePair | None:
        self._discard_stale()
        if not self.heap: return None
        neg_freq, neg_bp, _ = heapq.heappop(self.heap)
        if neg_freq > -2:
            # Nothing more to merge.
            return None
        del self.entry[neg_bp.data]
        return neg_bp.data
    
    def insert_or_inc(self, bp: BytePair, delta: int):
        freq, v = self.entry.get(bp, (0, 0))
        freq += delta
        v += 1
        self.entry[bp] = (freq, v)
        heapq.heappush(self.heap, (-freq, NegBytePair(bp), v))


def _init_byte_pair_freqs(sequences: List[Tuple[Tuple[bytes, ...], int]]) -> tuple[BytePairQueue, Dict[BytePair, Counter[int]]]:
    counts: Dict[BytePair, int] = {}
    occurances: Dict[BytePair, Counter[int]] = {}
    # Inline the tight loop; use locals for speed
    get = counts.get
    setitem = counts.__setitem__

    for i, (seq, freq) in enumerate(sequences):
        if len(seq) < 2: continue
        for bp in _byte_pairs(seq):
            # We could track the offset of the bp in occurances as well.
            # This would allow us for simpler apply_merge_to_seq implementation.
            # But would require dynamic updates to the offset after merging.
            occurances.setdefault(bp, Counter())[i] += 1
            setitem(bp, get(bp, 0) + freq)
    # Remove the tail of rare byte pairs that will never be used.
    for bp in [bp for (bp, freq) in counts.items() if freq == 1]:
        del counts[bp]
        del occurances[bp]
    return BytePairQueue(counts), occurances


def _apply_merge_to_seq(seq: Tuple[bytes, ...], merge: BytePair) -> tuple[Tuple[bytes, ...], Dict[BytePair, int]]:
    ret_value = []
    post_prev_merge = 0
    i = 0
    deltas = defaultdict(int)
    seq_len = len(seq)
    new_value = merge[0] + merge[1]
    while i < seq_len - 1:
        if seq[i] == merge[0] and seq[i+1] == merge[1]:
            ret_value.extend(seq[post_prev_merge:i])
            ret_value.append(new_value)
            if i > 0:
                deltas[(seq[i-1], seq[i])] -= 1
                deltas[(seq[i-1], new_value)] += 1
            if i < seq_len - 2:
                deltas[(seq[i+1], seq[i+2])] -= 1
                deltas[(new_value, seq[i+2])] += 1
            i += 1
            post_prev_merge = i + 1
        i += 1
    if post_prev_merge < seq_len:
        ret_value.extend(seq[post_prev_merge:])
    return tuple(ret_value), deltas

def _apply_merge(
        sequences: List[Tuple[Tuple[bytes, ...], int]],
        occurances: Dict[BytePair, Counter[int]],
        bp_queue: BytePairQueue,
        merge: BytePair):
    seq_ids = set(occurances[merge].keys())
    # Computing deltas across all sequences to reduce heap operations.
    deltas = defaultdict(int)
    for i in seq_ids:
        seq, freq = sequences[i]
        new_seq, seq_deltas = _apply_merge_to_seq(seq, merge)
        if new_seq:
            sequences[i] = (new_seq, freq)
        for bp, delta in seq_deltas.items():
            deltas[bp] += delta * freq
            occurances.setdefault(bp, Counter())[i] += delta
    for bp, delta in deltas.items():
        bp_queue.insert_or_inc(bp, delta)
    del occurances[merge]


def _train_bpe(vocab_size: int,
               vocab: dict[int, bytes],
               sequences: List[Tuple[Tuple[bytes, ...], int]]) -> tuple[Vocab, MergeList]:
    merge_list = []
    bp_queue, occurances = _init_byte_pair_freqs(sequences)
    for _ in range(vocab_size - len(vocab)):
        # Note: This could be optimized using a heap.
        merge = bp_queue.pop()
        if not merge:
            break
        merge_list.append(merge)
        vocab[len(vocab)] = merge[0] + merge[1]
        # Changes input collections in place.
        _apply_merge(sequences, occurances, bp_queue, merge)
    return (vocab, merge_list)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[Vocab, MergeList]:
    # Basic vocab.
    vocab = {i: t.encode("utf-8") for i, t in enumerate(special_tokens)}
    vocab |= {i+len(vocab): bytes([i]) for i in range(256)}

    num_processes = kwargs["num_processes"]
    pre_tokenization_re = kwargs["pre_tokenizer_pattern"] if "pre_tokenizer_pattern" in kwargs else None
    pre_tokenization_re = pre_tokenization_re or PAT
    pre_tokenization_re = re.compile(pre_tokenization_re)
    special_token = "<|endoftext|>" if "<|endoftext|>" in special_tokens else special_tokens[0]
    
    special_tokens = [t for t in special_tokens if t != special_token]
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    specials_re = re.compile("(" + "|".join(re.escape(t) for t in special_tokens) + ")") if special_tokens else None
    special_token = special_token.encode("utf-8")

    sequences = stopwatch(_pre_tokenize_input)(input_path, special_token, specials_re, pre_tokenization_re, num_processes)
    return stopwatch(_train_bpe)(vocab_size, vocab, sequences)

class VocabNode:
    def __init__(self, token: int):
        self.edges = [None] * 256
        self.token = token

class VocabTrie:
    def __init__(self, reverse: dict[bytes, int], special_tokens: list[bytes]):
        self._root = VocabNode(-1)
        for seq, token in reverse.items():
            if seq in special_tokens: continue
            self._add_node(seq, token)

    def next_token(self, seq: bytes, i: int) -> Tuple[int, int]:
        seq_len = len(seq)
        node = self._root
        last_good_node = node
        last_good_i = i
        while i < seq_len:
            next = node.edges[seq[i]]
            if next == None:
                return (last_good_node.token, last_good_i)
            node = next
            i += 1
            if node.token != -1:
                last_good_node = node
                last_good_i = i
        return (last_good_node.token, last_good_i)

    def _add_node(self, seq: bytes, token: int):
        node = self._root
        for chr in seq:
            next = node.edges[chr]
            if next == None:
                # Init with -1, it will be overriden later.
                next = VocabNode(-1)
                node.edges[chr] = next
            node = next
        node.token = token

class TrieTokenizer:
    MALFORMED_CHAR_BYTES = "�".encode("utf-8")

    def __init__(self, vocab: Vocab, _: MergeList, special_tokens: list[bytes] | None = None):
        self._vocab = dict(vocab)
        self._sepcial_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        self._init()

    @classmethod
    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[bytes] | None = None):
        vocab, merge_list = token_utils.load_vocab_and_merges(vocab_filepath, merges_filepath)
        return TrieTokenizer(vocab, merge_list, special_tokens)

    def _init(self):
        self._pre_tokenization_re = re.compile(PAT)
        if self._sepcial_tokens:
            self._specials_re = re.compile("(" + "|".join(re.escape(t.decode("utf-8")) for t in self._sepcial_tokens) + ")")
        else:
            self._specials_re = None
        reverse = {seq: token for token, seq in self._vocab.items()}
        self._special_mapping = {}
        for t in self._sepcial_tokens:
            if t not in reverse:
                reverse[t] = len(self._vocab)
                self._vocab[len(self._vocab)] = t
            self._special_mapping[t] = reverse[t]
        self._trie = VocabTrie(reverse, self._sepcial_tokens)

    def encode(self, text: str) -> list[int]:
        out = []

        for pre_token, is_special in _get_pretokenized_sequence(text, self._specials_re, self._pre_tokenization_re):
            pre_token = pre_token.encode("utf-8")
            if is_special:
                out.append(self._special_mapping[pre_token])
                continue
            seq_len = len(pre_token)
            i = 0
            while i < seq_len:
                token, i = self._trie.next_token(pre_token, i)
                out.append(token)

        return out
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token
    
    def decode(self, ids: list[int]) -> str:
        return b"".join([self._vocab.get(id, self.MALFORMED_CHAR_BYTES) for id in ids]).decode("utf-8", errors="replace")
