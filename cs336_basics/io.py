import numpy as np
import os
import random

from typing import BinaryIO, Sequence, Tuple

_mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

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

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        offset, _ = _find_next_doc_start(file, split_special_token)
        if offset is None:
            chunk_boundaries[bi] = file_size
        else:
            chunk_boundaries[bi] = initial_position + offset

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _find_next_doc_start(f: BinaryIO, sep: bytes) -> Tuple[int | None, bytes]:
    # Returns: (offset to start current document, rest of mini chunk).
    init_pos = 0
    # find beginning of the doc.
    while True:
        mini_chunk = f.read(_mini_chunk_size)
        if mini_chunk == b"":
            # EOF
            return (None, mini_chunk)
        found_at = mini_chunk.find(sep)
        if found_at != -1:
            init_pos += found_at
            content = mini_chunk[found_at + len(sep):]
            return (init_pos, content)
        init_pos += _mini_chunk_size

def next_doc(f: BinaryIO, sep: bytes, content: bytes) -> Tuple[bytes, bytes]:
    # Returns (doc content, rest of the mini chunk)
    # Assumes the f is at a document start/beginning of the file or content has the mini chunk at the start of this doc.
    found_at = content.find(sep)
    if found_at != -1:
        return (content[:found_at], content[found_at + len(sep):])
    # find end of the doc
    while True:
        mini_chunk = f.read(_mini_chunk_size)
        if mini_chunk == b"":
            return (content, mini_chunk)
        found_at = mini_chunk.find(sep)
        if found_at != -1:
            content += mini_chunk[:found_at]
            return (content, mini_chunk[found_at + len(sep):])
        content += mini_chunk

def sample_docs(path: str | os.PathLike, num: int, sep: bytes) -> Sequence[str]:
    out = []
    with open(path, "rb") as f:
        # Get total file size in bytes
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        docs = set()
        while len(out) < num:
            n = random.randint(0, file_size - 1)
            f.seek(n)  # Start at boundary guess
            pos, content = _find_next_doc_start(f, sep)
            if pos is None: continue
            pos += n
            if pos in docs: continue
            doc, _ = next_doc(f, sep, content)
            docs.add(pos)
            out.append(doc.decode("utf-8", errors="ignore"))
    return out

def write_tokens(path: str | os.PathLike, tokens: list[int]):
    arr = np.array(tokens, dtype=np.uint16)
    arr = arr.astype('<u2', copy=False)   # standardize to little-endian
    np.save(path, arr)

def read_tokens(path: str | os.PathLike) -> list[int]:
    return np.load(path, mmap_mode='r')
