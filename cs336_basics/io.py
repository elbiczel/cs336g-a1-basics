import numpy as np
import os
import random
import numpy.typing as npt

from typing import BinaryIO, Sequence, Optional, Iterator, Iterable

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
        offset = _find_next_doc_start(file, split_special_token)
        if offset is None:
            chunk_boundaries[bi] = file_size
        else:
            chunk_boundaries[bi] = initial_position + offset

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _find_next_doc_start(f: BinaryIO, sep: bytes) -> Optional[int]:
    pos = 0
    # find beginning of the doc.
    while True:
        mini_chunk = f.read(_mini_chunk_size)
        if mini_chunk == b"":
            # EOF
            return None
        found_at = mini_chunk.find(sep)
        if found_at != -1:
            pos += found_at + len(sep)
            return pos
        pos += _mini_chunk_size

def next_doc(f: BinaryIO, sep: bytes) -> bytes:
    buf = getattr(f, "_nd_buf", None)
    if buf is None:
        buf = bytearray()
        setattr(f, "_nd_buf", buf)
    sep_len = len(sep)
    while True:
        i = buf.find(sep)
        if i != -1:
            end = i + sep_len
            out = bytes(buf[:end])
            del buf[:end]
            return out

        chunk = f.read(_mini_chunk_size)
        if not chunk:
            if buf:
                out = bytes(buf)
                buf.clear()
                return out
            return b""
        else:
            buf += chunk


def docs_in_range(
    f: BinaryIO,
    sep: bytes,
    start: int,
    end: int,
) -> Iterator[bytes]:
    # Reset per-file buffer state
    setattr(f, "_nd_buf", bytearray())

    # Seek to start offset
    f.seek(start)

    read_limit = end - start
    total_read = 0

    while True:
        doc = next_doc(f, sep)
        if not doc:
            break

        total_read += len(doc)
        if total_read == read_limit:
            yield doc
            break
        if total_read > read_limit:
            # Clip the last doc to fit inside end
            excess = total_read - read_limit
            assert excess == 0
            yield doc[:-excess]
            break
        else:
            yield doc

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
            pos = _find_next_doc_start(f, sep)
            if pos is None: continue
            pos += n
            if pos in docs: continue
            f.seek(pos)
            doc = next_doc(f, sep)
            out.append(doc.decode("utf-8", errors="ignore"))
            docs.add(pos)
    return out

class TokenWriter:
    def __init__(self, path: str | os.PathLike, chunk_size: int = _mini_chunk_size):
        self.path = path
        self.chunk_size = chunk_size
        self.buf = []
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "ab")
        return self

    def write(self, tokens: int | Iterator[int]):
        assert self.f != None
        if isinstance(tokens, int):
            self.buf.append(tokens)
        else:
            self.buf.extend(tokens)
        if len(self.buf) >= self.chunk_size:
            np.asarray(self.buf, dtype="<u2").tofile(self.f)
            self.buf.clear()

    def __exit__(self, exc_type, exc, tb):
        assert self.f != None
        if self.buf:
            np.asarray(self.buf, dtype="<u2").tofile(self.f)
            self.buf.clear()
        self.f.close()

def read_tokens(path: str | os.PathLike) -> npt.NDArray[np.uint16]:
    return np.memmap(path, dtype="<u2", mode='r')
