import os
import sys
from functools import lru_cache
from multiprocessing import Pool
from typing import BinaryIO

import pandas as pd
import regex as re
from tqdm import tqdm


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

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


def get_pre_tokens(text: str, special_tokens: list[str] | None) -> list[str]:
    """
    This function takes a text (with special tokens)
    and returns a list of pre-tokens found in the text.
    """
    # Remove the special tokens before applying the pre-tokenization
    if special_tokens is None:
        docs_in_text = [text]
    else:
        docs_in_text = re.split(
            r"|".join([re.escape(tk) for tk in special_tokens]), text
        )

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = []
    for doc in docs_in_text:
        # Find all matches of the pattern in the document
        matches = re.findall(PAT, doc)
        pre_tokens = pre_tokens + matches

    return pre_tokens


def count_pre_tokens(pre_tokens: list[str]) -> dict[tuple[bytes], int]:
    """
    Count the occurrences of each pr
    
    e-token in the list.

    Returns:
        A dictionary with pre-tokens as keys (as tuple of bytes) \
        and pre-token counts as values.
    """
    # unique_pre_tokens = set(pre_tokens)
    # pre_token_counts = {}
    # for pre_token in unique_pre_tokens:
    #     pre_token_counts[pre_token.encode("utf-8")] = pre_tokens.count(pre_token)

    # faster implementation using pandas
    pre_token_counts = pd.Series(pre_tokens).value_counts()
    pre_token_counts.index = pre_token_counts.index.map(lambda x: x.encode("utf-8"))
    pre_token_counts = pre_token_counts.to_dict()
    return pre_token_counts


def _pre_tokenize_chunk_parallel_process(
    args: tuple[str, list[str]],
) -> dict[tuple[bytes], int]:
    """
    Pre-tokenize a chunk of the file in parallel.
    """
    chunk = args[0]
    special_tokens = args[1]

    pre_tokens = get_pre_tokens(chunk, special_tokens)

    pre_token_counts = count_pre_tokens(pre_tokens)

    return pre_token_counts


@lru_cache(maxsize=16)
def count_chunk_split_token_occurrences(
    file_path: str | os.PathLike, token: str
) -> int:
    token_bytes = token.encode("utf-8")
    with open(file_path, "rb") as f:
        content = f.read()
    return content.count(token_bytes)


def merge_pre_token_counts(
    all_pre_token_counts: list[dict[tuple[bytes], int]],
) -> pd.Series:
    s_all_pre_token_counts = (
        pd.concat([pd.Series(d) for d in all_pre_token_counts], axis=1)
        .sum(axis=1)
        .astype(int)
    )
    s_all_pre_token_counts.index = s_all_pre_token_counts.index.map(list)

    return s_all_pre_token_counts


def pre_tokenize_file(
    file_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
    parallel: bool = True,
) -> pd.Series:  # dict[tuple[bytes], int]:
    chunk_split_token = "<|endoftext|>"
    assert chunk_split_token in special_tokens, (
        f"Chunk split token {chunk_split_token} must be in special tokens list. "
        "Otherwise, please change here the chunk_split_token to the one relevant to your corpus."
    )

    ## Usage
    with open(file_path, "rb") as f:
        if num_processes is None:
            num_occurences = count_chunk_split_token_occurrences(
                file_path, chunk_split_token
            )
            if num_occurences == 0:
                num_occurences = 1000
            num_processes = min(int(num_occurences * 0.01), 8_000)
            # 0.05 is empirically the sweet spot for parallelization

        boundaries = find_chunk_boundaries(
            f, num_processes, chunk_split_token.encode("utf-8")
        )

        if not parallel:
            all_pre_token_counts = []

            # Serial implementation
            for start, end in tqdm(
                zip(boundaries[:-1], boundaries[1:]),
                total=len(boundaries) - 1,
                desc="Generating pre-tokens: ",
            ):
                f.seek(start)
                # Add the replace here because Windows reads \r\n when Unix uses \n
                # This is a workaround for the fact that Windows uses \r\n as line endings
                chunk = (
                    f.read(end - start)
                    .replace(b"\r\n", b"\n")
                    .decode("utf-8", errors="ignore")
                )

                pre_token_counts = _pre_tokenize_chunk_parallel_process(
                    (chunk, special_tokens)
                )

                if start == 0:
                    size_one_iteration = sys.getsizeof(pre_token_counts)
                    print(
                        f"INFO: The pre_tokenization will take up at least "
                        f"{size_one_iteration*(len(boundaries) - 1)/1_000_000_000:.2f} Gb"
                    )

                all_pre_token_counts.append(pre_token_counts)
        else:
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(
                    f.read(end - start)
                    .replace(b"\r\n", b"\n")
                    .decode("utf-8", errors="ignore")
                )

            all_pre_token_counts = []
            with Pool() as pool:
                for result in tqdm(
                    pool.imap(
                        _pre_tokenize_chunk_parallel_process,
                        [(chunk, special_tokens) for chunk in chunks],
                    ),
                    desc="Generating pre-tokens: ",
                    total=len(chunks),
                ):
                    all_pre_token_counts.append(result)

    print("Counting pre-tokens...", end="")
    pre_token_counts = merge_pre_token_counts(all_pre_token_counts)
    print("Done.")
    return pre_token_counts
