import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cs336_basics.pretokenization import pre_tokenize_file
from cs336_basics.utils_from_tests_folder import DATA_FOLDER


def get_list_pairs_in_list_tokens(list_tokens: list[int]) -> list[tuple[int, int]]:
    """
    Get all pairs of consecutive indices in a list of tokens.

    For instance, if the list of tokens is [0, 1, 2], the pairs are [(0, 1), (1, 2)].
    """
    return [(list_tokens[i], list_tokens[i + 1]) for i in range(len(list_tokens) - 1)]


def expand_pairs(df: pd.DataFrame) -> pd.Series:
    """Transform a Series with a list of tuples as index to a Series with the tuples as index.
    
    For instance, s looks like: \\
    Index                    | Value \\
    [(0, 0), (0, 1), (0, 2)] | "first" \\
    [(1,0), (1,1)]           | "second" \\

    And is going to be transformed into: \\
    Index | Value \\
    (0,0) | "first" \\
    (0,1) | "first" \\
    (0,2) | "first" \\
    (1,0) | "second" \\
    (1,1) | "second" \\

    Args:
        s (pd.Series): Series to transform.

    Returns:
        pd.Series: Transformed Series.
    """
    new_index = [tuple_ for sublist in df["list_pairs"] for tuple_ in sublist]
    new_values = [
        count for count, sublist in zip(df["count"], df["list_pairs"]) for _ in sublist
    ]
    return pd.Series(new_values, index=new_index)


def convert_pair_tokens_to_bytes(
    pair: tuple[int, int], vocab: dict[int, bytes]
) -> tuple[bytes, bytes]:
    return (vocab[pair[0]], vocab[pair[1]])


def apply_merge_to_list_of_pairs(
    list_of_pairs: list[tuple[int, int]], merge_pair: tuple[int, int], merge_token: int
) -> list[tuple[int, int]]:
    """
    Apply a merge to a list of pairs of tokens.

    Example:
        >>> apply_merge_to_list_of_pairs(
        ...     [(32, 97), (97, 108), (108, 108), (108, 32), (32, 97), (97, 32), (32, 97)],
        ...     (32, 97),
        ...     999,
        ... )
        [(999, 108), (108, 108), (108, 999), (999, 999)]
    """
    new_list = []
    is_previous_merge = False
    for pair in list_of_pairs:
        if pair == merge_pair:
            is_previous_merge = True
            if new_list:
                new_list[-1] = (new_list[-1][0], merge_token)
        else:
            if is_previous_merge:
                new_list.append((merge_token, pair[1]))
                is_previous_merge = False
            else:
                new_list.append(pair)
    return new_list


def generate_one_merge(
    df_pre_token_counts: pd.DataFrame,
    df_token_pair_counts: pd.Series,
    vocab: dict[int, bytes],
) -> tuple[tuple[int, int], pd.Series]:
    """
    Generate one merge from the pre-token counts.
    This function applies the merge inplace to df_pre_token_counts, and also
    returns the merge.

    Args:
        pre_token_counts (pd.DataFrame): Pre-token counts. Looks like: \
            Index | list_pairs            | count \\
            int   | list[tuple[int, int]] | int   \\
        df_token_pair_counts (pd.Series): Token pair counts. Looks like: \
            Index           | value \\
            tuple[int, int] | int   \\
        vocab (dict[int, bytes]): Vocabulary.

    Returns:
        tuple[int, int]: The merge, i.e. the two tokens to be merged.
    """
    # Get the pairs that are the most frequent
    max_count = df_token_pair_counts.max()
    most_frequent_pairs = df_token_pair_counts[df_token_pair_counts == max_count]

    # Convert those pairs to their string representation, so that we can find which
    # one is lexicographically greatest
    idx_lex_greater = (
        most_frequent_pairs.reset_index()["index"]
        .map(lambda x: convert_pair_tokens_to_bytes(x, vocab))
        .idxmax()
    )

    # Finally, get the most_frequent pair for this merge
    most_frequent_pair: tuple[int, int] = most_frequent_pairs.index[idx_lex_greater]  # type: ignore

    # We need to apply this merge to the list of pre-tokens
    mask = df_pre_token_counts.loc[:, "list_pairs"].map(
        lambda x: most_frequent_pair in x
    )
    df_affected_token_pair_counts_before_merge = (
        expand_pairs(df_pre_token_counts.loc[mask]).groupby(level=0).sum()
    )

    merge_token_number = len(
        vocab
    )  # The new token number is the length of the vocabulary
    df_pre_token_counts.loc[mask, "list_pairs"] = df_pre_token_counts.loc[
        mask, "list_pairs"
    ].map(
        lambda x: apply_merge_to_list_of_pairs(
            x, most_frequent_pair, merge_token_number
        )
    )
    df_affected_token_pair_counts_after_merge = (
        expand_pairs(df_pre_token_counts.loc[mask]).groupby(level=0).sum()
    )

    df_pair_counts_diff = pd.concat(
        [
            df_affected_token_pair_counts_before_merge,
            df_affected_token_pair_counts_after_merge,
        ],
        axis=1,
    )
    df_pair_counts_diff.columns = ["before", "after"]
    df_pair_counts_diff.loc[:, "after"] = (
        df_pair_counts_diff["after"].infer_objects(copy=False).fillna(0)
    )

    # we use the "after" counts to update the pairs that are in before
    # and that changed after the merge
    mask_pairs_to_update = df_pair_counts_diff[
        (~df_pair_counts_diff["before"].isna())
        & (df_pair_counts_diff["before"] != df_pair_counts_diff["after"])
    ].index
    df_token_pair_counts.loc[mask_pairs_to_update] += (
        df_pair_counts_diff.loc[mask_pairs_to_update, "after"]
        - df_pair_counts_diff.loc[mask_pairs_to_update, "before"]
    ).astype(int)

    # we add the new pair to the token pair counts (the pairs that are only in after)
    pairs_to_add = df_pair_counts_diff.loc[
        df_pair_counts_diff["before"].isna(), "after"
    ].astype(int)
    if not pairs_to_add.empty:
        df_token_pair_counts = pd.concat([df_token_pair_counts, pairs_to_add])

    # Clean the DataFrames/Series
    # drop rows that only had the merged pair (it no longer has a pair now)
    # df_pre_token_counts.drop(
    #     df_pre_token_counts[df_pre_token_counts["list_pairs"].map(len) == 0].index,
    #     axis=0,
    #     inplace=True,
    # )

    return most_frequent_pair, df_token_pair_counts


def bpe_tokenizer_training(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    parallel: bool = True,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        parallel (bool): Whether to use parallel processing for pre-tokenization.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab = initialize_vocabulary()
    current_vocab_size = len(vocab) + len(special_tokens)
    assert (
        vocab_size > current_vocab_size
    ), f"Target vocab size {vocab_size} must be greater than initial vocab size {current_vocab_size}."

    pre_token_counts = pre_tokenize_file(
        input_path, special_tokens=special_tokens, parallel=parallel
    )

    # Clean the pre-token counts by removing the tokens that only have 1 byte
    # and that won't see a merge.
    pre_token_counts = pre_token_counts.loc[pre_token_counts.index.map(len) > 1]

    df_pre_token_counts = pre_token_counts.reset_index().rename(
        columns={"index": "list_pairs", 0: "count"}
    )

    # Separate the index in a list of pairs of tokens
    df_pre_token_counts["list_pairs"] = df_pre_token_counts["list_pairs"].map(
        get_list_pairs_in_list_tokens
    )

    # Separate each token pair into its own row
    df_token_pair_counts = expand_pairs(df_pre_token_counts)

    # A pair can appear multiple times in the pre-token counts,
    # so we need to group by the pair and sum the counts
    df_token_pair_counts = df_token_pair_counts.groupby(level=0).sum()

    merges = []
    for _ in tqdm(range(vocab_size - current_vocab_size), desc="Merging"):
        pair_to_merge, df_token_pair_counts = generate_one_merge(
            df_pre_token_counts, df_token_pair_counts, vocab
        )

        # convert the pair of tokens (int, int) to a pair of bytes (bytes, bytes)
        pair_to_merge_bytes = (
            vocab[pair_to_merge[0]],
            vocab[pair_to_merge[1]],
        )

        # Merge the pair of tokens in the vocabulary
        vocab[len(vocab)] = pair_to_merge_bytes[0] + pair_to_merge_bytes[1]

        merges.append(pair_to_merge_bytes)
        current_vocab_size += 1

    # Finally, we add he special tokens to the vocabulary.
    # These tokens are not merged, and are not part of the BPE merges.
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    return vocab, merges


def read_txt_file(file_path: str) -> str:
    """
    Read a text file and return its contents as a list of lines.
    """
    with open(file_path, encoding="utf-8") as file:
        return file.read()


def initialize_vocabulary() -> dict[int, bytes]:
    """
    Initialize the vocabulary with the UTF-8 bytes.
    """
    vocab = {i: bytes([i]) for i in range(256)}  # Initialize with byte values

    return vocab


def save_tokenizer_to_txt(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_txt: str | Path,
    merges_txt: str | Path,
) -> None:
    """
    Save vocabulary and merges to text files by converting bytes to hex strings.

    Args:
        vocab (dict[int, bytes]): Vocabulary mapping token IDs to bytes.
        merges (list[tuple[bytes, bytes]]): List of merge tuples.
        vocab_txt (str | Path): File path for vocabulary output.
        merges_txt (str | Path): File path for merges output.
    """
    # # Convert vocabulary values to hexadecimal strings for JSON serialization
    vocab_serializable = {k: v.hex() for k, v in vocab.items()}

    # # Convert merges (list of tuples of bytes) into tuples of hex strings
    merges_serializable = [(a.hex(), b.hex()) for a, b in merges]

    with open(vocab_txt, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, indent=2)

    with open(merges_txt, "w", encoding="utf-8") as f:
        json.dump(merges_serializable, f, indent=2)


def load_tokenizer_from_txt(
    vocab_txt: str | Path, merges_txt: str | Path
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Load and decode vocabulary and merges from text files that were saved in JSON format.

    Args:
        vocab_txt (str | Path): File path for vocabulary input.
        merges_txt (str | Path): File path for merges input.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            A tuple containing:
                - The vocabulary mapping token IDs (int) to token bytes.
                - The list of merge tuples, each a tuple of bytes objects.
    """
    import json

    # Load and decode vocabulary
    with open(vocab_txt, encoding="utf-8") as f:
        vocab_serializable = json.load(f)
    # Since JSON object keys become strings, convert them back to ints and decode the hex values.
    vocab = {int(k): bytes.fromhex(v) for k, v in vocab_serializable.items()}

    # Load and decode merges
    with open(merges_txt, encoding="utf-8") as f:
        merges_serializable = json.load(f)
    # Convert each tuple of hex strings to a tuple of bytes.
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_serializable]

    return vocab, merges


if __name__ == "__main__":

    print(DATA_FOLDER)

    vocab, merges = bpe_tokenizer_training(
        DATA_FOLDER / "TinyStoriesV2-GPT4-valid.txt",
        500,
        special_tokens=["<|endoftext|>"],
        parallel=False,
    )

    vocab_filepath = DATA_FOLDER / "temp_vocabulary.txt"
    merges_filepath = DATA_FOLDER / "temp_merges.txt"
    save_tokenizer_to_txt(vocab, merges, vocab_filepath, merges_filepath)
