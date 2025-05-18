import warnings
from collections.abc import Iterable, Iterator
from typing import Literal

import pandas as pd
import regex as re

from cs336_basics.pretokenization import get_pre_tokens
from cs336_basics.tokenizer_training import (
    load_tokenizer_from_txt,
)
from cs336_basics.utils_from_tests_folder import FIXTURES_PATH, get_vocab_merges_path


def apply_merge_to_list_of_tokens(
    tokens: list[bytes], merge: tuple[bytes, bytes]
) -> list[bytes]:
    new_tokens = []
    first, second = merge  # Cache merge values

    # --------- initial implementation
    is_previous_merge = False
    for token in tokens:
        if (token == first) and not is_previous_merge:
            # we keep that in mind and if the next token is the second part of the merge,
            # we will merge them
            new_tokens.append(token)
            is_previous_merge = True
        elif token == second:
            if is_previous_merge:
                # if the previous token was the first part of the merge,
                # we merge them
                new_tokens[-1] = first + second
            else:
                # in this case, we just add the token to the new list (there is no merge)
                new_tokens.append(token)
            is_previous_merge = False
        else:
            # no merge, we just add the token to the new list
            new_tokens.append(token)
            is_previous_merge = False

    # --------- Attempt for a faster implementation, that wasn't faster
    # i = 0
    # while i < len(tokens):
    #     # Check if current and next token match the merge pattern
    #     if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
    #         # We found a merge, add the merged token
    #         new_tokens.append(first + second)
    #         i += 2  # Skip both tokens we just merged
    #     else:
    #         # No merge, just add the current token
    #         new_tokens.append(tokens[i])
    #         i += 1
    return new_tokens


def split_on_special_tokens(
    text: str, special_tokens: list[str]
) -> tuple[list[str], list[str]]:
    """Split a text into substrings and special tokens.

    Args:
        text (str): The input text to split.
        special_tokens (list[str]): A list of special tokens to look for.

    Returns:
        tuple[list[str], list[str]]: Two lists of the same size: a list of substrings \
            and a list of special tokens found. Each special token found comes AFTER \
            the corresponding substring.
    """
    sub_strings = []
    special_tokens_found = []

    pattern = r"|".join([re.escape(tk) for tk in special_tokens])
    previous_end = 0

    for special_token_found in re.finditer(pattern, text):
        start = special_token_found.start()
        sub_strings.append(text[previous_end:start])
        special_tokens_found.append(special_token_found.group())
        previous_end = special_token_found.end()

    if previous_end < len(text):
        sub_strings.append(text[previous_end:])
        special_tokens_found.append("")

    return sub_strings, special_tokens_found


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary,
        list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = None
        else:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)

        self.reverse_vocab = {value: key for key, value in self.vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab, merges = load_tokenizer_from_txt(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode_old(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        warnings.warn(
            "This method is deprecated because not optimized for performance."
            " Please use the `encode` method instead.",
            DeprecationWarning,
        )

        if self.special_tokens:
            sub_strings, special_tokens_found = split_on_special_tokens(
                text, self.special_tokens
            )
        else:
            sub_strings = [text]
            special_tokens_found = [""]

        pre_tokens = pd.Series()
        for sub_string in sub_strings:
            # we add the special token to the list of pre-tokens
            pre_tokens.loc[len(pre_tokens)] = get_pre_tokens(sub_string, None)
        # pre tokens looks like pd.Series([["the", " cat", " ate"], [" an", " apple"]])

        pre_tokens = pre_tokens.apply(
            lambda pre_tokens_in_sub_string: [
                [bytes([byte]) for byte in pre_token.encode("utf-8")]
                for pre_token in pre_tokens_in_sub_string
            ]
        )
        # pre_tokens looks like pd.Series([
        #   [[b"t", b"h", b"e"], [b" ", b"c", b"a", b"t"], [b" ", b"a", b"t", b"e"]],
        #   [[b" ", b"a", b"n"], [b" ", b"a", b"p", b"p", b"l", b"e"]]
        # ])
        # each pre_token is a list of bytes

        for merge in self.merges:
            pre_tokens = pre_tokens.map(
                lambda pre_tokens_in_sub_string: [
                    apply_merge_to_list_of_tokens(bytes_in_pre_token, merge)
                    for bytes_in_pre_token in pre_tokens_in_sub_string
                ]
            )
        # pre_tokens looks like pd.Series([
        #   [[b"the"], [b" c", b"a", b"t"], [b" at", b"e"]],
        #   [[b" a", b"n"], [b" a", b"pp", b"l", b"e"]]
        # ])
        # the pre_tokens bytes are merges and become a list of tokens (in their byte form)

        pre_tokens = pre_tokens.apply(
            lambda pre_tokens_in_sub_string: [
                [self.reverse_vocab[byte] for byte in token_in_pre_token]
                for token_in_pre_token in pre_tokens_in_sub_string
            ]
        )
        # pre_tokens looks like pd.Series([
        #   [[9], [7, 1, 5], [10, 3]],
        #   [[8, 12], [8, 11, 13, 3]]
        # ])

        for i, special_token in enumerate(special_tokens_found):
            if special_token != "":
                pre_tokens.iloc[i] = pre_tokens.iloc[i] + [
                    self.reverse_vocab[special_token.encode("utf-8")]
                ]

        return pre_tokens.explode().explode().dropna().tolist()

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        if self.special_tokens:
            sub_strings, special_tokens_found = split_on_special_tokens(
                text, self.special_tokens
            )
        else:
            sub_strings = [text]
            special_tokens_found = [""]

        # Create initial byte tokens (using a list instead of pandas Series for better performance)
        all_byte_tokens = []
        for sub_string in sub_strings:
            # sub_string looks like "the cat ate"
            pre_tokens_in_substring = get_pre_tokens(sub_string, None)
            # pre_tokens_in_substring looks like ["the", " cat", " ate"]

            byte_tokens_in_substring = [
                [bytes([byte]) for byte in pre_token.encode("utf-8")]
                for pre_token in pre_tokens_in_substring
            ]
            # byte_tokens_in_substring looks like [[b"t", b"h", b"e"], [b" ", b"c", b"a", b"t"], [b" ", b"a", b"t", b"e"]]
            all_byte_tokens.append(byte_tokens_in_substring)

        # Apply merges more efficiently
        # Process each substring's tokens
        for i, byte_tokens_in_substring in enumerate(all_byte_tokens):
            # Process each token sequence
            for j, token_sequence in enumerate(byte_tokens_in_substring):
                # Apply all merges to this sequence
                for merge in self.merges:
                    token_sequence = apply_merge_to_list_of_tokens(
                        token_sequence, merge
                    )
                all_byte_tokens[i][j] = token_sequence
            # byte_tokens_in_substring looks like [[b"the"], [b" c", b"a", b"t"], [b" at", b"e"]]

        # Convert to token IDs
        all_token_ids = []
        for i, byte_tokens_in_substring in enumerate(all_byte_tokens):
            token_ids = []
            for token_sequence in byte_tokens_in_substring:
                for byte in token_sequence:
                    token_ids.append(self.reverse_vocab[byte])

            # Add special token if present
            if special_tokens_found[i] != "":
                token_ids.append(
                    self.reverse_vocab[special_tokens_found[i].encode("utf-8")]
                )

            all_token_ids.extend(token_ids)

        return all_token_ids

    def encode_iterable(self, iterable: Iterable[str | bytes]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs. This is required
        for memory-efficient tokenization of large files that we cannot
        directly load into memory."""
        buffer = ""

        for chunk in iterable:
            # Convert bytes to string if needed
            if isinstance(chunk, bytes):
                text = chunk.decode("utf-8")
            else:
                text = chunk

            # Add the new text to our buffer
            buffer += text

            # Find a good breaking point - for example, the last space
            # You might need to adjust this logic based on your tokenization needs
            if " " in buffer:
                last_space_index = buffer.rindex(" ")
                # Process text up to the last space
                to_process = buffer[:last_space_index]  # Include the space
                buffer = buffer[last_space_index:]  # Keep the rest for next time

                # Tokenize and yield
                token_ids = self.encode(to_process)
                yield from token_ids

        # Process any remaining text in the buffer
        if buffer:
            token_ids = self.encode(buffer)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        try:
            list_bytes = [self.vocab[i] for i in ids]
        except KeyError as e:
            raise ValueError(
                f"Token ID {e} not found in vocabulary. "
                "Please check the vocabulary and the token IDs."
            )

        # Convert the list of bytes to a single string
        decoded_str = b"".join(list_bytes).decode("utf-8", errors="replace")
        return decoded_str


if __name__ == "__main__":
    test_mode: Literal["simple", "complex"] = "complex"

    if test_mode == "simple":
        test_str = "the cat ate<|endoftext|> an apple"

        test_vocab = {
            0: b" ",
            1: b"a",
            2: b"c",
            3: b"e",
            4: b"h",
            5: b"t",
            6: b"th",
            7: b" c",
            8: b" a",
            9: b"the",
            10: b" at",
            11: b"pp",
            12: b"n",
            13: b"l",
            14: b"<|endoftext|>",
        }

        test_merges = [
            (b"t", b"h"),
            (b" ", b"c"),
            (b" ", b"a"),
            (b"th", b"e"),
            (b" a", b"t"),
            (b"p", b"p"),
        ]

        tk = Tokenizer(test_vocab, test_merges, special_tokens=["<|endoftext|>"])
        print(tk.encode(test_str))

    else:
        VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
        MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
        special_tokens = None

        vocab, merges = get_vocab_merges_path(
            vocab_path=VOCAB_PATH,
            merges_path=MERGES_PATH,
            special_tokens=special_tokens,
        )
        tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
        with open(FIXTURES_PATH / "tinystories_sample.txt", mode="rb") as f:
            corpus_contents = f.read().decode("utf-8")

        ids = tokenizer.encode(corpus_contents)
        assert tokenizer.decode(ids) == corpus_contents
