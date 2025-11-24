from pathlib import Path
import importlib.util

import numpy as np
from bpe_tokenizer import Tokenizer


def load_trained_tokenizer():
    """Load the BPE tokenizer trained for this project."""
    tokenizer = Tokenizer(vocab = {}, merges = [])
    vocab_path =  "/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/vocab.pkl"
    merges_path =  "/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/merges.pkl"
    special_tokens = ["<|endoftext|>"]
    return Tokenizer.from_files(Tokenizer,vocab_filepath = vocab_path, merges_filepath = merges_path, special_tokens=special_tokens)


def get_longest_token(tokenizer: Tokenizer):
    token_id, token_bytes = max(tokenizer.vocab.items(), key=lambda kv: len(kv[1]))
    token_str = token_bytes.decode("utf-8", errors="replace")
    return token_id, token_bytes, token_str


def count_256_in_npy(npy_path: str | Path) -> int:
    """
    Count how many entries equal 256 inside a .npy file.

    Uses mmap loading to keep memory usage low for large arrays.
    """
    array = np.load(Path(npy_path), mmap_mode="r")
    return int(np.count_nonzero(array == 256))


if __name__ == "__main__":
    tokenizer = load_trained_tokenizer()
    print(tokenizer.encode('123<|endoftext|>'))
    print(count_256_in_npy('/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/tokenized_data/train_ids_special.npy'))
