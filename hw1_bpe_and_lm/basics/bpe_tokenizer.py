import numpy as np
import typing
from typing import BinaryIO, Iterable, Iterator
import os
import regex as re
from collections import Counter, defaultdict
from multiprocessing import Pool
from functools import partial
class bpe_tokenizer:
    def __init__(self):
        self.vocab = {} #dict(int:bytes)
    
    def vocab_initialization(self):
        self.vocab = {i:bytes([i]) for i in range (256)}
    
    def find_chunk_boundaries(
    self,
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
   
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
    def read_chunks_generator(self, input_path: str, boundaries: list[int]) :
       
        with open(input_path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if start >= end:
                    continue
                f.seek(start)
                chunk_bytes = f.read(end - start)
                yield chunk_bytes.decode("utf-8", errors="ignore")


    def pre_tokenization(self,corpus:str, special_tokens:list[str]):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretoken_counter = Counter()
        if special_tokens:
            pattern = "|".join(re.escape(tok) for tok in special_tokens)
            chunks = re.split(pattern, corpus)
        else:
            chunks = [corpus]
        for chunk in chunks:
            if not chunk.strip():  
                continue
            for match in re.finditer(PAT, chunk):
                token = match.group(0)
                if not token:
                    continue
                token_bytes = token.encode("utf-8")
                token_tuple = tuple(bytes([b]) for b in token_bytes)
                pretoken_counter[token_tuple] += 1
        return dict(pretoken_counter)
    
    def merge_token(self, token:tuple[bytes, ...], best_pair:tuple[bytes, bytes]):
        merged = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and token[i] == best_pair[0] and token[i + 1] == best_pair[1]:
                merged.append(token[i] + token[i + 1])  
                i += 2  
            else:
                merged.append(token[i])
                i += 1

        return tuple(merged)
    
    def train_bpe_tokenizer(self, input_path:str, vocab_size:int, special_tokens:list[str]):
        #initialization part: add 256 bytes and special tokens into the vocab
        self.vocab_initialization()
        curr_vocab_size = len(self.vocab)
        for special_token in special_tokens:
            self.vocab[curr_vocab_size] = bytes(special_token.encode('utf-8'))
            curr_vocab_size += 1
        # end of the initialization part
   
        #begin of the pre_tokenization part

        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = self.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            chunks = []
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunk_generator = self.read_chunks_generator(input_path, boundaries)
        pretoken_counter = Counter()
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        with Pool(processes=num_processes) as pool:
            task = partial(self.pre_tokenization, special_tokens=special_tokens)
            results_iterator = pool.imap_unordered(task, chunk_generator)
            for res in results_iterator:
                pretoken_counter.update(res)
        

        #end of the pre_tokenization part

        #begin of the merging part
        pair_counter = Counter()
        pair_to_tokens_map = defaultdict(set)
        #cal all the initial pair freqs by iterating through all the byte pairs
        for byte_seq, freq in pretoken_counter.items():
            for i in range(len(byte_seq)-1):
                pair = (byte_seq[i], byte_seq[i+1])
                pair_counter[pair] += freq
                pair_to_tokens_map[pair].add(byte_seq)
        #begin merging until the vocab size reaches the max size
        merges = []
        while curr_vocab_size < vocab_size:
            if not pair_counter:
                break
            #find the best pair
            best_pair = max(pair_counter, key=lambda p: (pair_counter[p], p))
            merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[curr_vocab_size] = new_token
            #fine all the tokens affected by the best pair. Note that all the merge happen inside token, so that we can do like this
            affected_tokens = list(pair_to_tokens_map.get(best_pair, set()))

            for token in affected_tokens:
                if token not in pretoken_counter:
                    continue
                freq = pretoken_counter.pop(token)
                new_token = self.merge_token(token, best_pair)
                pretoken_counter[new_token] += freq

                old_pairs = list(zip(token, token[1:]))
                new_pairs = list(zip(new_token, new_token[1:]))
                #for all the pairs that emerge in the original token, subtract the freq by the original token's freq
                for p in old_pairs:
                    pair_counter[p] -= freq
                    if pair_counter[p] <= 0:
                        pair_counter.pop(p, None)
                    if p in pair_to_tokens_map:
                        pair_to_tokens_map[p].discard(token)
                        if not pair_to_tokens_map[p]: 
                            del pair_to_tokens_map[p]
    
                #for all the pairs that emerge in the new token, add the freq by the new token's freq
                
                for p in new_pairs:
                    pair_counter[p] += freq
                    pair_to_tokens_map[p].add(new_token)
            curr_vocab_size += 1
            if curr_vocab_size % 100 == 0:
                print(f'curr_vocab_size:{curr_vocab_size}')
        return self.vocab, merges

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None): 

        #Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:

        self.vocab: dict[int, bytes] = vocab
        self.token_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens.sort(key=len, reverse=True)
        if special_tokens is not None:
            for tok in special_tokens:
                tok_bytes = tok.encode("utf-8")
                if tok_bytes not in self.token_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = tok_bytes
                    self.token_to_id[tok_bytes] = new_id
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        if self.special_tokens:
            special_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
            self.special_regex = re.compile(f"({special_pattern})")
        else:
            self.special_regex = None
            
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        #Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens. This method should accept the following additional parameters:

        with open(vocab_filepath, "rb") as f:
            import pickle
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        final_ids = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if self.special_regex is None:
            for match in re.finditer(PAT, text):
                word_bytes = tuple(match.group(0).encode("utf-8"))
                final_ids.extend(self.bpe(word_bytes))
            return final_ids

        last_end = 0
        for match in self.special_regex.finditer(text):
            start, end = match.span()
            if start > last_end:
                pre_special_chunk = text[last_end:start]
                for inner_match in re.finditer(PAT, pre_special_chunk):
                    word_bytes = tuple(inner_match.group(0).encode("utf-8"))
                    final_ids.extend(self.bpe(word_bytes))
            
            special_token_bytes = match.group(0).encode("utf-8")
            final_ids.append(self.token_to_id[special_token_bytes])
            last_end = end
        
        if last_end < len(text):
            post_special_chunk = text[last_end:]
            for inner_match in re.finditer(PAT, post_special_chunk):
                word_bytes = tuple(inner_match.group(0).encode("utf-8"))
                final_ids.extend(self.bpe(word_bytes))
                
        return final_ids
    
    def encode_text_without_special(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = [m.group(0) for m in re.finditer(PAT, text)]
        ids = []
        for word in pre_tokens:
            word_bytes = tuple(word.encode("utf-8"))
            ids.extend(self.bpe(word_bytes))
        return ids
    
    def bpe(self, word_bytes: tuple[int]) -> list[int]:
        tokens = [bytes([b]) for b in word_bytes]
        pairs = self.get_pairs(tokens)
        while pairs:
            merge = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))
            if merge not in self.merge_ranks:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == merge:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            pairs = self.get_pairs(tokens)
        return [self.token_to_id[t] for t in tokens]
    
    def get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        #Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        for text in iterable:
            if not text:
                continue  
            token_ids = self.encode(text)
            yield from token_ids
            
    def decode(self, ids: list[int]) -> str:
        #Decode a sequence of token IDs into text.
        byte_pieces = []
        for i in ids:
            if i in self.vocab:
                byte_pieces.append(self.vocab[i])
            else:
                byte_pieces.append(b'?')

        byte_stream = b"".join(byte_pieces)

        text = byte_stream.decode("utf-8", errors="replace")

        return text