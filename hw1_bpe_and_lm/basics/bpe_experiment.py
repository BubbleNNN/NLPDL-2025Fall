import os
import re
import pickle
import time
import glob
import numpy as np
from typing import Iterable, Iterator
from bpe_tokenizer import Tokenizer

VOCAB_FILE_PATH = "/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/vocab.pkl"  
MERGES_FILE_PATH = "/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/merges.pkl" 


TRAIN_DATA_PATH = "/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/data/TinyStoriesV2-GPT4-train.txt" 
DEV_DATA_PATH = "/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/data/TinyStoriesV2-GPT4-valid.txt"   


OUTPUT_DIR = "tokenized_data"


VOCAB_SIZE = 10000
NUM_SAMPLES_FOR_TEST = 10
PILE_DATASET_SIZE_GB = 825

if __name__ == "__main__":
    

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading pretrained TinyStories Tokenizer ...")
    tokenizer = Tokenizer.from_files(
        Tokenizer,
        vocab_filepath=VOCAB_FILE_PATH,
        merges_filepath=MERGES_FILE_PATH,
        special_tokens=['<|endoftext|>']
    )
    print(f"Tokenizer successfully loaded。Vocab size: {len(tokenizer.vocab)}")
    

    print("\n" + "="*80)
    print("Experiment (a) & (b)")
    print("="*80)

    total_bytes = 0
    total_tokens = 0
    sampled_docs = []

    
    
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
        docs = content.split('<|endoftext|>')
        sampled_docs = [doc.strip() for doc in docs if doc.strip()][:NUM_SAMPLES_FOR_TEST]


    start_time = time.perf_counter()
    for doc in sampled_docs:
        doc_bytes = len(doc.encode("utf-8"))
        total_bytes += doc_bytes

        encoded_ids = tokenizer.encode(doc)
        total_tokens += len(encoded_ids)
    end_time = time.perf_counter()

    duration = end_time - start_time
    

    compression_ratio = total_bytes / total_tokens
    print(f"\n--- Experiment (a)  ---")
    print(f"Original Bytes: {total_bytes:,}")
    print(f"Token Num: {total_tokens:,}")
    print(f"Compression Ratio (bytes/token): {compression_ratio:.2f}")


    throughput_bps = total_bytes / duration
    throughput_mbps = throughput_bps / (1024 * 1024)
    print(f"\n--- Experiment (b)  ---")
    print(f"It takes {duration:.4f} seconds to process {len(sampled_docs)} documents")
    print(f"Tokenizer throughput: {throughput_bps:,.2f} bytes/s")


    pile_size_bytes = PILE_DATASET_SIZE_GB * (1024**3)
    time_for_pile_seconds = pile_size_bytes / throughput_bps
    time_for_pile_days = time_for_pile_seconds / (60 * 60 * 24)
    time_for_pile_years = time_for_pile_days / 365.25
    
    print(f"It would take  {time_for_pile_days:.2f} days to process the Pile dataset")




    print("\n" + "="*80)
    print("Experiment (c)")
    print("="*80)

    def process_and_save_split(split_name, path):
        print(f"\nstart processing '{split_name}' dataset...")
        all_ids = []
        
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            print('start encoding')
            all_ids = tokenizer.encode(text)
            print('end encoding')

        print("\nFinished。")
        

        output_path = os.path.join(OUTPUT_DIR, f"{split_name}_ids_special.npy")
        token_array = np.array(all_ids, dtype=np.uint16)
        
        print(f"Saving to '{output_path}'...")
        np.save(output_path, token_array)
        
        print(f"'{split_name}' Finished。")
        print(f"Tokens : {token_array.shape[0]:,}")
        print(f"Memory usage: {token_array.nbytes / (1024*1024):.2f} MB")

    try:
        process_and_save_split("train", TRAIN_DATA_PATH)
        process_and_save_split("dev", DEV_DATA_PATH)
    except Exception as e:
        print(f"\nError: {e}")

    print("\n" + "="*80)
    print("Done。")
    print("="*80)