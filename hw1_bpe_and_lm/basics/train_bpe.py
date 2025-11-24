from basics.bpe_tokenizer import bpe_tokenizer
import threading
import time
import os
import psutil
peak_memory_usage = 0.0
monitoring_active = True
def monitor_memory():
    global peak_memory_usage
    
    parent_process = psutil.Process(os.getpid())
    
    while monitoring_active:
        try:

            parent_mem = parent_process.memory_info().rss
            
            children_mem = 0
            children = parent_process.children(recursive=True)
            for child in children:
                try:
                    children_mem += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            
            total_mem_bytes = parent_mem + children_mem
            total_mem_mb = total_mem_bytes / (1024 * 1024)
            

            if total_mem_mb > peak_memory_usage:
                peak_memory_usage = total_mem_mb
                
        except psutil.NoSuchProcess:
            break
            

        time.sleep(0.1)
if __name__ == '__main__':
    start_time = time.time()
    print("Start program")
    
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    print("Memory monitor start")
    import pickle
    tokenizer = bpe_tokenizer()
    print('start training')
    vocab, merges = tokenizer.train_bpe_tokenizer(
    "data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"]
    )
    print('end training')
    serial_time = time.time()
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    serial_end_time = time.time()

    print(f"Training finished. Size of vocab: {len(vocab)}, Number of merges: {len(merges)}")
    print(f"longest token: {max(len(v) for v in vocab.values())}")
    monitoring_active = False
    time.sleep(0.2) 
    print("Memory monitor stopped。")
    end_time = time.time()
    print(f"==============================================")
    print(f"Peak Memory Usage: {peak_memory_usage:.2f} MB")
    print(f'Running time:{start_time-end_time}, Serial cost:{serial_end_time-serial_time}')
    print(f"==============================================")
