from basics.bpe_tokenizer import Tokenizer
from basics.transformer import TransformerLM
from basics.training import load_checkpoint, AdamW
import torch
if __name__ == "__main__":
    print(": Generating text...")

    prompt_text = "Once upon a time"
    tokenizer = Tokenizer.from_files(Tokenizer, vocab_filepath = '/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/vocab.pkl', merges_filepath = '/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/merges.pkl', special_tokens = ['<|endoftext|>'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_transformer = False
    if use_transformer:
        model = TransformerLM(
            vocab_size=10000,
            context_length=128,
            num_layers=4,
            d_model=256,
            num_heads=8,
            d_ff=1024,
            device=device
        ).to(device)
    else:
        from basics.transformer import LSTMLM
        model = LSTMLM(
            vocab_size=10000,
            context_length=128,
            num_layers=3,
            d_model=512,
            device=device
        ).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    load_checkpoint(
        model = model,
        optimizer = optimizer,
        inp = '/share/project/zhaomingxuan/nlp/NLPDL-2025Fall/hw1_bpe_and_lm/basics/checkpoints_lstm_h100_special/checkpoint_15000.pt'
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt_ids = tokenizer.encode(prompt_text)


    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"Prompt: '{prompt_text}'")
    print("-" * 30)

    eos_id = tokenizer.token_to_id[b"<|endoftext|>"]
    generated_ids_tensor = model.generate(
        input_ids=input_ids,
        max_new_tokens=500,     
        temperature=0.8,        
        top_p=0.9,              
        eos_token_id=eos_id 
    )


    generated_ids_list = generated_ids_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids_list)

    print("Generated Text:")
    print(generated_text)