import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional, Tuple, Union, BinaryIO
import numpy as np
import os
import torch
import math
from tqdm import tqdm
from basics.transformer import TransformerLM, LSTMLM
import wandb
import yaml
import argparse


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    x_max, _ = torch.max(logits, dim=-1, keepdim=True)
    x_exp = torch.exp(logits - x_max)
    log_sum_exp = torch.log(torch.sum(x_exp, dim=-1))
    target_logits = torch.gather(logits - x_max, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = -(target_logits - log_sum_exp)
    return loss.mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  
                t = state.get("t", 0)  
                grad = p.grad.data  
                p.data -= lr / math.sqrt(t + 1) * grad  
                state["t"] = t + 1 
        

def ex_lr_tuning():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)

    for t in range(10):
        opt.zero_grad() 
        loss = (weights**2).mean() 
        print(loss.cpu().item())
        loss.backward()  
        opt.step() 
        print(loss.item())

    
class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999), 
                 eps=1e-8, 
                 weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']

                state['step'] += 1
                t = state['step']

                state['m'] = beta1 * m + (1 - beta1) * grad
                state['v'] = beta2 * v + (1 - beta2) * (grad * grad)
                if weight_decay != 0:
                    p.data = p.data - lr * weight_decay * p.data
                m_hat = state['m'] / (1 - beta1 ** t)
                v_hat = state['v'] / (1 - beta2 ** t)
                update_value = lr * (m_hat / (v_hat.sqrt() + eps))
                p.data = p.data - update_value

def cos_learning_rate_schedule(t:int, 
                               alpha_max:float, 
                               alpha_min:float, 
                               warm_up_steps:int, 
                               iterations:int)-> float:
    lr = 0
    if t < warm_up_steps:
        lr = t/warm_up_steps * alpha_max
    elif t > iterations:
        lr = alpha_min
    else:
        lr = alpha_min + 0.5 * (1 + math.cos((t - warm_up_steps) / (iterations - warm_up_steps) * math.pi))  * (alpha_max - alpha_min) 
    return lr

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.data.mul_(scale)
    
def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu")->None:
    n = len(x)
    ix = np.random.randint(0, n - context_length , size=batch_size)
    
    inputs = np.stack([x[i : i + context_length] for i in ix])
    targets = np.stack([x[i + 1 : i + 1 + context_length] for i in ix])
    
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    
    return inputs, targets 

def save_checkpoint(model, optimizer, iteration: int, out: Union[str, os.PathLike, BinaryIO]):

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(model, optimizer, inp: Union[str, os.PathLike, BinaryIO]):
    
    checkpoint = torch.load(inp, map_location="cpu")
    state_dict = checkpoint["model_state"]
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]
    return iteration

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
def train(config):
    print('-'*20)
    print('Start Training')
    print(config)
    print('-'*20)
    print('loading Wandb')
    print('-'*20)
    if config.use_wandb:
        wandb.init(project="nlp-hw1", config=config.__dict__)
        model_tag = "lstm" if getattr(config, "is_LSTM", False) else "transformer"
        wandb.run.name = f"{model_tag}_d{config.d_model}_lr{config.lr}_b{config.batch_size}"
    
    device = torch.device(config.device)
    print('lodaing data')
    print('-'*20)
    train_data = np.load(config.train_data, mmap_mode="r")
    val_data = np.load(config.val_data, mmap_mode="r")
    print('loading model')
    print('-'*20)
    vocab_size = config.vocab_size
    if getattr(config, "is_LSTM", False):
        model = LSTMLM(
            vocab_size=vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            d_model=config.d_model,
            device=device
        ).to(device)
    else:
        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=config.context_length,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            device=device
        ).to(device)
    print('lodading optimizer')
    print('-'*20)
    optimizer = AdamW(model.parameters(), 
                      lr=config.lr, 
                      betas=(0.9, 0.95), 
                      weight_decay=config.weight_decay)

    start_iter = 0
    if config.resume and os.path.exists(config.resume):
        print(f"Resuming from checkpoint: {config.resume}")
        start_iter = load_checkpoint(model, optimizer, config.resume)
        print(f"Resumed from iteration {start_iter}")

    for iteration in tqdm(range(start_iter, config.max_iters)):
        lr = cos_learning_rate_schedule(
            t=iteration,
            alpha_max=config.lr,
            alpha_min=config.lr_min,
            warm_up_steps=config.warmup_iters,
            iterations=config.cosine_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, config.batch_size, config.context_length, device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0] 
        loss = cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_norm = 1.0)
        optimizer.step()

        if iteration % config.log_interval == 0:
            print(f"Iter {iteration}: loss = {loss.item():.4f}, lr = {lr:.2e}")
            if config.use_wandb:
                wandb.log({
                            "train/loss": loss.item(),
                            "iteration": iteration + start_iter
                        })
            

        if iteration % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_x, val_y = get_batch(val_data, config.batch_size, config.context_length, device)
                val_logits = model(val_x)
                if isinstance(val_logits, tuple):
                    val_logits = val_logits[0]
                val_loss = cross_entropy(val_logits, val_y)
                ppl = torch.exp(val_loss)
                print(f"Validation loss = {val_loss.item():.4f}")
            model.train()
            if config.use_wandb:
                wandb.log({
                            "val/loss": val_loss.item(),
                            "val/perplexity": ppl.item(),
                            "iteration": iteration + start_iter
                        })

        if iteration % config.ckpt_interval == 0 and iteration > 0:
            save_path = os.path.join(config.out_dir, f"checkpoint_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, save_path)
            print(f"Checkpoint saved to {save_path}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(**config_dict)
    
    train(config)
