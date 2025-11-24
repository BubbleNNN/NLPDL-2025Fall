import torch
import torch.nn as nn
from typing import Optional,Tuple

class Linear(nn.Module):
  """Applies a linear transformation to the input: y = xA^T + b."""

  def __init__(
       self,
       in_features: int,
       out_features: int,
       bias: bool = False,
       device: Optional[torch.device] = None,
       dtype: Optional[torch.dtype] = None,
  ) -> None:
       """Initializes the linear module.

       Args:
           in_features (int): Size of each input sample.
           out_features (int): Size of each output sample.
           bias (bool, optional): If True, includes a bias term. Defaults to False.
           device (torch.device, optional): Device to store parameters. Defaults to None.
           dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
       """
       super().__init__()
       self.weight = nn.Parameter(torch.empty(size = (out_features, in_features), device = device, dtype = dtype))
       sigma = (2 / (in_features + out_features)) ** 0.5
       self.weight = nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
       if bias:
            self.b = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
       else:
            self.register_parameter("b", None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Applies the linear transformation.

       Args:
           x (torch.Tensor): Input tensor of shape (..., in_features).

       Returns:
           torch.Tensor: Output tensor of shape (..., out_features).
       """
       if self.b is not None:
           return x @ self.weight.T  + self.b
       else:
           return x @ self.weight.T
class Embedding(nn.Module):
  """A lookup table that maps indices to embedding vectors."""

  def __init__(
       self,
       num_embeddings: int,
       embedding_dim: int,
       device: Optional[torch.device] = None,
       dtype: Optional[torch.dtype] = None,
  ) -> None:
       """Initializes the embedding module.

       Args:
           num_embeddings (int): Size of the vocabulary.
           embedding_dim (int): Dimension of the embedding vectors.
           device (torch.device, optional): Device to store parameters. Defaults to None.
           dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
       """
       super().__init__()
       self.weight= nn.Parameter(torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype))
       self.weight = nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
       """Looks up embedding vectors for token IDs.

       Args:
           token_ids (torch.Tensor): Input tensor of shape (...).

       Returns:
           torch.Tensor: Output tensor of shape (..., embedding_dim).
       """
       return self.weight[token_ids]
class RMSNorm(nn.Module):
  """Applies Root Mean Square Layer Normalization (RMSNorm)."""  

  def __init__(
       self,
       d_model: int,
       eps: float = 1e-5,
       device: Optional[torch.device] = None,
       dtype: Optional[torch.dtype] = None,
  ) -> None:
       """Initializes the RMSNorm module.

       Args:
           d_model (int): Hidden dimension of the model.
           eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
           device (torch.device, optional): Device to store parameters. Defaults to None.
           dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
       """
       super().__init__()
       self.weight = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
       self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Applies RMSNorm to the input.

       Args:
           x (torch.Tensor): Input tensor of shape (..., d_model).

       Returns:
           torch.Tensor: Output tensor of shape (..., d_model).
       """
       in_dtype = x.dtype
       x = x.to(torch.float32)
       rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
       result = x/rms * self.weight
       return result.to(in_dtype)
class SwiGLU(nn.Module):
  """Applies the SwiGLU feedforward transformation."""

  def __init__(
       self,
       d_model: int,
       d_ff: int,
       device: Optional[torch.device] = None,
       dtype: Optional[torch.dtype] = None,
  ) -> None:
       """Initializes the SwiGLU module.

       Args:
           d_model (int): Hidden dimension of the model.
           d_ff (int): Inner dimension of the feedforward layer.
           device (torch.device, optional): Device to store parameters. Defaults to None.
           dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
       """
       super().__init__()
       self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
       self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
       self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Applies the SwiGLU transformation.

       Args:
           x (torch.Tensor): Input tensor of shape (..., d_model).

       Returns:
           torch.Tensor: Output tensor of shape (..., d_model).
       """
       x1 = self.w1(x)
       x3 = self.w3(x)
       silu = x1 * torch.sigmoid(x1)
       return self.w2(silu * x3)    
   
class RoPE(nn.Module):
    """Applies Rotary Position Embeddings (RoPE)."""
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        """Initializes the RoPE module.
        Args:
            theta (float): Θ value for the rotary embedding.
            d_k (int): Dimension of query and key vectors.
            max_seq_len (int): Maximum sequence length supported.
            device (torch.device, optional): Device to store buffers. Defaults to None.
        """
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        half_dim = d_k // 2
        freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (freq_seq / half_dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Applies rotary position embeddings.
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k).
            token_positions (torch.Tensor): Tensor of shape (..., seq_len) specifying token positions.
        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_k).
        """
        if token_positions is None:
            seq_len = x.shape[-2]

            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )
        return x_rotated.flatten(-2, -1)
    
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """Softmax activation function.

  Applies the softmax function to the input tensor along the specified dimension.

  Args:
   x: Input tensor.
   dim: Dimension along which softmax will be computed. Defaults to -1.

  Returns:
   Tensor with softmax applied along the specified dimension.
  """
  x_max, _ = torch.max(x, dim=dim, keepdim=True)
  x_exp = torch.exp(x - x_max)
  return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
   query: torch.Tensor,
   key: torch.Tensor,
   value: torch.Tensor,
   mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
   """Scaled dot-product attention function.

   Args:
       query: Tensor of shape (batch_size, ..., seq_len_q, d_k)
       key: Tensor of shape (batch_size, ..., seq_len_k, d_k)  
       value: Tensor of shape (batch_size, ..., seq_len_v, d_v)
       mask: Boolean tensor of shape (seq_len_q, seq_len_k) or broadcastable shape

   Returns:
       Tensor of shape (batch_size, ..., seq_len_q, d_v)
   """
   score = torch.einsum("...id,...jd->...ij",query, key) / query.shape[-1]**0.5
   masked_score = score.masked_fill(~mask, -torch.inf)
   return softmax(masked_score, -1)@value
    

class CasualMultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
    self,
    d_model: int,
    num_heads: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_rope: bool = False,
    theta: Optional[float] = None,
    max_seq_len: Optional[int] = None,
    ) -> None:
        """Initializes the attention module.

        Args:
            d_model (int): Hidden dimension of the model.
            num_heads (int): Number of attention heads.
            device (torch.device, optional): Device to store parameters. Defaults to None.
        dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
            use_rope (bool, optional): Whether to apply RoPE. Defaults to False.
            theta (float, optional): Θ parameter for RoPE when enabled. Defaults to None.
            max_seq_len (int, optional): Maximum sequence length for RoPE buffers.
                Defaults to None.
        """
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, num_heads * self.d_k, device, dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_k, device, dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_v,device, dtype )
        self.output_proj = Linear(num_heads * self.d_v, d_model, device, dtype)
        
        if self.use_rope:
            self.rope = RoPE(self.theta,self.d_k,self.max_seq_len,)
    def forward(
    self,
    x: torch.Tensor,
    token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Applies causal multi-head self-attention.

        Args:
        x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).
            token_positions (torch.Tensor, optional): Tensor of shape (..., seq_len)
                with token positions; required if `use_rope` is True. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (..., seq_len, d_model).
        """
        import einops
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)
        
        Q = einops.rearrange(Q,"... seq (h d)->... h seq d", h = self.num_heads)
        K = einops.rearrange(K,"... seq (h d)->... h seq d", h = self.num_heads)
        V = einops.rearrange(V,"... seq (h d)->... h seq d", h = self.num_heads)
        
        if self.use_rope:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
            
        attn = scaled_dot_product_attention(Q,K,V,mask)
        attn_output = einops.rearrange(attn,"... h seq d->... seq (h d)")
        
        return self.output_proj.forward(attn_output)

class TransformerBlock(nn.Module):
  """A single Transformer block with self-attention and feedforward network."""

  def __init__(
       self,
       d_model: int,
       num_heads: int,
       d_ff: int,
       device: Optional[torch.device] = None,
       dtype: Optional[torch.dtype] = None,
       use_rope: bool = False,
       theta: Optional[float] = None,
       max_seq_len: Optional[int] = None,
  ) -> None:
       """Initializes the Transformer block.

       Args:
           d_model (int): Hidden dimension of the model.
           num_heads (int): Number of attention heads.
           d_ff (int): Hidden dimension of the feedforward layer.
           device (torch.device, optional): Device to store parameters. Defaults to None.
           dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
           use_rope (bool, optional): Whether to apply RoPE in self-attention. Defaults to False.
           theta (float, optional): Θ parameter for RoPE. Defaults to None.
           max_seq_len (int, optional): Maximum sequence length for RoPE buffers. Defaults to None.
       """
       super().__init__()
       self.ln1 = RMSNorm(d_model, device = device, dtype = dtype)
       self.ln2 = RMSNorm(d_model, device = device, dtype = dtype)
       self.attn = CasualMultiheadSelfAttention(d_model, num_heads, device, dtype, use_rope, theta, max_seq_len)
       self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
       """Applies the Transformer block.

       Args:
           x (torch.Tensor): Input tensor of shape (..., seq_len, d_model).

       Returns:
           torch.Tensor: Output tensor of shape (..., seq_len, d_model).
       """
       y =  x + self.attn.forward(self.ln1(x))
       output = y +self.ffn (self.ln2(y))
       return output
class TransformerLM(nn.Module):
  """A Transformer-based language model."""

  def __init__(
       self,
       vocab_size: int,
       context_length: int,
       num_layers: int,
       d_model: int,
       num_heads: int,
       d_ff: int,
       device: Optional[torch.device] = None,
       dtype: Optional[torch.dtype] = None,
       use_rope: bool = False,
       theta: Optional[float] = None,
  ) -> None:
       """Initializes the Transformer language model.

       Args:
           vocab_size (int): Vocabulary size for token embeddings.
           context_length (int): Maximum sequence length for positional encodings.
           num_layers (int): Number of Transformer blocks.
           d_model (int): Hidden dimension of the model.
           num_heads (int): Number of attention heads.
           d_ff (int): Hidden dimension of the feedforward layer.
           device (torch.device, optional): Device to store parameters. Defaults to None.
           dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
           use_rope (bool, optional): Whether to apply RoPE. Defaults to False.
           theta (float, optional): Θ parameter for RoPE. Defaults to None.
       """
       super().__init__()
       self.token_embeddings = Embedding(num_embeddings = vocab_size, embedding_dim = d_model, device = device, dtype = dtype)
       self.layers = nn.ModuleList([TransformerBlock(d_model = d_model, 
                                                     num_heads=num_heads, 
                                                     d_ff = d_ff, 
                                                     device = device, 
                                                     dtype = dtype, 
                                                     use_rope = use_rope, 
                                                     theta = theta, 
                                                     max_seq_len = context_length) 
                                    for i in range (num_layers)])
       self.ln_final = RMSNorm(d_model, device = device, dtype = dtype)
       self.lm_head = Linear(d_model, vocab_size, device = device, dtype = dtype)
       self.context_length = context_length
       self.vocab_size = vocab_size

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
       """Applies the Transformer language model.

       Args:
           input_ids (torch.Tensor): Token IDs of shape (..., seq_len).

       Returns:
           torch.Tensor: Logits of shape (..., seq_len, vocab_size).
       """
       x = self.token_embeddings(input_ids)
       for block in self.layers:
            x = block(x)
       return self.lm_head(self.ln_final(x))
   
  @torch.no_grad()
  def generate(self, 
               input_ids, 
               max_new_tokens=50, 
               temperature=1.0, 
               top_p=0.9, 
               eos_token_id=None):
        
        self.eval()
        generated = input_ids.clone()

        for i in range(max_new_tokens):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :]  
            next_logits = next_logits / temperature
            probs = softmax(next_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)

            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated


class LSTMCell(nn.Module):
    """A single Long Short-Term Memory (LSTM) cell."""

    def __init__(
    self,
    d_model: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initializes the LSTM cell.

        Args:
            d_model (int): Hidden dimension of the LSTM.
            device (torch.device, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type of parameters. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.x_proj = Linear(d_model, 4 * d_model, bias=True, device=device, dtype=dtype)
        self.h_proj = Linear(d_model, 4 * d_model, bias=True, device=device, dtype=dtype)

    def forward(
    self,
    x: torch.Tensor,
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LSTM cell.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, d_model).
            state (tuple[torch.Tensor, torch.Tensor], optional): Tuple of
                (hidden_state, cell_state), each of shape (batch_size, d_model).
                If None, both are initialized to zeros. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The next (hidden_state, cell_state),
            each of shape (batch_size, d_model).
        """
        batch_size = x.size(0)
        if state is None:
            h_prev = x.new_zeros(batch_size, self.d_model)
            c_prev = x.new_zeros(batch_size, self.d_model)
        else:
            h_prev, c_prev = state

        gates = self.x_proj(x) + self.h_proj(h_prev)
        i, f, o, g = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t
    
class LSTM(nn.Module):
    """Stacked LSTM with multiple layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.layers = nn.ModuleList([
            LSTMCell(d_model, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Applies stacked LSTM layers.

        Args:
            x: (batch_size, seq_len, d_model)
            state: Optional[(num_layers, batch_size, d_model), (num_layers, batch_size, d_model)]

        Returns:
            (output, (h_n, c_n))
            output: (batch_size, seq_len, d_model)
            h_n, c_n: each (num_layers, batch_size, d_model)
        """
        batch_size, seq_len, _ = x.shape

        if state is None:
            h_prev = x.new_zeros(self.num_layers, batch_size, self.d_model)
            c_prev = x.new_zeros(self.num_layers, batch_size, self.d_model)
        else:
            h_prev, c_prev = state

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_next, c_next = [], []
            for layer_idx, cell in enumerate(self.layers):
                h_t, c_t = cell(x_t, (h_prev[layer_idx], c_prev[layer_idx]))
                h_next.append(h_t)
                c_next.append(c_t)
                x_t = h_t  
            h_prev = torch.stack(h_next)
            c_prev = torch.stack(c_next)
            outputs.append(x_t.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        return output, (h_prev, c_prev)
    
class LSTMLM(nn.Module):
    """An LSTM-based language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Initializes the LSTM language model.

        Args:
            vocab_size (int): Vocabulary size for token embeddings.
            context_length (int): Max sequence length.
            num_layers (int): Number of LSTM layers.
            d_model (int): Hidden dimension of the model.
        """
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.lstm = LSTM(num_layers=num_layers, d_model=d_model, device=device, dtype=dtype)
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_proj = Linear(in_features=d_model,out_features=vocab_size,bias=False,device=device,dtype=dtype)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Applies the LSTM language model.

        Args:
            input_ids: (batch_size, seq_len)
            state: Optional hidden/cell state

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)  
        output, new_state = self.lstm(x, state)  
        output = self.norm(output)
        logits = self.output_proj(output)
        return logits, new_state
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_p=0.9, eos_token_id=None):
        self.eval()
        generated = input_ids.clone()
        state = None

        for _ in range(max_new_tokens):
            logits, state = self.forward(generated[:, -1:], state=state)
            next_logits = logits[:, -1, :]
            next_logits = next_logits / temperature
            probs = softmax(next_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)

            generated = torch.cat([generated, next_token], dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated