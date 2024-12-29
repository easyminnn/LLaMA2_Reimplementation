import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the querires
    n_kv_heads: Optional[int] = None # Number of heads for the keys and values
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len : int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len : int, device: str, theta: float = 10000.0):
    # As written in the paper, we need to precompute the frequencies for the positional encoding
    assert head_dim % 2 == 0, "Head dim must be divisible by 2"

    # Build the theta parameters
    # According to the formula theta_i = 10000^(-2(i-1) / dim) for i = [1, 2, ..., dim / 2]
    # Shape : (Head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape : (Head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the "m" parameter)
    # Shape : (seq_len)
    m = torch.arange(seq_len, device = device)
    # Multiply each theat by each position using the outer product
    # Shape : (seq_len) outer product* (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # Shape : (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device : str):
    # x : (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    #  elementwise : (B, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (B, seq_len, H, head_dim /2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim / 2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # The gamma parameter     
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # x : (B, seq_len, dim)
        return x  * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps)
        
    def forward(self, x: torch.Tensor):
        # dim *  (B, seq_len, dim) = (B, seq_len, dim)
        return self._norm(x.float()).type_as(x) * self.weight
    

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the self attention
        self.attention_norm = RMSNorm(self.dim, eps = args.norm_eps)
        # Normalization BEFORE the feedforward
        self.ffn_norm = RMSNorm(self.dim, eps = args.norm_eps)


    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out 
    




        
    

class Transformer(nn.Module):

    def __init__(self, args : ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != 0, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim) # First, embedding layer

        self.layers = nn.ModuleList([])

        for _ in range(args.n_layers): # list of layers
            self.layers.append(EncoderBlock(args))
 
        self.norm = RMSNorm(args.dim, eps = args.norm_eps) # eps is used for normalization calculation
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only supporting one token at a time"

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens) # dim of embeddings : 4096

        # Retrieve the pairs (m, theta) for the positional encoding [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()

