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

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else :
        return (
            # (B, seq_len, H_KV, 1, Head_Dim) 
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim) # introduce a new dimension
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim) # and then wrap it 
        )

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
    

class SelfAttention(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        # remove parallelization : because of less of GPU 
        # Indicates the number of heads for the keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the queries
        self.n_heads_q = args.n_heads
        # Indicates the number of repetitions for the heads of keys and values
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of the heads (because we split the dimension in n_heads parts)
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))


    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, seq_len, dim) -> seq_len : 1 (only one token)

        # Apply the Wq, Wk, and Wv matrices to queries, keys, and values
        # (B, 1, dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, 1, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, 1, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, 1, self.n_kv_heads, self.head_dim)

        # Does not change the shape of the tensors
        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # Retrieve all the cached keys and values
        # (B, seq_len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        # Repeat the heads of the K and V to match the number of heads of the queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2) 
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_KV, Head_Dim, seq_len_KV) -> (B, H_Q, 1, seq_len_KV)
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)

        # (B, H_Q, 1, seq_len) @ (B, H_KV, seq_len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, dim) -> (B, 1, dim)


class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        # hidden_size = 7, multiple_of = 5
        # (7 + 4) //5 = 2
        # 2 * 5 = 10

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x



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

        return output

