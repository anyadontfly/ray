import logging
import random
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch import Tensor


logger = logging.getLogger()

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * norm


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()

        self.dim = dim
        self.base = base

        # Compute inverse frequency (used for RoPE)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Applies Rotary Positional Embeddings (RoPE).
        x: (batch_size, seq_len, num_heads, head_dim)
        """

        batch_size, seq_len, num_heads, head_dim = x.shape

        # Ensure head_dim is divisible by 2
        assert head_dim % 2 == 0, "head_dim must be divisible by 2 for Rotary Embeddings"

        # Compute position index (shape: seq_len, 1)
        t = torch.arange(seq_len, device=x.device).unsqueeze(1)

        # Compute angles (shape: seq_len, head_dim/2)
        angle = t * self.inv_freq.unsqueeze(0)

        # Compute sin and cos (shape: seq_len, head_dim/2)
        sin, cos = angle.sin(), angle.cos()

        # Expand dimensions to match input shape: (batch_size, seq_len, num_heads, head_dim/2)
        sin = sin.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, num_heads, head_dim // 2)
        cos = cos.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, num_heads, head_dim // 2)

        # Split x into even and odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotary transformation
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return x_rotated


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # Ensure head_dim is consistent

        # Rotary Embedding applied per head
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        # Corrected Projection Layers: All match head_dim × num_heads = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # (1024 → 1024)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # (1024 → 1024)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # (1024 → 1024)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)  # (1024 → 1024)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply Rotary Embedding to Queries and Keys
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        # Compute Self-Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads back (batch_size, seq_len, hidden_size)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_heads)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, x):
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x)
        x = x + attn_output
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x)
        return x + mlp_output


class Llama(nn.Module):
    # def __init__(self, vocab_size=128256, hidden_size=4096, num_layers=32, num_heads=32, intermediate_size=14336):
    def __init__(self, vocab_size=128256, hidden_size=1024, num_layers=4, num_heads=4, intermediate_size=3584):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x)

        logits = self.fc(x)
        return logits


class BucketParameter(nn.Module):
    def __init__(self, layers: List[nn.Module], to_flat: bool = False):
        super().__init__()
        if to_flat:
            assert len(layers) == 1
        self.layers = torch.nn.ModuleList(layers)
        self.to_flat = to_flat

        self.x = None
        self.y = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.layers.parameters(), lr=0.001)

    def forward(self, x: Tensor) -> Tensor:
        if self.to_flat:
            x = torch.flatten(x, 1)
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(
        self,
        loss: Optional[torch.Tensor] = None,
        pred: Optional[torch.Tensor] = None,
        grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if loss is not None:
            assert pred is None
            loss.backward()
        elif pred is not None:
            assert grad is not None
            pred.backward(grad)

        # [TODO] Check if `parameters()` is deterministic.
        grads_cat = parameters_to_vector(
            [p.grad for p in self.layers.parameters() if p.grad is not None]
        )
        return grads_cat

    def update(self, grads_cat: torch.Tensor, grads_passed: bool) -> None:
        if grads_passed:
            offset = 0
            # [TODO] Check if `parameters()` is deterministic.
            for p in self.layers.parameters():
                if p.grad is None:
                    continue
                size = p.data.numel()
                grad = grads_cat[offset : offset + size].reshape(p.data.shape)
                p.grad = grad
                offset += size

        self.optimizer.step()
        self.optimizer.zero_grad()


class LlamaMP(nn.Module):
    def __init__(self, vocab_size=128256, hidden_size=1024, num_layers=4, num_heads=4, intermediate_size=3584):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, vocab_size, bias=False)
        self.process_bucket_params()

    def process_bucket_params(self):
        def show_layer_size(layer, indent=0):
            num_params = sum(p.numel() for p in layer.parameters())
            size_mib = num_params * 4 / (1024 * 1024)
            indent_str = "  " * indent
            logger.info(f"{indent_str}{layer.__class__.__name__}: {size_mib:.2f} MiB")
            if size_mib < 25:
                return
            for _, child in layer.named_children():
                show_layer_size(child, indent + 1)

        def calculate_layer_size(layer) -> float:
            num_params = sum(p.numel() for p in layer.parameters())
            size_mib = num_params * 4 / (1024 * 1024)
            return size_mib
        
        for layer in self.layers:
            show_layer_size(layer)
        
        BUCKET_SIZE = 120
        self.bucket_params: List[BucketParameter] = []
        bucket_layers: List[nn.Module] = []
        bucket_size = 0
        for layer in self.layers:
            size = calculate_layer_size(layer)
            if bucket_size + size <= BUCKET_SIZE:
                bucket_layers.append(layer)
                bucket_size += size
            else:
                self.bucket_params.append(BucketParameter(bucket_layers))
                bucket_layers = [layer]
                bucket_size = size
        if len(bucket_layers) > 0:
            self.bucket_params.append(BucketParameter(bucket_layers))
        self.bucket_params.append(BucketParameter([self.fc]))

        for bparam in self.bucket_params:
            logger.info(
                f"Bucket size: {sum(calculate_layer_size(m) for m in bparam.layers):.2f} MiB"
            )
            for layer in bparam.layers:
                logger.info(f"  {layer.__class__.__name__}")

    def forward(self, input_ids):
        x = self.embed(input_ids)

        for bparam in self.bucket_params:
            x = bparam(x)
        return x


set_seed(32)
model = Llama().to("cuda")
input_ids = torch.full((10, 1000), torch.randint(0, 128256, (1,)).item(), dtype=torch.long).to("cuda")
output = model(input_ids)

# print(model)
print(f"Output value: {output}")
# print(f"Output shape: {output.shape}")  # Expected: (1, 10, 128256)
