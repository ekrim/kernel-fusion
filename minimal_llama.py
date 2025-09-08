import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=11008):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size=4096, num_attention_heads=32, num_key_value_heads=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        # Minimal attention - just return projected output for shape compatibility
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Simplified attention computation
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        # Basic attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=11008, num_attention_heads=32, 
                 num_key_value_heads=32, rms_norm_eps=1e-6):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_attention_heads, num_key_value_heads)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, **kwargs):
        # hidden_states: [batch_size, sequence_length, 4096]
        # Pre-attention norm + self-attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        # Pre-MLP norm + MLP + residual  
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

def time_fn(fn, warmup=30, iters=100):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    
    with torch.profiler.record_function("model_execution"):
        for _ in range(iters):
            out = fn()
    torch.cuda.synchronize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiled', action='store_true', help='Use torch.compile for optimization')
    args = parser.parse_args()
    
    # Test tensor for LlamaDecoderLayer.forward()
    batch_size = 2
    seq_len = 128
    hidden_size = 4096
    hidden_states = torch.randn(batch_size, seq_len, hidden_size,
        dtype=torch.float16, device="cuda")

    transformer = LlamaDecoderLayer(hidden_size=hidden_size).cuda()

    if args.compiled:
        # Completely disable CUDA graphs
        import os
        os.environ['TORCH_COMPILE_DEBUG'] = '1'
        os.environ['PYTORCH_DISABLE_CUDA_GRAPHS'] = '1'
        transformer = torch.compile(transformer, fullgraph=False, dynamic=False, mode="max-autotune")
        #torch._inductor.config.triton.cudagraphs = False

    transformer.eval()
    with torch.inference_mode():
        time_fn(lambda: transformer(hidden_states))
        print("Profiling complete")
