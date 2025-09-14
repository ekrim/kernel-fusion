import argparse
import torch
import torch._inductor.config as iCfg
import torch.nn as nn
from torch.cuda import nvtx
import torch.nn.functional as F


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

    def forward(self, hidden_states, past_key_value=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        # Handle KV cache for decode
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)

        # Store current KV for next decode step
        present_key_value = (key_states, value_states)

        kv_seq_len = key_states.shape[1]

        # Basic attention
        # query_states: [bsz, q_len, num_heads, head_dim]
        # key_states: [bsz, kv_seq_len, num_kv_heads, head_dim]
        # Need to transpose for matmul: [bsz, num_heads, q_len, head_dim] x [bsz, num_kv_heads, head_dim, kv_seq_len]
        query_states = query_states.transpose(1, 2)  # [bsz, num_heads, q_len, head_dim]
        key_states = key_states.transpose(1, 2)      # [bsz, num_kv_heads, kv_seq_len, head_dim]
        value_states = value_states.transpose(1, 2)  # [bsz, num_kv_heads, kv_seq_len, head_dim]
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        # attn_output: [bsz, num_heads, q_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [bsz, q_len, num_heads, head_dim]
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=11008, num_attention_heads=32, 
                 num_key_value_heads=32, rms_norm_eps=1e-6):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_attention_heads, num_key_value_heads)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states, past_key_value=None, **kwargs):
        # hidden_states: [batch_size, sequence_length, 4096]
        # Pre-attention norm + self-attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(hidden_states, past_key_value=past_key_value, **kwargs)
        hidden_states = residual + hidden_states

        # Pre-MLP norm + MLP + residual  
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value

def time_fn(fn, warmup=20, iters=3):
    for _ in range(warmup):
        _ = fn()

    s = torch.cuda.current_stream()
    s.synchronize()

    for _ in range(iters):
        nvtx.range_push("BENCHMARK")
        out = fn()
        s.synchronize()
        nvtx.range_pop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiled', action='store_true', help='Use torch.compile for optimization')
    parser.add_argument('--decode', action='store_true', help='Simulate decode instead of prefill')
    args = parser.parse_args()

    batch_size = 1
    hidden_size = 4096
    transformer = LlamaDecoderLayer(hidden_size=hidden_size).cuda()

    if args.compiled:
        iCfg.triton.cudagraphs = False  # disable CUDA graphs for Inductor/Triton
        transformer = torch.compile(transformer, options={"triton.cudagraphs": False, "max-autotune": True})

    transformer.eval()
    with torch.inference_mode():
        if args.decode:
            # Decode simulation: process tokens one by one
            seq_len = 512  # context length
            # Prefill first
            prefill_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda")
            _, past_kv = transformer(prefill_input)

            # Now decode new tokens one by one
            def decode_step():
                token_input = torch.randn(batch_size, 1, hidden_size, dtype=torch.float16, device="cuda")
                _, new_kv = transformer(token_input, past_key_value=past_kv)
                return new_kv

            time_fn(decode_step)
            print("Decode profiling complete")
        else:
            # Original prefill behavior
            seq_len = 512
            hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda")
            time_fn(lambda: transformer(hidden_states))
            print("Prefill profiling complete")
