import os, time
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- JIT-build & load the extension (no pybind; dispatcher registration only) ---
root = Path(__file__).parent
ext = load(
    name="fused_linear_resid",  # arbitrary; not used directly
    sources=[str(root/"src/fused_linear_resid.cpp"),
             str(root/"src/lt_linear_bias_act_resid.cu")],
    extra_ldflags=["-lcublasLt","-lcublas"],
    verbose=True,
)

# after load(), TORCH_LIBRARY registrations are active:
# call as: torch.ops.myops.linear_bias_gelu_residual(...)

@torch.inference_mode()
def mlp_ref(x, W1, b1, W2, b2, residual):
    y = torch.nn.functional.gelu(x @ W1 + b1, approximate="tanh")
    return y @ W2 + b2 + residual

def make_inputs(M=2048, d=4096, ff=11008, dtype=torch.bfloat16):
    dev = "cuda"
    x  = torch.randn(M, d,  dtype=dtype, device=dev)
    W1 = torch.randn(d, ff, dtype=dtype, device=dev)
    b1 = torch.randn(ff,   dtype=dtype, device=dev)
    W2 = torch.randn(ff, d, dtype=dtype, device=dev)
    b2 = torch.randn(d,    dtype=dtype, device=dev)
    res= torch.randn(M, d, dtype=dtype, device=dev)
    return x, W1, b1, W2, b2, res

def time_fn(fn, warmup=30, iters=100):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt, out

def tflops(M, d, ff, seconds):
    flops = 2*M*d*ff + 2*M*ff*d  # two GEMMs
    return flops / seconds / 1e12

def run_case(M, d, ff, dtype):
    x, W1, b1, W2, b2, residual = make_inputs(M, d, ff, dtype)

    # 1) eager
    def eager():
        return mlp_ref(x, W1, b1, W2, b2, residual)
    t_e, y_e = time_fn(eager)

    # 2) torch.compile
    compiled_fn = torch.compile(eager, fullgraph=False, dynamic=False, mode="max-autotune")
    t_c, y_c = time_fn(compiled_fn)

    # 3) cuBLASLt epilogues via our dispatcher op
    def lt_op():
        return torch.ops.myops.linear_bias_gelu_residual(x, W1, b1, W2, b2, residual)
    t_l, y_l = time_fn(lt_op)

    # check numeric diff (fp32 compare)
    err_ce = (y_c.float() - y_e.float()).abs().max().item()
    err_le = (y_l.float() - y_e.float()).abs().max().item()

    print(f"\nM={M} d={d} ff={ff} dtype={dtype}")
    print(f"eager:          {t_e*1e3:7.2f} ms  | {tflops(M,d,ff,t_e):6.2f} TFLOP/s")
    print(f"torch.compile:  {t_c*1e3:7.2f} ms  | {tflops(M,d,ff,t_c):6.2f} TFLOP/s  (max|Δ| vs eager={err_ce:.3e})")
    print(f"cuBLASLt epi:   {t_l*1e3:7.2f} ms  | {tflops(M,d,ff,t_l):6.2f} TFLOP/s  (max|Δ| vs eager={err_le:.3e})")

if __name__ == "__main__":
    # Optional: see Inductor decisions
    os.environ.setdefault("TORCH_LOGS", "graph")

    cases = [
        (32,   4096, 11008, torch.float16),   # decode-ish
        (2048, 4096, 11008, torch.float16),   # prefill mid
        (8192, 4096, 11008, torch.float16),   # prefill large
        (2048, 4096, 11008, torch.bfloat16),
    ]
    for M, d, ff, dt in cases:
        run_case(M, d, ff, dt)
