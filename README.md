ncu -k "cublasLtMatmul|triton_.*" --target-processes all \
    --set full python bench.py



nsys profile -c nvtx -w true -p "BENCH"@* -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 python minimal_llama.py
nsys profile -c cudaProfilerApi -w true python minimal_llama.py

nsys stats --report cuda_gpu_kern_sum --format csv --output - report1.nsys-rep


# Nsight Compute

ncu --set full \
    --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_active,lts__t_bytes.sum,sm__pipe_tensor_active.avg.pct_of_peak_sustained_active \
    --kernel-name regex:ampere_sgemm_.* \
    -o gemm_profile \
    python minimal_llama.py --compiled


# Nsight Systems (all metrics)

TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 nsys profile --trace=cuda,cublas,nvtx,osrt --cuda-graph-trace=node --gpu-metrics-device=all -o out_profile python minimal_llama.py --compiled


# Avoid graphs

# We're using this one
TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1  nsys profile --cuda-graph-trace=node python minimal_llama.py --compiled

import torch
import torch._inductor.config as iCfg

iCfg.triton.cudagraphs = False  # disable CUDA graphs for Inductor/Triton
model = torch.compile(model, options={"triton.cudagraphs": False}, mode="max-autotune")




# Profile with CUDA events

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
s = torch.cuda.current_stream()

start.record(s)
# enqueue work...
end.record(s)
end.synchronize()
ms = start.elapsed_time(end)
from torch.cuda import nvtx
nvtx.mark(f"attn.step gpu_ms={ms:.3f}")



TODO

Mandatory
- get minimal_llama to run on CUDA!
- see if the compiled/eager produce actual cuda_gpu_kern_sum with nvtx signposts
- NEW: TRY CUDA EVENTS

Nice
- Analyze the difference between eager and compiled, find an opportunity for fusion
- Demonstrate fusion (cublasLt + epilogue?)
