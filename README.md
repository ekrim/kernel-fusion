ncu -k "cublasLtMatmul|triton_.*" --target-processes all \
    --set full python bench.py



nsys profile -c nvtx -w true -p "BENCH"@* -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 python minimal_llama.py
nsys profile -c cudaProfilerApi -w true python minimal_llama.py

nsys stats --report cuda_gpu_kern_sum --format csv --output - report1.nsys-rep