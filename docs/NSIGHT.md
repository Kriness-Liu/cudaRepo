# Nsight profiling workflow

Install Nsight Systems and Nsight Compute, then check that `nsys` and `ncu` are available.

Capture the CPU/CUDA timeline and NVTX ranges:

```powershell
New-Item -ItemType Directory -Force nsight | Out-Null
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true --output=nsight/cudarepo_timeline python scripts/profile_extension.py
```

Inspect one fused kernel in Nsight Compute:

```powershell
ncu --set full --kernel-name regex:fused_bias_relu --launch-count 1 --export nsight/fused_bias_relu python scripts/profile_extension.py --iterations 5
```

Questions to answer from the reports:

1. How many kernel launches and memory copies occur in each NVTX range?
2. Does the extension execute on PyTorch's current stream?
3. Are global loads/stores coalesced, and what limits achieved occupancy?
4. Is the fused operator memory-bound, launch-bound, or compute-bound?
5. Does the measured timeline agree with CUDA Event P50/P95 results?

`.nsys-rep`, `.ncu-rep` and exported SQLite files belong in `nsight/` and are not committed by default.
