# kernels

CUDA kernels written from scratch and benchmarked against cuBLAS.

## Layout

```
common/cuda_utils.cuh   error-checking macros (CUDA_CHECK, CUBLAS_CHECK), CEIL_DIV
scripts/gpu.sh          sync this repo to a remote GPU box and run a command there
SGEMM/
  sgemm.cu              benchmark harness: correctness check + timing vs cuBLAS
  01_naive.cuh          kernel 1
```

## Running

Code is edited locally and run on a remote GPU (a RunPod pod) over SSH.

1. Add a host alias to `~/.ssh/config` (use the pod's "SSH over exposed TCP"
   address; the `ssh.runpod.io` proxy cannot run remote commands or rsync):
   ```
   Host runpod
       HostName <IP>
       Port <PORT>
       User root
       IdentityFile ~/.ssh/runpodssh
       IdentitiesOnly yes
   ```
2. Build and benchmark:
   ```sh
   scripts/gpu.sh nvidia-smi
   scripts/gpu.sh 'make -C SGEMM && SGEMM/build/sgemm 1'          # kernel 1, default sizes
   scripts/gpu.sh 'make -C SGEMM && SGEMM/build/sgemm 1 1024 4096' # specific sizes
   scripts/gpu.sh                                                   # shell on the pod
   ```
   `GPU_HOST` and `GPU_DIR` override the SSH alias and remote directory
   (default `runpod` and `/workspace/kernels`).

## The SGEMM harness

`SGEMM/build/sgemm <kernel_id> [size ...]` computes, in row-major order,
`C = alpha * A @ B + beta * C` for square matrices. For each size it:

- runs the chosen kernel and cuBLAS on the same random inputs and checks the
  results match (with a tolerance, since FP32 sums in a different order differ slightly);
- times both with CUDA events, averaged over enough repetitions to take about 0.5s;
- prints GFLOP/s (`2*M*N*K / time`) and the result as a percentage of cuBLAS.

cuBLAS runs with `CUBLAS_PEDANTIC_MATH`, so it uses true FP32 rather than TF32
tensor cores. Kernel 0 is cuBLAS itself.

### Adding a kernel

1. Write `SGEMM/NN_name.cuh` with a launcher
   `run_sgemm_name(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)`.
2. In `SGEMM/sgemm.cu`: `#include` it, add its name to `KERNEL_NAMES`, and add a
   `case` to `run_kernel`.
