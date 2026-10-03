// SGEMM benchmark harness.
//
//   ./build/sgemm <kernel_id> [size ...]
//
// For each square size: runs your kernel and cuBLAS on the same random inputs,
// checks your output against cuBLAS, then times both and reports GFLOP/s.
//
// Convention (row-major): C[M][N] = alpha * A[M][K] @ B[K][N] + beta * C[M][N]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_utils.cuh"
#include "01_naive.cuh"

static const char *KERNEL_NAMES[] = {
    "cuBLAS",
    "naive",
};
static const int NUM_KERNELS = sizeof(KERNEL_NAMES) / sizeof(KERNEL_NAMES[0]);

static void run_cublas(cublasHandle_t handle, int M, int N, int K, float alpha,
                       const float *A, const float *B, float beta, float *C) {
    // cuBLAS is column-major. A row-major matrix reinterpreted as column-major
    // is its transpose, so we ask cuBLAS for C^T = B^T @ A^T, which lands in
    // memory exactly as row-major C = A @ B.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             B, N, A, K, &beta, C, N));
}

static void run_kernel(int id, cublasHandle_t handle, int M, int N, int K,
                       float alpha, const float *A, const float *B, float beta,
                       float *C) {
    switch (id) {
    case 0: run_cublas(handle, M, N, K, alpha, A, B, beta, C); break;
    case 1: run_sgemm_naive(M, N, K, alpha, A, B, beta, C); break;
    default: fprintf(stderr, "unknown kernel %d\n", id); exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Average milliseconds per call. GPU work is asynchronous, so we time with CUDA
// events recorded on the stream rather than a CPU clock.
static float time_kernel(int id, cublasHandle_t handle, int M, int N, int K,
                         float alpha, const float *A, const float *B,
                         float beta, float *C) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup (first launches pay one-off costs), and a rough estimate so we can
    // pick a repetition count that runs for ~0.5s total.
    run_kernel(id, handle, M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaEventRecord(start));
    run_kernel(id, handle, M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float est_ms;
    CUDA_CHECK(cudaEventElapsedTime(&est_ms, start, stop));
    int reps = std::clamp((int)(500.0f / std::max(est_ms, 1e-3f)), 3, 200);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; i++)
        run_kernel(id, handle, M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / reps;
}

// FP32 sums in a different order give slightly different answers, so compare
// with a tolerance rather than exact equality.
static bool verify(const std::vector<float> &out, const std::vector<float> &ref,
                   int N) {
    for (size_t i = 0; i < out.size(); i++) {
        float diff = std::fabs(out[i] - ref[i]);
        if (!(diff <= 1e-2f + 1e-3f * std::fabs(ref[i]))) {  // also catches NaN
            fprintf(stderr, "  mismatch at C[%zu][%zu]: got %f, expected %f\n",
                    i / N, i % N, out[i], ref[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <kernel_id> [size ...]\n", argv[0]);
        for (int i = 0; i < NUM_KERNELS; i++)
            fprintf(stderr, "  %d: %s\n", i, KERNEL_NAMES[i]);
        return 1;
    }
    int kernel_id = atoi(argv[1]);
    if (kernel_id < 0 || kernel_id >= NUM_KERNELS) {
        fprintf(stderr, "unknown kernel %d\n", kernel_id);
        return 1;
    }
    std::vector<int> sizes;
    for (int i = 2; i < argc; i++) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {128, 256, 512, 1024, 2048, 4096};

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d, %d SMs, %.1f GB)\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1073741824.0);
    printf("Kernel %d: %s\n\n", kernel_id, KERNEL_NAMES[kernel_id]);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // Keep cuBLAS in true FP32 (no TF32 tensor cores) so the comparison is fair.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    printf("%6s %12s %12s %12s %10s %6s\n", "size", "time (ms)", "GFLOP/s",
           "cuBLAS", "% cuBLAS", "check");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const float alpha = 1.0f, beta = 0.5f;  // nonzero beta tests the C read path

    for (int n : sizes) {
        int M = n, N = n, K = n;
        std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
        for (auto &x : hA) x = dist(rng);
        for (auto &x : hB) x = dist(rng);
        for (auto &x : hC) x = dist(rng);

        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dRef, hC.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dRef, hC.data(), hC.size() * sizeof(float), cudaMemcpyHostToDevice));

        // Correctness: one run of each from identical starting C.
        run_kernel(0, handle, M, N, K, alpha, dA, dB, beta, dRef);
        run_kernel(kernel_id, handle, M, N, K, alpha, dA, dB, beta, dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> out(hC.size()), ref(hC.size());
        CUDA_CHECK(cudaMemcpy(out.data(), dC, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ref.data(), dRef, ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = verify(out, ref, N);

        // A GEMM does M*N*K multiply-adds = 2*M*N*K FLOPs.
        double flops = 2.0 * M * N * K;
        float ref_ms = time_kernel(0, handle, M, N, K, alpha, dA, dB, beta, dRef);
        double ref_gflops = flops / (ref_ms * 1e6);
        if (ok) {
            float ms = time_kernel(kernel_id, handle, M, N, K, alpha, dA, dB, beta, dC);
            double gflops = flops / (ms * 1e6);
            printf("%6d %12.3f %12.1f %12.1f %9.1f%% %6s\n", n, ms, gflops,
                   ref_gflops, 100.0 * gflops / ref_gflops, "PASS");
        } else {
            printf("%6d %12s %12s %12.1f %10s %6s\n", n, "-", "-", ref_gflops,
                   "-", "FAIL");
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dRef));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
