#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Every CUDA API call returns an error code. Kernel launches don't return one,
// so after a launch call CUDA_CHECK(cudaGetLastError()) to catch bad configs.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n  in: %s\n", __FILE__,    \
                    __LINE__, cudaGetErrorString(err_), #call);                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d: %s\n  in: %s\n", __FILE__,  \
                    __LINE__, cublasGetStatusString(st_), #call);               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Integer ceil(a / b): how many blocks of size b are needed to cover a items.
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
