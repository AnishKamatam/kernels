#pragma once
#include "cuda_utils.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // each thread computes one element of C, x is row, y is column
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y; 
    // check if the thread is in the bounds of C
    if (x < M && y < N) {
        // dot product of the row of A and the column of B
        float tmp = 0.0f;
        for (int i = 0; i < K; i++){
            tmp += A[x * K + i] * B[i *N +y];
            //accumulate dot product
        }
        // write the result to C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

inline void run_sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // define the grid size and block size and ensure all of C is covered in the grid
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    // block size is 32x32, each thread computes one element of C, total of 1024 threads per block
    dim3 blockDim(32, 32, 1);
    //launch the kernel
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
