+++
title = "GEMM optimization with CUDA"
date = "2025-07-31T18:42:52+02:00"
#dateFormat = "2006-01-02" # This value can be configured for per-post date formatting
author = "thom"
authorTwitter = "" #do not include @
cover = ""
tags = ["cuda", "gpu"]
keywords = ["", ""]
description = ""
showFullContent = false
readingTime = false
hideComments = false
+++


Recently I wanted to learn more about CUDA programming and GPU optimizations in general, so I started this small exploration project of iteratively optimizing GEMM starting from a probably terrible implementation I came up with, up to SOTA standards.
<!--more-->

## GEMM
GEMM (GEneral Matrix Multiplication) is the operation $C = \alpha AB + \beta C$, with A and B input matrices of size (M x K) and (K x N), C being a pre-existing matrix (M x N) that'll be overwritten by the operation, $\alpha$ and $\beta$ being two input scalars. Matrix multiplications are present in a lot of neural networks operations, that's why it's interesting to try and optimize it.

{{< image src="matrix_1.png" alt="" width="300" >}}

## 0.1 - CPU implementation
As a first step, I implemented GEMM on the CPU, a simple C++ sequential program, without trying to optimize it.

Just some memory allocation and randomly filling the matrices, then give it to this kernel:
```cpp
void cpu_mat_mul_kernel(float ** A, float ** B, float ** C, float alpha, float beta, int C_rows, int C_cols, int A_cols) {
    for (int row = 0; row<C_rows; row++) {
        for (int col = 0; col<C_cols; col++) {
            float sum = 0;
            for (int k = 0; k<A_cols; k++) {
                sum += A[row][k] * B[k][col];
            }
            C[row][col] = alpha * sum + beta*C[row][col];
        }
    }
}
```

Very straightforward, and probably terrible performance. But now that I'm talking about it, how am I going to assess performance ?

## Assessing performance
The resulting matrix of a GEMM operation has M x N values, each of them was computed by a dot product of a K-elements vector. Each operation of the dot product is made of a multiplication and an addition, so its a total of 2 * M * N * K FLOP (ignoring $\alpha$ and $\beta$ as they are negligible if matrices are large enough). 

Then we divide the number of FLOP by the time, to get a FLOP/s performance score.

Using this method, my CPU implementation achieves around 1.35 GFLOP/s.

I know it has to be terrible, I didn't optimize anything, but I need some comparison to be sure

## 0.2 - Eigen library
Eigen ([https://eigen.tuxfamily.org/](https://eigen.tuxfamily.org/)) is a C++ library for linear algebra, so I tried the same kernel using their methods / data structures to compare.

```cpp
void eigen_mat_mul_kernel(Eigen::MatrixXf *A, Eigen::MatrixXf * B, Eigen::MatrixXf *C, 
                          float alpha, float beta, int C_rows, int C_cols, int A_cols) {

    *C = alpha * (*A * *B) + beta * *C;
}
```

=> achieves around 30 GFLOP/s, so about 22x more than my implementation, as expected it is a specialized linear algebra library (coughing baby vs hydrogen bomb ...)

Would be interesting to dig into Eigen source code to understand how it works (see if it is multi-threaded or sequential for example), but I'll leave it to another day, today is for CUDA !

## Reference: cuBLAS
To be able to do some comparison, we need a reference: we're going the GEMM method from cuBLAS library (which is a NVIDIA BLAS implementation in CUDA).


## 1 - First CUDA implementation

Very basic, creating one CUDA thread for each cell of the resulting C matrix, with a block dimension of 16x16 threads (kinda arbitrary, i had this number in mind),
and a grid containing as many blocks as needed to cover the whole matrix
```cpp
dim3 threadsPerBlock(16,16); // 256
dim3 nbBlocks(  (N+threadsPerBlock.x-1) / threadsPerBlock.x, 
                (M+threadsPerBlock.y-1) / threadsPerBlock.y);
```



```cpp
__global__ void cuda_1_mat_mul_kernel(float * A, float * B, float * C, float alpha, float beta, int C_rows, int C_cols, int A_cols) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y + blockDim.y;
    if ( (row < C_rows) && (col < C_cols)) {
        float sum = 0;
        for (int k = 0; k<A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * C_cols + col];
        }
        C[row*C_cols + col] = alpha * sum + beta*C[row * C_cols + col];
    }
}
```


{{< image src="benchmark_1.png" width="" >}}

=> achieving ~100 GFLOP/s, pretty good speedup compared to the first CPU implementation, as expected and kinda obvious since we went from sequential to parallel, but as we see on the graph it's only the beginning ! Let's see what kind of optimizations we can implement.


## 2 - Memory coalescing

After we copied data from host to device, our matrix is in the global memory.

{{< image src="memory_layout.png" width="500" >}}
^ image above from [modal.com](https://modal.com/gpu-glossary/device-software/cuda-programming-model)

When a thread within a warp accesses global memory it **coalesces** the memory accesses of threads within this warp into one (or more if needed) memory transaction when possible.
For this to possible, consecutive threads within a warp must access consecutive memory addresses.

{{< image src="coalescing.png"  width="400" >}}

So we want consecutive threads to be on the same row, consecutive columns of the result matrix.

Actually it's not the case in our kernel, because in CUDA thread indexing, for some reason, x represents the columns and y the row. 

So the first thread has coords (0, 0), and the second (1, 0). Therefore, we need to swap `row` and `col` at the beginning of our kernel.
```cpp
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y + blockDim.y;
```

{{< image src="benchmark_2.png" width="" >}}


This tiny modification drastically changes how data is accessed during the kernel execution, and brings us to ~430 GFLOP/s.


## 3 - Shared memory / tiled matrix multiplication

The next step to optimize our kernel is taking advantage of Shared Memory. 

Shared Memory is a on-chip memory area allocated per thread block, and as its name suggests, threads can access data in shared memory loaded from global memory by other threads in the same block. It is much faster than global memory (100x lower latency), and we want to take advantage of it.

However, it is also much smaller, so we cannot load all the rows / columns a thread block would need at once, this wouldn't be suitable for big matrices. Therefore, we use tiling, computing sub-matrices of the result matrix C.

One tile after another, we bring the corresponding elements from matrices A and B into shared memory, compute partial dot vector, then proceed to the next tile. Once we did all necessary tiles for this thread block, we can compute the final result value for these cells of the resulting matrix C.

{{< image src="tiling.png"  width="700" >}}

One tile represents one thread block.

Here for 4x3 and 3x4 matrices, each cell is a dot product of 3 elements and as we use 2x2 tiles, we'll need two iterations of this to compute for each cell. We lose a bit of parallelism here, as we need to wait for a tiling phase to be completed before continuing, but the speedup gained from Shared Memory still makes it better than having all threads read from global memory while they could (literally) share memory.

{{< image src="benchmark_3.png" width="" >}}


This kernel version brings us to around 800 GFLOP/s, but we are still far from cuBLAS (notice that the y-scale is logarithmic on the graph).

## To be continued

There is still much to do in this project, the optimizations implemented here are very basic, but I figured I should dive deeper into CUDA to really understand all of this, so I'll come back later !

