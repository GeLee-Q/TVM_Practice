#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

#include <omp.h>
#include <Eigen/Core>
#include <ctime>
#include "tensor2d.cuh"


/* 
优化思路，增加计算强度，利用延迟隐藏

利用cache line的大小为128字节，将四个float堆叠为float_4字节，增加缓存行吞吐的利用率

每个线程负责 4 x 4 的矩阵块

利用 TensorA.addOffset 线程定位到要处理的块
 */

__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
    constexpr unsigned ratio = sizeof(float_4) / sizeof(float);
    unsigned int m = (threadIdx.x + blockDim.x * blockIdx.x) * ratio;
    unsigned int n = (threadIdx.y + blockDim.y * blockIdx.y) * ratio;
    Tensor2D<const float> pA{A, M, K};
    pA.addOffset(m, 0);
    Tensor2D<const float_4> pB{B, K, N / ratio};
    pB.addOffset(0, n / ratio);
    Tensor2D<float_4> pC{C, M, N / ratio};
    pC.addOffset(m, n / ratio);
    if (!pC.validOffset(0, 0)) return;

    float_4 c[4];
    memset(c, 0, sizeof(c));
    for (unsigned k = 0; k < K; ++k) {
      float_4 fragmentA{};
#pragma unroll
    for (unsigned i = 0; i < ratio; ++i) {
        fragmentA[i] = pA(i, k);
      }
      float_4 fragmentB = pB(k, 0);

#pragma unroll
    for (unsigned i = 0; i < ratio; ++i) {
        c[i] = c[i] + fragmentB * fragmentA[i];
      }
    }

#pragma unroll
    for (auto &a : c) {
        a = a * alpha;
    }

#pragma unroll
    for (unsigned i = 0; i < ratio; ++i) {
        float_4 result = c[i];
        if (beta != 0) {
          result = c[i] + pC(i, 0) * beta;
        }
        pC(i, 0) = result;
    }
}


void gemm_128(const float *deviceAPtr, const float *deviceBPtr,
                float *deviceCPtr, float alpha, float beta, unsigned M,
                unsigned N, unsigned K) {
  dim3 block(16, 16);
  dim3 grid((M / 4 - 1) / block.x + 1, (N / 4 - 1) / block.y + 1);

  gemmKernel<<<grid, block>>>(deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta,
                              M, N, K);
}

int main() {
    int gpu_rank = 0;
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, gpu_rank);
    cudaSetDevice(gpu_rank);
    printf("GPU %s status: ", deviceProp.name);
    double boostFrequency = deviceProp.clockRate / 1e6;
    int fp32CoresNum = 5120;
    double peakPerformance = boostFrequency * fp32CoresNum * 2;
    printf(
        "clock rate %.3f GHz, FP32 cores num %d, FP32 peak throughput %.3f "
        "GFLOPS\n",
        boostFrequency, fp32CoresNum, peakPerformance);
    
    // 启用openmp 使用cpu的多线程计算一般矩阵乘法
    omp_set_num_threads(omp_get_num_procs());

    // 生成数据并执行
    unsigned M = 1024, N = 1024, K = 1024;
    float alpha = 1., beta = 0.;
    float *deviceAPrt, *deviceBPtr, *deviceCPtr;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A{M, K},
        B{K, N}, C{M, N};


    A.setRandom();
    B.setRandom();
    C.setRandom();
    cudaMalloc(&deviceAPrt, M * K * sizeof(float));
    cudaMemcpy(deviceAPrt, A.data(), M * K * sizeof(float),
                cudaMemcpyHostToDevice);
    cudaMalloc(&deviceBPtr, K * N * sizeof(float));
    cudaMemcpy(deviceBPtr, B.data(), K * N * sizeof(float),
                cudaMemcpyHostToDevice);
    cudaMalloc(&deviceCPtr, M * N * sizeof(float));
    cudaMemcpy(deviceCPtr, C.data(), M * N * sizeof(float),
                cudaMemcpyHostToDevice);

    //cuda_event 记录GPU端程序事务消耗的时间
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    // 调用GPU计算
    gemm_128(deviceAPrt, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("GPU use: %.3f(ms)\n", milliseconds);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(startEvent);

    // cpu的计算时间
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        hostResult{M, N}, deviceResult{M, N};
    clock_t begin, end;
    begin = clock();
    hostResult = alpha * (A * B) + beta * C;
    end = clock();
    printf("CPU use: %.3f(ms)\n", double(end - begin) / CLOCKS_PER_SEC * 1e3);

    // 检验二者精度差
    cudaMemcpy(deviceResult.data(), deviceCPtr, M * N * sizeof(float),
                cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> diffArray =
        (hostResult - deviceResult).array().abs();
    printf("Max Error: %f\n", diffArray.maxCoeff());


    // GPU gemm 计算峰值
    double GFLOPS = 2 * 1e-9 * M * N * K / (milliseconds * 1e-3);
    printf("GPU Throughput: %.3f GFLOPS\n", GFLOPS);
}