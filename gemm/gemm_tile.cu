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
获取 FP32 的核心数
 */
int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major) {
    case 2:  // Fermi
      if (devProp.minor == 1)
        cores = mp * 48;
      else
        cores = mp * 32;
      break;
    case 3:  // Kepler
      cores = mp * 192;
      break;
    case 5:  // Maxwell
      cores = mp * 128;
      break;
    case 6:  // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2))
        cores = mp * 128;
      else if (devProp.minor == 0)
        cores = mp * 64;
      else
        throw std::runtime_error("Unknown device type");
      break;
    case 7:  // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5))
        cores = mp * 64;
      else
        throw std::runtime_error("Unknown device type");
      break;
    case 8:  // Ampere
      if (devProp.minor == 0)
        cores = mp * 64;
      else if (devProp.minor == 6)
        cores = mp * 128;
      else
        throw std::runtime_error("Unknown device type");
      break;
    default:
      throw std::runtime_error("Unknown device type");
  }
  return cores;
}



template <typename LayoutTile, typename LayoutBlock, typename LayoutThread>
__global__ void gemmKernel(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
    constexpr unsigned ratio = sizeof(float_4) / sizeof(float);
    unsigned int m = threadIdx.x * LayoutThread::m + LayoutTile::m * blockIdx.x;
    unsigned int n = threadIdx.y * LayoutThread::n + LayoutTile::n * blockIdx.y;
    Tensor2D<const float> pA{A, M, K};
    pA.addOffset(m, 0);
    Tensor2D<const float_4> pB{B, K, N / ratio};
    pB.addOffset(0, n / ratio);
    Tensor2D<float_4> pC{C, M, N / ratio};
    pC.addOffset(m, n / ratio);

    /* 
    之前的每个线程从处理每个数据块，变为多个数据块，需要额外的变量进行记录
    iterationA 处理 tile 行方向的迭代的次数， intervalA 表示的是tile子矩阵在行方向上间隔;
    iterationB  处理 tile 列方向的迭代的次数， intervalB 表示的是tile子矩阵在列方向上间隔;
     */
    const unsigned iterationA = LayoutTile::m / LayoutBlock::m / LayoutThread::m;
    const unsigned iterationB = LayoutTile::n / LayoutBlock::n / LayoutThread::n;
    const unsigned intervalA = LayoutTile::m / iterationA;
    const unsigned intervalB = LayoutTile::n / iterationB;

    bool validLoadTileA[iterationA];
    bool validLoadTileB[iterationB];

#pragma unroll
    for (unsigned i = 0; i < iterationA; ++i) {
        validLoadTileA[i] = pA.validRowOffset(i * intervalA);
    }

#pragma unroll
    for (unsigned i = 0; i < iterationB; ++i) {
        validLoadTileB[i] = pB.validColOffset(i * intervalB / ratio);
    }

    /* 
    数据的读取和累加计算需要增加循环
     */
    constexpr float_4 float4Zero{0.f, 0.f, 0.f, 0.f};

    float_4 c[iterationA][iterationB][4];
    memset(c, 0, sizeof(c));
    for (unsigned k = 0; k < K; ++k) {
#pragma unroll
        for (unsigned iterA = 0; iterA < iterationA; ++iterA) {
        float_4 fragmentA{};
        validLoadTileA[iterA] &= pA.validColOffset(k);
#pragma unroll
            for (unsigned i = 0; i < ratio; ++i) {
                // fragmentA[i] = validLoadTileA[i] ? pA(i + iterA * intervalA, k) : 0.0f;
                fragmentA[i] = validLoadTileA[i] ? pA(i + iterA * intervalA, k) : 0;
            }
#pragma unroll
            for (unsigned iterB = 0; iterB < iterationB; ++iterB) {
                validLoadTileB[iterB] &= pB.validRowOffset(k);
                float_4 fragmentB = validLoadTileB[iterB]
                                                ? pB(k, iterB * intervalB / ratio)
                                                : float4Zero;

#pragma unroll
                for (unsigned i = 0; i < ratio; ++i) {
                c[iterA][iterB][i] = c[iterA][iterB][i] + fragmentB * fragmentA[i];
                }
            }
        }
    }

#pragma unroll
    for (auto &termA : c) {
#pragma unroll
        for (auto &termB : termA) {
#pragma unroll
            for (auto &term : termB) {
                term = term * alpha;
            }
        }
    }

#pragma unroll
    for (unsigned iterA = 0; iterA < iterationA; ++iterA) {
#pragma unroll
        for (unsigned iterB = 0; iterB < iterationB; ++iterB) {
#pragma unroll
            for (unsigned i = 0; i < ratio; ++i) {
                float_4 result{c[iterA][iterB][i]};
                if (beta != 0) {
                result = result +
                        pC(i + iterA * intervalA, iterB * intervalB / ratio) * beta;
                }
                pC(i + iterA * intervalA, iterB * intervalB / ratio) = result;
            }
        }
    }
}


void gemm_tile(const float *deviceAPtr, const float *deviceBPtr,
                float *deviceCPtr, float alpha, float beta, unsigned M,
                unsigned N, unsigned K) {
    using LayoutTile = Layout<128, 128, 16>;
    using LayoutBlock = Layout<16, 16>;
    using LayoutThread = Layout<4, 4>;

    dim3 block(LayoutBlock::m, LayoutBlock::n);
    dim3 grid((M * LayoutBlock::m / LayoutTile::m - 1) / LayoutBlock::m + 1,
              (N * LayoutBlock::n / LayoutTile::n - 1) / LayoutBlock::n + 1);

    gemmKernel<LayoutTile, LayoutBlock, LayoutThread><<<grid, block>>>(
        deviceAPtr, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
}

int main() {
    int gpu_rank = 0;
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, gpu_rank);
    cudaSetDevice(gpu_rank);
    printf("GPU %s status: ", deviceProp.name);
    double boostFrequency = deviceProp.clockRate/ 1e6;
    // int fp32CoresNum = 5120;
    int fp32CoresNum = getSPcores(deviceProp);
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
    gemm_tile(deviceAPrt, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);

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
    printf("Throughput / peak : %.3f %\n", GFLOPS /  peakPerformance * 100);
}