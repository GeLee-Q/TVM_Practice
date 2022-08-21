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
    using LayoutTileT =
        Layout<LayoutTile::m / ratio, LayoutTile::n / ratio,
                          LayoutTile::k / ratio>;
    using LayoutThreadT =
        Layout<LayoutThread::m / ratio, LayoutThread::n / ratio>;
    constexpr unsigned blockSize = LayoutBlock::m * LayoutBlock::n;
    constexpr float_4 float4Zero{0.f, 0.f, 0.f, 0.f};

    __shared__ float_4 tileA[LayoutTile::k][LayoutTileT::m];
    __shared__ float_4 tileB[LayoutTile::k][LayoutTileT::n];

    const unsigned nInTileC = threadIdx.x % LayoutBlock::m;
    const unsigned mInTileC = threadIdx.x / LayoutBlock::m;

    const unsigned kInTileA = threadIdx.x % LayoutTileT::k;
    const unsigned mInTileA = threadIdx.x / LayoutTileT::k;

    const unsigned nInTileB = threadIdx.x % LayoutTileT::n;
    const unsigned kinTileB = threadIdx.x / LayoutTileT::n;

    Tensor2D<const float_4> pA{A, M, K / ratio};
    pA.addOffset(LayoutTile::m * blockIdx.y + mInTileA, kInTileA);
    Tensor2D<const float_4> pB{B, K, N / ratio};
    pB.addOffset(kinTileB,
                LayoutTileT::n * blockIdx.x + nInTileB * LayoutThreadT::n);
    Tensor2D<float_4> pC{C, M, N / ratio};
    pC.addOffset(LayoutTile::m * blockIdx.y + mInTileC * LayoutThread::m,
                LayoutTileT::n * blockIdx.x + nInTileC * LayoutThreadT::n);

    constexpr unsigned tileSizeA = LayoutTile::m * LayoutTile::k;
    constexpr unsigned tileSizeB = LayoutTile::n * LayoutTile::k;
    constexpr unsigned tileIterationsA = tileSizeA / blockSize / ratio;
    constexpr unsigned tileGlobalIntervalA = blockSize / LayoutTileT::k;
    constexpr unsigned tileComputeIterationsA = LayoutTileT::m / LayoutBlock::m;
    constexpr unsigned tileSharedIntervalAT =
        LayoutTileT::m / tileComputeIterationsA;
    constexpr unsigned tileIterationsB = tileSizeB / blockSize / ratio;
    constexpr unsigned tileGlobalIntervalB = blockSize / LayoutTileT::n;
    constexpr unsigned tileComputeIterationsB = LayoutTileT::n / LayoutBlock::n;
    constexpr unsigned tileSharedIntervalBT =
        LayoutTileT::n / tileComputeIterationsB;

    float_4 bufferA[tileIterationsA];
    float_4 bufferB[tileIterationsB];
    bool validLoadTileA[tileIterationsA];
    bool validLoadTileB[tileIterationsB];

#pragma unroll
    for (unsigned i = 0; i < tileIterationsA; ++i) {
        validLoadTileA[i] = pA.validRowOffset(i * tileGlobalIntervalA);
    }

#pragma unroll
    for (unsigned i = 0; i < tileIterationsB; ++i) {
        validLoadTileB[i] = pB.validColOffset(0);
    }

    float_4 c[tileComputeIterationsA * LayoutThread::m]
              [tileComputeIterationsB * LayoutThreadT::n];
    memset(c, 0, sizeof(c));

    float_4 fragmentA[tileComputeIterationsA * LayoutThreadT::m];
    float_4 fragmentB[tileComputeIterationsB * LayoutThreadT::n];

    for (unsigned i = 0; i < K; i += LayoutTile::k) {
#pragma unroll
        for (unsigned j = 0; j < tileIterationsA; ++j) {
            validLoadTileA[j] &= pA.validColOffset(0);
            bufferA[j] =
                validLoadTileA[j] ? pA(j * tileGlobalIntervalA, 0) : float4Zero;
        }

#pragma unroll
        for (unsigned j = 0; j < tileIterationsB; ++j) {
            validLoadTileB[j] &= pB.validRowOffset(j * tileGlobalIntervalB);
            bufferB[j] =
                validLoadTileB[j] ? pB(j * tileGlobalIntervalB, 0) : float4Zero;
        }

        __syncthreads();
#pragma unroll
        for (unsigned a = 0; a < tileIterationsA; ++a) {
#pragma unroll
          for (unsigned j = 0; j < LayoutThread::m; ++j) {
              tileA[kInTileA * ratio + j]
                  [(a * tileGlobalIntervalA + mInTileA) / ratio]
                  [(a * tileGlobalIntervalA + mInTileA) % ratio] = bufferA[a][j];
          }
      }

#pragma unroll
        for (unsigned a = 0; a < tileIterationsB; ++a) {
            tileB[kinTileB + a * tileGlobalIntervalB][nInTileB] = bufferB[a];
        }
        __syncthreads();

#pragma unroll
        for (unsigned j = 0; j < LayoutTile::k; j++) {
#pragma unroll
            for (unsigned a = 0; a < tileComputeIterationsA; ++a) {
                fragmentA[a] = tileA[j][a * tileSharedIntervalAT + mInTileC];
           }
#pragma unroll
            for (unsigned a = 0; a < tileComputeIterationsB; ++a) {
                fragmentB[a] = tileB[j][a * tileSharedIntervalBT + nInTileC];
            }
#pragma unroll
            for (unsigned d = 0; d < tileComputeIterationsA * LayoutThread::m; ++d) {
#pragma unroll
                for (unsigned e = 0; e < tileComputeIterationsB * LayoutThreadT::n; ++e) {
                  c[d][e] =  c[d][e] + fragmentB[e] *
                              fragmentA[d / LayoutThread::m][d % LayoutThread::m];
                }
            }
        }
        pA.addOffset(0, LayoutTileT::k);
        pB.addOffset(LayoutTile::k, 0);
    }

#pragma unroll
    for (auto &a : c) {
#pragma unroll
        for (auto &b : a) {
            b = b * alpha;
        }
    }

#pragma unroll
    for (unsigned i = 0; i < tileComputeIterationsA; ++i) {
#pragma unroll
        for (unsigned a = 0; a < LayoutThread::m; a++) {
          const bool mValid = pC.validRowOffset(a);
#pragma unroll
            for (unsigned b = 0; b < tileComputeIterationsB; b++) {
                const bool nValid = pC.validColOffset(b * tileSharedIntervalBT);
                if (mValid && nValid) {
                    float_4 result{c[a + i * LayoutThread::m][b]};
                    if (beta != 0) {
                        result = result + pC(a, b * tileSharedIntervalBT) * beta;
                    }
                    pC(a, b * tileSharedIntervalBT) = result;
                }
            }
        }
        pC.addOffset(tileSharedIntervalAT * ratio, 0);
    }
}



void gemm_tile(const float *deviceAPtr, const float *deviceBPtr,
                float *deviceCPtr, float alpha, float beta, unsigned M,
                unsigned N, unsigned K) {
    using LayoutTile = Layout<128, 128, 16>;
    using LayoutBlock = Layout<16, 16>;
    using LayoutThread = Layout<4, 4>;

    dim3 block(LayoutBlock::m * LayoutBlock::n);
    dim3 grid((M - 1) / LayoutTile::m + 1, (N - 1) / LayoutTile::n + 1);

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