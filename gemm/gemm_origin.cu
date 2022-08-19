#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

#include <omp.h>
#include <Eigen/Core>
#include <ctime>
#include "tensor2d.cuh"



__global__ void gemmKernel(const float *A, const float *B, float *C,
                           float alpha, float beta, unsigned M, unsigned N,
                           unsigned K) {
    unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
    Tensor2D<const float> tensorA{A, M, K};
    Tensor2D<const float> tensorB{B, K, N};
    Tensor2D<float> tensorC{C, M, N};
    if(!tensorC.validOffset(m, n)) return;

    float c = 0;
    for(unsigned k = 0; k < K ; k++){
        c += tensorA(m, k) * tensorB(k, n);
    }
    c *= alpha;
    
    float ans = c;
    if(beta != 0){
      ans = ans + tensorC(m, n) * beta;
    }
    tensorC(m, n) = ans;
}

// 启动核函数
void gemmNaive(const float *A, const float *B, float *C, float alpha,
               float beta, unsigned M, unsigned N, unsigned K) {
    dim3 block(32, 32);
    dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);

    gemmKernel<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

int main() {
    int gpu_rank = 0;
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, gpu_rank);
    cudaSetDevice(gpu_rank);
    printf("GPU %s status: ", deviceProp.name);
    double boostFrequency = deviceProp.clockRate / 1e6;
    int fp32CoresNum = 640;
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
    gemmNaive(deviceAPrt, deviceBPtr, deviceCPtr, alpha, beta, M, N, K);
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