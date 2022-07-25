
- [MLC: Machine Learning Compiler (TVM)](#mlc-machine-learning-compiler-tvm)
  - [Chapter 2 TensorIR](#chapter-2-tensorir)
    - [TensorIR 基础概念](#tensorir-基础概念)
    - [TensorIR_Practice.ipynb](#tensorir_practiceipynb)
      - [TVM 广播加法](#tvm-广播加法)
      - [TVM 二维卷积](#tvm-二维卷积)
      - [TVM 变换批量矩阵乘法](#tvm-变换批量矩阵乘法)
  - [Chapter 3 IRModule](#chapter-3-irmodule)
    - [IRModule.ipynb](#irmoduleipynb)

# MLC: Machine Learning Compiler (TVM)

- 深度学习编译器实践

- 陈天奇主讲的MLC课程:
    - https://mlc.ai/summer22-zh/
    - https://github.com/mlc-ai/mlc-zh


## Chapter 2 TensorIR

### TensorIR 基础概念

两种构建TensorIR的方法：
- TVMSript
- Tensor Expression
   
TensorIR抽象
- 函数参数 与 缓冲区 `T.buffer[(dim0, dim1), "dtype"]`
- For 循环迭代      `for i, j, l in T.grid(128, 128, 128)`
- 计算块    
  - `T.block("Buffer")`
  -  块轴的属性：spaital， reduce
  -  块轴绑定的语法糖 `vi, vj, vk = T.axis.remap("SSR", [i, j, k])`
- 函数属性 和 装饰器
  - ` T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})`
  - `"global_symbol"` 对应函数名
  -   `"tir.noalias"` 表示缓冲区的存储器不重叠


### TensorIR_Practice.ipynb
#### TVM 广播加法

```python
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A:T.Buffer[(4,4), "int64"],
          B:T.Buffer[(4,), "int64"],
          C:T.Buffer[(4,4), "int64"]):
    T.func_attr({"global_symbol": "add", "tir.noalias": True})
    for i, j in T.grid(4,4):
        with T.block("C"):
          vi = T.axis.spatial(4, i)
          vj = T.axis.spatial(4, j)
          C[vi, vj] = A[vi, vj] + B[vj]
```

#### TVM 二维卷积
- 目前推荐TensorIR的`T.Buffer[]` 的参数 `shape` 写 `constant`(from 冯思远)


```python
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)
```
- 先写出numpy版本，更容易去参照着优化
```python
def convol_1(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    for i in range(1): 
        for j in range(2):
          for k in range(6):
            for l in range(6):
              for m in range(3):
                for n in range(3):
                  C[i, j, k ,l] += A[i, i, k + m, l + n] * B[j, i, m, n]

```

```python

@tvm.script.ir_module
class MyConv:
  @T.prim_func
  def conv(A:T.Buffer[(1, 1, 8, 8),"int64"],
        B:T.Buffer[(2, 1, 3, 3),"int64"],
        C:T.Buffer[(1, 2, 6, 6),"int64"]):
    T.func_attr({"global_symbol": "conv", "tir.noalias": True})
    for i, j, k, l, m, n in T.grid(1, 2, 6, 6, 3, 3):
      with T.block("C"):
        vi, vj, vk, vl, vm, vn = T.axis.remap("SSSSRR",[i, j, k, l, m, n])
        with T.init():
          C[vi, vj, vk, vl]= T.int64(0)
        C[vi, vj, vk, vl] = C[vi, vj, vk, vl] + A[vi, vi, vk + vm, vl + vn] *  B[vj, vi, vm, vn]
```

#### TVM 变换批量矩阵乘法
>bmm_relu
- numpy版本
```python
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)
```
- TVM 版本
```python
@tvm.script.ir_module
class MyBmmRelu:
  @T.prim_func
  def bmm_relu(A: T.Buffer[(16, 128, 128), "float32"],
               B: T.Buffer[(16, 128, 128), "float32"],
               C: T.Buffer[(16, 128, 128), "float32"]):
    T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
    Y = T.alloc_buffer((16, 128, 128))
    for n ,i, j, k in T.grid(16, 128, 128, 128):
      with T.block("Y"):
        vn, vi, vj ,vk = T.axis.remap("SSSR", [n, i, j, k])
        with T.init():
            Y[vn, vi, vj] = T.float32(0)
        Y[vn, vi, vj] = Y[vn, vi, vj] + A[vn, vi ,vk] * B[vn, vk, vj]
    
    for n, i, j in T.grid(16, 128, 128):
      with T.block("C"):
        vn, vi ,vj = T.axis.remap("SSS", [n, i, j])
        C[vn, vi, vj]  = T.max(Y[vn, vi, vj], T.float32(0))
```
- 变换TensorRT
```python
sch = tvm.tir.Schedule(MyBmmRelu)
# Step 1. Get blocks 首先对Y进行拆解
Y = sch.get_block("Y", func_name="bmm_relu")
# Step 2. Get loops
n, i, j, k = sch.get_loops(Y)
# Step 3. Organize the loops 
j0, j1 = sch.split(j, factors = [None, 8])
sch.reorder(n, i, j0, k, j1)
n, i, j0, k, j1 = sch.get_loops(Y) 
sch.parallel(n)
C = sch.get_block("C", "bmm_relu")
sch.reverse_compute_at(C, j0)
# Step 4. decompose reduction 将初始化与规约分开
sch.decompose_reduction(Y, k)
# Step 5. vectorize / parallel / unroll
Y_init = sch.get_block("Y_init", "bmm_relu")
ax0_init = sch.get_loops(Y_init)
sch.vectorize(ax0_init[3])

C = sch.get_block("C", "bmm_relu")
ax0 = sch.get_loops(C)
sch.vectorize(ax0[3])

k0, k1 = sch.split(k, factors = [None, 4])
sch.unroll(k1)

```

```python
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer[(16, 128, 128), "float32"], B: T.Buffer[(16, 128, 128), "float32"], C: T.Buffer[(16, 128, 128), "float32"]) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16): # n = i0
            for i1, i2_0 in T.grid(128, 16): # i1 = i , j0 = i2_0
                #初始化Y 
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                #计算Y
                for ax1_0 in T.serial(32):  # [ax1_0,ax1_1] =[ 32 ,4] k 
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8): # j1 = ax0
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                #Relu计算
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))
```

- 实验结果
![res of transform](figs/hw03.png)

## Chapter 3 IRModule

### IRModule.ipynb
- 基于计算图进行优化，尽量将低层次代码抽象成计算图模式（吸取TF经验）
- 利用 `call_tir` 和 `dataflow block`
  - `call_tir`  完成对低层次函数的封装，构造成计算图模式
  - `dataflow block` 确定计算图优化区域
- IRModule 允许注册现有库函数，并且可以和自己写的`TensorIR`使用

