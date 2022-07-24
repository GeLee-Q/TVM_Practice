


# MLC: Machine Learning Compiler (TVM)

- 深度学习编译器实践

- 陈天奇主讲的MLC课程:
    - https://mlc.ai/summer22-zh/
    - https://github.com/mlc-ai/mlc-zh


## Chapter 3 End to End Model Execution

### TVM（End_to_End Model）.ipynb
- 基于计算图进行优化，尽量将低层次代码抽象成计算图模式（吸取TF经验）
- 利用 `call_tir` 和 `dataflow block`
  - `call_tir`  完成对低层次函数的封装，构造成计算图模式
  - `dataflow block` 确定计算图优化区域
- IRModule 允许注册现有库函数，并且可以和自己写的`TensorIR`使用