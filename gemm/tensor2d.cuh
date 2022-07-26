

/* 
将尺寸信息在编译期确定，减少执行期的额外的数据读取开销
 */
template <int _m, int _n, int _k = 1>
struct Layout {
  static constexpr int m = _m;
  static constexpr int n = _n;
  static constexpr int k = _k;
};


/* 
float_4  堆叠4个 float 到一个缓存行大小
 */
struct __device_builtin__ __builtin_align__(16) float_4 {
  float data[4];

  __host__ __device__ float operator[](unsigned idx) const { return data[idx]; }

  __host__ __device__ float &operator[](unsigned idx) { return data[idx]; }

  __host__ __device__ float_4 operator*(float other) const {
    return float_4{data[0] * other, data[1] * other, data[2] * other,
                  data[3] * other};
  }

  __host__ __device__ float_4 operator+(const float_4 &other) const {
    return float_4{data[0] + other.data[0], data[1] + other.data[1],
                  data[2] + other.data[2], data[3] + other.data[3]};
  }
};

/* 
封装结构，代替指针，避免手动的去做 一维到高维的地址变换
 */
template <typename T>
struct __device_builtin__ Tensor2D {
  T *const __restrict__ ptr;
  const unsigned rows, cols;
  int _rowOffset{0}, _colOffset{0};

  template <typename t>
  __host__ __device__ Tensor2D(t &&ptr, unsigned rows, unsigned cols)
      : ptr{reinterpret_cast<T *>(ptr)}, rows{rows}, cols{cols} {};

  template <typename t = T>
  __host__ __device__ void addOffset(int rowOffset, int colOffset) {
    _rowOffset += rowOffset;
    _colOffset += colOffset * sizeof(t) / sizeof(T);
  }

  __host__ __device__ bool validRowOffset(int rowOffset) const {
    return (_rowOffset + rowOffset) < rows;
  }

  __host__ __device__ bool validColOffset(int colOffset) const {
    return (_colOffset + colOffset) < cols;
  }

  __host__ __device__ bool validOffset(int rowOffset, int colOffset) const {
    return validRowOffset(rowOffset) && validColOffset(colOffset);
  }

  __host__ __device__ T &operator()(int row, int col) const {
    return ptr[_colOffset + col + (row + _rowOffset) * cols];
  }
};