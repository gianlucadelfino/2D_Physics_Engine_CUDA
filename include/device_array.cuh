#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>

#define gpuErrorCheck(ans)                                                                         \
  {                                                                                                \
    gpuAssert((ans), __FILE__, __LINE__);                                                          \
  }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (cudaSuccess != code)
  {
    fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

/**
 * device_array is a RAII class to contain CUDA arrays.
 */
template <typename ValueType>
class device_array
{
public:
  explicit device_array(unsigned int N) : _ptr(0), _size(N)
  {
    gpuErrorCheck(cudaMalloc((void**)&_ptr, N * sizeof(ValueType)));
  }

  ValueType* GetPtr() const { return _ptr; }

  /**
   * Initialize uses cudaMemset that can only assign ints.
   */
  void Initialize(int val)
  {
    gpuErrorCheck(cudaMemset(_ptr, val, _size * sizeof(ValueType)));
  }

  void copyHostToDevice(const ValueType* const other_array)
  {
    gpuErrorCheck(
        cudaMemcpy(_ptr, other_array, _size * sizeof(ValueType), cudaMemcpyHostToDevice));
  }

  void copyDeviceToHost(ValueType* other_array)
  {
    gpuErrorCheck(
        cudaMemcpy(other_array, _ptr, _size * sizeof(ValueType), cudaMemcpyDeviceToHost));
  }

  ~device_array() { gpuErrorCheck(cudaFree(_ptr)); }

private:
  // forbid copy and assignment
  device_array(const device_array&);
  device_array& operator=(const device_array&);

  ValueType* _ptr;
  unsigned int _size;
};
#endif
