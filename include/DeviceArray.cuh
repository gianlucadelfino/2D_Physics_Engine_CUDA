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
 * DeviceArray is a RAII class to contain CUDA arrays.
 */
template <typename ValueType>
class DeviceArray
{
public:
  explicit DeviceArray(unsigned int N) : _ptr(0), _size(N)
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

  ~DeviceArray() { gpuErrorCheck(cudaFree(_ptr)); }

private:
  // forbid copy and assignment
  DeviceArray(const DeviceArray&);
  DeviceArray& operator=(const DeviceArray&);

  ValueType* _ptr;
  unsigned int _size;
};
#endif
