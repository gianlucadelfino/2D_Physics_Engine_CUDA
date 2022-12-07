#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>

#define gpuErrorCheck(ans)                                                                         \
  {                                                                                                \
    gpu_assert((ans), __FILE__, __LINE__);                                                          \
  }
inline void gpu_assert(cudaError_t code, const char* file, int line, bool abort = true)
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
template <typename value_type>
class device_array
{
public:
  explicit device_array(unsigned int N) : _ptr(0), _size(N)
  {
    gpuErrorCheck(cudaMalloc((void**)&_ptr, N * sizeof(value_type)));
  }

  value_type* get() const { return _ptr; }

  /**
   * initialize uses cudaMemset that can only assign ints.
   */
  void initialize(int val)
  {
    gpuErrorCheck(cudaMemset(_ptr, val, _size * sizeof(value_type)));
  }

  void copy_to_device(const value_type* const other_array)
  {
    gpuErrorCheck(
        cudaMemcpy(_ptr, other_array, _size * sizeof(value_type), cudaMemcpyHostToDevice));
  }

  void copy_to_host(value_type* other_array)
  {
    gpuErrorCheck(
        cudaMemcpy(other_array, _ptr, _size * sizeof(value_type), cudaMemcpyDeviceToHost));
  }

  ~device_array() { gpuErrorCheck(cudaFree(_ptr)); }

private:
  // forbid copy and assignment
  device_array(const device_array&);
  device_array& operator=(const device_array&);

  value_type* _ptr;
  unsigned int _size;
};
#endif
