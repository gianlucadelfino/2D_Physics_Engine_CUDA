#include "cuda_runtime.h"

#include "cuda_utils.cuh"
#include <cstdio>

bool cuda_utils::is_cuda_device_available()
{
    bool CUDA_compatible_device_found = true;
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if ( cudaSuccess != cudaGetDeviceCount(&deviceCount) || deviceCount == 0 )
    {
        CUDA_compatible_device_found = false;
    }
    return CUDA_compatible_device_found;
}
