#include "cuda.h"
#include "cuda_runtime.h"

#include "CUDA_utils.cuh"
#include <cstdio>

bool CUDA_utils::IsCUDACompatibleDeviceAvailable()
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