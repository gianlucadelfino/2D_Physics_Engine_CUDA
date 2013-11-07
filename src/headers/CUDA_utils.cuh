#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda.h"
#include "cuda_runtime.h"

namespace CUDA_utils
{
	bool CheckCUDACompatibleDevice()
	{
		bool CUDA_compatible_device_found = true;
		//check CUDA Availability
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		if ( cudaSuccess != cudaGetDeviceCount(&deviceCount) || deviceCount == 0 )
		{
			CUDA_compatible_device_found = false;
		}
		return CUDA_compatible_device_found;
	}
}

#endif