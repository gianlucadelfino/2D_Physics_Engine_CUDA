#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda.h"
#include "cuda_runtime.h"

/**
* CUDA_utils is a namespace containing utility function related to the CUDA runtime
*/
namespace CUDA_utils
{
	/**
	* CheckCUDACompatibleDevice returns true if a CUDA compatible device was found.
	*/
	bool CheckCUDACompatibleDevice()
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
}

#endif