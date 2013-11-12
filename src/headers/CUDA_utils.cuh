#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

/**
* CUDA_utils is a namespace containing utility function related to the CUDA runtime
*/
namespace CUDA_utils
{
	/**
	* IsCUDACompatibleDeviceAvailable returns true if a CUDA compatible device was found.
	*/
	bool IsCUDACompatibleDeviceAvailable();
}

#endif