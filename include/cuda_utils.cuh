#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

/**
* cuda_utils is a namespace containing utility function related to the CUDA runtime
*/
namespace cuda_utils
{
    /**
    * is_cuda_device_available returns true if a CUDA compatible device was found.
    */
    bool is_cuda_device_available();
}

#endif
