#ifndef DEVICE_ARRAY_CUH
#define DEVICE_ARRAY_CUH

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>

#define gpuErrorCheck( ans ) { gpuAssert( (ans), __FILE__, __LINE__ ); }
inline void gpuAssert( cudaError_t code, char* file, int line, bool abort=true )
{
	if ( cudaSuccess != code )
	{
		fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if ( abort ) exit(code);
	}
}

/**
* DeviceArray is a RAII class to contain CUDA arrays.
*/
template <typename ValueType>
class DeviceArray {
public:
	explicit DeviceArray(unsigned int N): m_ptr(0), m_size(N)
	{
		gpuErrorCheck( cudaMalloc( (void**)&this->m_ptr, N*sizeof(ValueType) ) );
	}

	ValueType* GetPtr() const { return m_ptr; }

	/**
	* Initialize uses cudaMemset that can only assign ints.
	*/
	void Initialize( int val )
	{
		gpuErrorCheck( cudaMemset( this->m_ptr, val, m_size*sizeof(ValueType) ) );
	}

	void copyHostToDevice( const ValueType* const other_array )
	{
		gpuErrorCheck( cudaMemcpy( this->m_ptr, other_array, m_size*sizeof(ValueType), cudaMemcpyHostToDevice ) );
	}

	void copyDeviceToHost( ValueType* other_array )
	{
		gpuErrorCheck( cudaMemcpy( other_array, this->m_ptr, m_size*sizeof(ValueType), cudaMemcpyDeviceToHost ) );
	}

	~DeviceArray()
	{
		gpuErrorCheck( cudaFree(m_ptr) );
	}

private:
	//forbid copy and assignment
	DeviceArray( const DeviceArray& );
	DeviceArray& operator=( const DeviceArray& );

	ValueType* m_ptr;
	unsigned int m_size;
};
#endif