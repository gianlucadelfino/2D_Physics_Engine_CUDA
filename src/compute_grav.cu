#include "CUDA.h"
#include "CUDA_runtime.h"
#include "device_launch_parameters.h"
#include "DeviceArray.cuh"

#include "assert.h"

#define BLOCK_SIZE 256
#define NEWTON 0.2f
#define MIN_DIST 50.0f

/******************************
Data distribution on GRID:
- The following represents the time progression of the 1 dimensional grid that the
compute_gravs kernel works on.
- We separate the N stars in 1D tiles, and each thread works out the gravitational
force acting on a star.
- We have to load the positions of the other stars into shared memory
in chuncks to make it fit. Each orizontal line represent the moment when we load the
new stars' positions

------------------------------- <-Start point
| xx | xx | xx | xx | xx | x  |
| xx | xx | xx | xx | xx | x  |
------------------------------- <-New tile, load new positions!
| xx | xx | xx | xx | xx | x  |
| xx | xx | xx | xx | xx | x  |		|
-------------------------------		| Time direction
| xx | xx | xx | xx | xx | x  |		|
| xx | xx | xx | xx | xx | x  |		V
-------------------------------
| xx | xx | xx | xx | xx | x  |
| xx | xx | xx | xx | xx | x  |
-------------------------------
| xx | xx | xx | xx | xx | x  |
|    |    |    |    |    |    |
------------------------------- <- End
******************************/

__global__ void compute_gravs( float2* d_pos, float* d_masses, float2* d_gravs, int num_stars )
{
	//local coordinate on the block
	int local_i = threadIdx.x;

	//global coordinate on the grid
	int global_i = blockIdx.x*blockDim.x + local_i;

	//this is safe from G80, since the device shuold keep track of the active threads and not wait for
	//the inactive ones in case of __syncthread
	if( global_i >= num_stars ) return;

	extern __shared__ float3 s_pos_mass[]; //we save the mass in the third entry (x,y,mass)

	float2 total_acc;
	total_acc.x = 0.0f;
	total_acc.y = 0.0f;

	float2 cur_pos = d_pos[global_i];

	for( unsigned int tile = 0; tile < gridDim.x; ++tile )
	{
		//current vertical position (the j we are at in this block)
		int current_j = tile*blockDim.x;
		if( (current_j+local_i) < num_stars )
		{
			float2 j_pos = d_pos[current_j + local_i]; //allow coalescing when loading the float2
			s_pos_mass[local_i].x = j_pos.x;
			s_pos_mass[local_i].y = j_pos.y;
			s_pos_mass[local_i].z = d_masses[current_j+local_i];
		}
		__syncthreads();

		//summing all the j's, we need to make sure we don't step outside the
		//matrix.
		int iterate_till = blockDim.x;
		if ( current_j + blockDim.x >= num_stars )
			iterate_till = num_stars - current_j;

#pragma unroll 128
		for( unsigned int k = 0; k < iterate_till; ++k )
		{
			//beware of the tile spanning the diagonal of the big matrix!
			if ( global_i != current_j + k )
			{
				float2 r;
				r.x = s_pos_mass[k].x - cur_pos.x;
				r.y = s_pos_mass[k].y - cur_pos.y ;

				float dist_square =  r.x*r.x + r.y*r.y + MIN_DIST; // impose min_dist to avoid infinities
				float inv_sqrt_dist = rsqrtf( dist_square ); // for computing the real gravity interaction

				//force = G*m*M/ r^2, here G = NEWTON, is multiplied only once at the end
				float acc_strength = s_pos_mass[k].z * inv_sqrt_dist * inv_sqrt_dist * inv_sqrt_dist;

				total_acc.x += acc_strength * r.x;
				total_acc.y += acc_strength * r.y;
			}
		}
		__syncthreads();
	}
	//Now we can multiply by G:
	total_acc.x *= NEWTON;
	total_acc.y *= NEWTON;

	//store the result in global memory
	d_gravs[global_i] = total_acc;
}

void compute_grav(
	float2* h_pos,
	float* h_masses,
	float2* h_gravs,
	unsigned int num_stars)
{
	//transfer data on the device
	DeviceArray<float2> d_pos( num_stars );
	d_pos.copyHostToDevice( h_pos );

	DeviceArray<float> d_masses( num_stars );
	d_masses.copyHostToDevice( h_masses );

	//instantiate a vectors to contain the gravitational forces
	DeviceArray<float2> d_gravs( num_stars );

	unsigned int grid_side = (num_stars + BLOCK_SIZE - 1) / BLOCK_SIZE;

	//get regular pointers to be passed to the kernel
	float2* d_pos_ptr = d_pos.GetPtr();
	float* d_masses_ptr = d_masses.GetPtr();
	float2* d_gravs_ptr = d_gravs.GetPtr();
	//take the positions and compute the partial sums of the forces acting on each star. We need to store partials because
	//we had to tile the matrix of the forces..
	compute_gravs<<< grid_side, BLOCK_SIZE, BLOCK_SIZE*3*sizeof(float) >>>( d_pos_ptr, d_masses_ptr, d_gravs_ptr, num_stars );
	cudaDeviceSynchronize();

	//transfer gravs back to host
	d_gravs.copyDeviceToHost( h_gravs );
}