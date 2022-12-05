#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DeviceArray.cuh"

#include <cassert>

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
| xx | xx | xx | xx | xx | x  |     |
-------------------------------     | Time direction
| xx | xx | xx | xx | xx | x  |     |
| xx | xx | xx | xx | xx | x  |     V
-------------------------------
| xx | xx | xx | xx | xx | x  |
| xx | xx | xx | xx | xx | x  |
-------------------------------
| xx | xx | xx | xx | xx | x  |
|    |    |    |    |    |    |
------------------------------- <- End
******************************/

__global__ void compute_gravs(const float* d_pos_x, const float* d_pos_y,
                              const float* d_masses, float2* d_gravs, int nu_stars )
{
    //local coordinate on the block
    const int local_i = threadIdx.x;

    //global coordinate on the grid
    const int global_i = blockIdx.x*blockDim.x + local_i;

    //this is safe from G80, since the device shuold keep track of the active threads and not wait for
    //the inactive ones in case of __syncthread
    if( global_i >= nu_stars ) return;

    extern __shared__ float s[];
    // Partition the SMEM stores so that each instruction
    // never gets any bank conflicts and coalesces access too.
    float2* pos = (float2*)s;
    float* mass = (float*)&s[2*blockDim.x];

    float2 total_acc = make_float2(0.0f, 0.0f);

    float2 cur_pos = make_float2(d_pos_x[global_i], d_pos_y[global_i]);

    for( unsigned int tile = 0; tile < gridDim.x; ++tile )
    {
        //current vertical position (the j we are at in this block)
        int current_j = tile*blockDim.x;
        if( (current_j+local_i) < nu_stars )
        {
            pos[local_i] = make_float2(
                d_pos_x[current_j + local_i], d_pos_y[current_j + local_i]);
            mass[local_i] = d_masses[current_j + local_i];
        }
        __syncthreads();

        //summing all the j's, we need to make sure we don't step outside the
        //matrix.
        int iterate_till = blockDim.x;
        if ( current_j + blockDim.x >= nu_stars )
            iterate_till = nu_stars - current_j;

#pragma unroll 128
        for( unsigned int k = 0; k < iterate_till; ++k )
        {
            //beware of the tile spanning the diagonal of the big matrix!
            if ( global_i != current_j + k )
            {
                float2 r;
                r.x = pos[k].x - cur_pos.x;
                r.y = pos[k].y - cur_pos.y ;

                const float dist_square =  r.x*r.x + r.y*r.y + MIN_DIST; // impose min_dist to avoid infinities
                const float inv_sqrt_dist = rsqrtf( dist_square ); // for computing the real gravity interaction

                //force = G*m*M/ r^2, here G = NEWTON, is multiplied only once at the end
                const float acc_strength = mass[k] * inv_sqrt_dist * inv_sqrt_dist * inv_sqrt_dist;

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
    float* h_pos_x,
    float* h_pos_y,
    float* h_masses,
    float2* h_gravs,
    unsigned int nu_stars)
{
    //transfer data on the device
    DeviceArray<float> d_pos_x( nu_stars );
    DeviceArray<float> d_pos_y( nu_stars );
    d_pos_x.copyHostToDevice( h_pos_x );
    d_pos_y.copyHostToDevice( h_pos_y );

    DeviceArray<float> d_masses( nu_stars );
    d_masses.copyHostToDevice( h_masses );

    //instantiate a vectors to contain the gravitational forces
    DeviceArray<float2> d_gravs( nu_stars );

    const unsigned int grid_side = (nu_stars + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //get regular pointers to be passed to the kernel
    float* d_pos_x_ptr = d_pos_x.GetPtr();
    float* d_pos_y_ptr = d_pos_y.GetPtr();
    float* d_masses_ptr = d_masses.GetPtr();
    float2* d_gravs_ptr = d_gravs.GetPtr();
    //take the positions and compute the partial sums of the forces acting on each star. We need to store partials because
    //we had to tile the matrix of the forces..
    compute_gravs<<< grid_side, BLOCK_SIZE, BLOCK_SIZE*3*sizeof(float) >>>(
        d_pos_x_ptr, d_pos_y_ptr, d_masses_ptr, d_gravs_ptr, nu_stars );
    cudaDeviceSynchronize();

    //transfer gravs back to host
    d_gravs.copyDeviceToHost( h_gravs );
}