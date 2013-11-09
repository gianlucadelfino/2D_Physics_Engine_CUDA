#ifndef DEVICE_COMPUTE_GRAV_FORCES_H
#define DEVICE_COMPUTE_GRAV_FORCES_H

#include "CUDA.h"
#include "CUDA_runtime.h"

/**
* compute_grav computes the forces acting on each stars.
* @param h_pos the pointer to the host vector holding the current positions of the stars
* @param h_masses the pointer to the host vector holding all the masses of the stars
* @param h_gravs the pointer to the host vector that is going to be filled with the forces
* @param num_stars is the total number of the stars
*/
void compute_grav(
	float2* h_pos,
	float* h_masses,
	float2* h_gravs,
	unsigned int num_stars);

#endif