#ifndef DEVICE_COMPUTE_GRAV_FORCES_H
#define DEVICE_COMPUTE_GRAV_FORCES_H

#include "cuda.h"
#include "cuda_runtime.h"

//forward declaration
void compute_grav(
	float2* h_pos,
	float* h_masses,
	float2* h_gravs,
	unsigned int num_stars);

#endif