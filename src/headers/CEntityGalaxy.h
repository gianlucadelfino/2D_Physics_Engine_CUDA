#ifndef CENTITYGALAXY_H
#define CENTITYGALAXY_H

#include <vector>
#include <memory>

#include "CEntity.h"
#include "IMoveable.h"
#include "CMoveableParticle.h"
#include "CEntityParticle.h"
#include "CRandom.h"
#include "C2DVector.h"

#include "device_compute_grav_forces.h"
#include "cuda.h"
#include "cuda_runtime.h"

/**
* CEntityGalaxy is the Entity that manages the galaxy simulation. Given a Star Entity as a prototype, the constructor will populate the galaxy
* with star_number_ elements.
* CEntityGalaxy follows the "Composite" patter: it IS an IEnity and manages many IEnities underneath it.
*/
class CEntityGalaxy : public CEntity
{
public:
	CEntityGalaxy( unsigned int id_, const C2DVector& initial_pos_,  const CEntityParticle& star_, unsigned int star_number_, float max_dist_, bool use_CUDA_ );

	virtual void Update( const C2DVector& external_force_, float dt );
	virtual void Draw() const;

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> coords_ );
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> coords_ );

	/**
	* SetsUseCUDA sets whether to use the CUDA and the GPU or the CPU
	*/
	void SetUseCUDA( bool use_CUDA_ );

private:
	//forbid copy and assignment
	CEntityGalaxy( const CEntityGalaxy& );
	CEntityGalaxy& operator=( const CEntityGalaxy& );

	void UpdateCUDA( const C2DVector& external_force_, float dt );
	void UpdateCPU( const C2DVector& external_force_, float dt );

	CRandom m_rand_pos;
	CRandom m_rand_mass;

	bool m_using_CUDA;

	typedef std::vector< std::unique_ptr< CEntityParticle > > StarList;
	StarList m_collection;
	unsigned int m_original_star_num;
};

#endif