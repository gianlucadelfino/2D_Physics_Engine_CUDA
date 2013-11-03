#ifndef CENTITYCLOTH_H
#define CENTITYCLOTH_H

#include <memory>
#include <vector>
#include <cmath>
#include "CEntityParticle.h"
#include "C2DVector.h"
#include "CDrawableLink.h"

/**
* CEntityCloth is a child of CEntityParticle to contain CEntities, following the Pattern of the "Composite".
* Calling Update on the collection will update all the Entities in it and impose all the constraints.
*/
class CEntityCloth: public IEntity
{
public:
	CEntityCloth( unsigned int id_, const C2DVector& initial_pos_, std::unique_ptr< CDrawableLink > drawable_,  const CEntityParticle& seam_, unsigned int side_length_ );

	CEntityCloth( const CEntityCloth& );
	CEntityCloth& operator=( const CEntityCloth& );

	void AddToCollection( std::shared_ptr<CEntityParticle> _new_entity );

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> coords_ );
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> coords_ );

	virtual void Update( const C2DVector& external_force_, float dt );
	virtual void Draw() const;

	//TODO: implement collision detection
	virtual void SolveCollision( IEntity& other_Entity ) {}
	virtual bool TestCollision( IEntity& other_Entity ) const { return false; }

private:
	//container, for now vector will do
	std::vector< std::shared_ptr<CEntityParticle> > m_collection;
	unsigned int m_side_length;
	unsigned int m_total_seams;
	C2DVector m_cloth_pos;

	//the more iteration, the more accurate the simulation
	static const int NUMBER_OF_ITERATIONS = 2;
	//define distance between cloth seams
	const float m_max_dist;

	virtual void ApplyCollectiveConstraints( const unsigned int id );
	void EnforceMaxDist( int id_a, int id_b );
};

#endif