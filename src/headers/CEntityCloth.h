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
	CEntityCloth( unsigned int _id, const C2DVector& _initial_pos, std::shared_ptr< CDrawableLink > _drawable,  const CEntityParticle& _seam, unsigned int _side_length );

	CEntityCloth( const CEntityCloth& );
	CEntityCloth& operator=( const CEntityCloth& );

	void AddToCollection( std::shared_ptr<CEntityParticle> _new_entity );

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> _coords );
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> _coords );

	virtual void Update( const C2DVector& _external_force, float dt );
	virtual void Draw() const;

	//TODO: implement collision detection
	virtual void SolveCollision( IEntity& _otherEntity ) {}
	virtual bool TestCollision( IEntity& _otherEntity ) const { return false; }

private:
	//container, for now vector will do
	std::vector< std::shared_ptr<CEntityParticle> > m_collection;
	std::shared_ptr< CDrawableLink > mp_drawable;
	unsigned int m_side_length;
	unsigned int m_total_seams;
	C2DVector m_cloth_pos;

	//define distance between cloth seams
	static const int NUMBER_OF_ITERATIONS = 2;
	const float m_max_dist;

	virtual void ApplyCollectiveConstraints( const unsigned int id );
	void EnforceMaxDist( int id_a, int id_b );
};

CEntityCloth::CEntityCloth( unsigned int _id, const C2DVector& _initial_pos, std::shared_ptr< CDrawableLink > _drawable, const CEntityParticle& _seam, unsigned int _side_length ):
	IEntity( _id, std::shared_ptr< IMoveable > ( new CMoveableParticle( _initial_pos ) ) ),
	mp_drawable( _drawable ),
	m_side_length( _side_length ),
	m_total_seams( _side_length*_side_length ),
	m_max_dist( 15.0f )
{
	// build the cloth
	for ( unsigned int i = 0; i < m_total_seams ; ++i )
	{
		// column number
		unsigned int col = i % m_side_length;
		// row number
		unsigned int row = i / m_side_length;
		// compute the position
		C2DVector pos( _initial_pos + m_max_dist * C2DVector( col * 1.0f, row * 1.0f ) );

		//add to the collection
		this->m_collection.push_back( std::shared_ptr<CEntityParticle>( new CEntityParticle( _seam ) ) );

		//reposition it and set the id..
		this->m_collection[i]->Reposition( pos );

		this->m_collection[i]->SetId(i);
	}

	//add constraint to top row so they dont move (this assumes linear internal structure like vector)
	for ( unsigned int i = 0; i < m_side_length; ++i )
	{
		this->m_collection[i]->Block();
	}
}

CEntityCloth::CEntityCloth( const CEntityCloth& _other ):
	IEntity( _other ), //assign the position of the first particle
	m_side_length( _other.m_side_length ),
	m_total_seams( _other.m_side_length*_other.m_side_length ),
	m_max_dist( _other.m_max_dist )
{
	// build the cloth
	for ( unsigned int i = 0; i < m_total_seams ; ++i )
	{
		// compute the position
		C2DVector pos = _other.m_collection[i]->GetPosition();

		//add to the collection
		this->m_collection.push_back( std::shared_ptr<CEntityParticle>( new CEntityParticle( *_other.m_collection[i] ) ) );//pass a copy of the particle

		//reposition it and set the id..
		this->m_collection[i]->Reposition( pos );

		this->m_collection[i]->SetId(i);
	}

	//add constraint to top row so they dont move (this assumes linear internal structure like vector)
	for ( unsigned int i = 0; i < m_side_length; ++i )
	{
		this->m_collection[i]->Block();
	}
}

CEntityCloth& CEntityCloth::operator=( const CEntityCloth& rhs )
{
	if ( this != &rhs )
	{
		IEntity::operator=( rhs );
		this->m_side_length = rhs.m_side_length;
		this->m_total_seams = rhs.m_side_length*rhs.m_side_length;
	}
	return *this;
}

void CEntityCloth::AddToCollection( std::shared_ptr<CEntityParticle> _new_entity )
{
	_new_entity->SetId( m_collection.size() );
	m_collection.push_back( _new_entity );
}

void CEntityCloth::HandleMouseButtonDown( std::shared_ptr<C2DVector> _coords )
{
	for ( std::vector< shared_ptr< CEntityParticle > >::iterator it = m_collection.begin(); it != m_collection.end(); ++it )
		(*it)->HandleMouseButtonDown( _coords );
}

void CEntityCloth::HandleMouseButtonUp( std::shared_ptr<C2DVector> _coords )
{
	for ( std::vector< shared_ptr< CEntityParticle > >::iterator it = m_collection.begin(); it != m_collection.end(); ++it )
		(*it)->HandleMouseButtonUp( _coords );
}

void CEntityCloth::Update( const C2DVector& _external_force, float dt )
{
	//call update on all elements of c_collection, passing the collection total force
	//this indirectly applies local constraints via the entity update
	for ( std::vector< std::shared_ptr<CEntityParticle> >::iterator it = m_collection.begin(); it != m_collection.end(); ++it )
	{
		(*it)->Update( _external_force, dt );
	}

	//applying the collective constraints, the more times we apply them, the more precise the simulation will be
	for ( unsigned int iter = 0; iter < NUMBER_OF_ITERATIONS; ++iter)
	{
		for ( unsigned int id = 0; id < m_collection.size(); ++id )
		{
			this->ApplyCollectiveConstraints( id );
		}
	}
}

void CEntityCloth::Draw() const
{
	//draw each link, for each id, we draw the left and top link
	for ( unsigned int id = 0; id < this->m_collection.size(); ++id )
	{
		//find the links to draw
		int top_id = id - m_side_length;
		int left_id = id - 1;

		//find the coords of the 3 points to use in drawing the 2 lines
		C2DVector id_pos = this->m_collection[id]->GetPosition();
		this->mp_drawable->SetOrigin( id_pos );
		if ( top_id >= 0 )
		{
			C2DVector top_id_position = this->m_collection[top_id]->GetPosition();
			this->mp_drawable->Draw( top_id_position );
		}

		if ( (left_id % m_side_length) < (id % m_side_length) )
		{
			C2DVector left_id_position = this->m_collection[left_id]->GetPosition();
			this->mp_drawable->Draw( left_id_position );
		}
	}
}

void CEntityCloth::ApplyCollectiveConstraints( const unsigned int id )
{
	if ( id != 0 )//the first one doesnt move
	{
		// look at the distance with the seam above and on the left. The id tells me the position in the matrix
		int top_id = id - m_side_length;
		int left_id = id - 1;

		// ensure particle is not on the top edge
		if ( top_id >= 0 )
			EnforceMaxDist( id, top_id );

		// ensure the particle is not on the left edge.
		if ( (left_id % m_side_length) < (id % m_side_length)  )
			EnforceMaxDist( id, left_id);
	}
}

void CEntityCloth::EnforceMaxDist( int id_a, int id_b )
{
	C2DVector a_pos = m_collection[id_a]->GetPosition();
	C2DVector b_pos = m_collection[id_b]->GetPosition();

	//compute the difference vector and the distance
	C2DVector ab_vect = a_pos - b_pos;
	float ab_length = sqrt( ab_vect.GetSquaredLength() );
	float delta = ( ab_length - m_max_dist ) / ab_length;

	if ( delta > 0.01f ) //impose only if be greater than a small treshold
	{
		a_pos -= ab_vect * delta * 0.5f;
		b_pos += ab_vect * delta * 0.5f;

		//apply the changes
		m_collection[id_a]->Boost( a_pos );
		m_collection[id_b]->Boost( b_pos );
	}
}

#endif