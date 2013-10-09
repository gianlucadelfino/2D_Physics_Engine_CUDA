#ifndef CENTITYGALAXY_H
#define CENTITYGALAXY_H

#include <vector>
#include <memory>

#include "IEntity.h"
#include "IMoveable.h"
#include "CMoveableParticle.h"
#include "CEntityParticle.h"
#include "CRandom.h"
#include "C2DVector.h"

#include "device_compute_grav_forces.h"
#include "cuda.h"
#include "cuda_runtime.h"

//#define USE_CUDA

/**
* CEntityGalaxy is the Entity that manages the galaxy simulation. Given a Star Entity as a prototype, the constructor will populate the galaxy
* with _star_number elements.
* CEntityGalaxy follows the "Composite" patter: it IS an IEnity and manages many IEnities underneath it.
*/
class CEntityGalaxy : public IEntity
{
public:
	CEntityGalaxy( unsigned int _id, const C2DVector& _initial_pos,  const CEntityParticle& _star, unsigned int _star_number, float _max_dist );

	virtual void Update( const C2DVector& _external_force, float dt );
	virtual void Draw() const;

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> _coords );
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> _coords );

	//TODO: implement collision detection
	virtual void SolveCollision( IEntity& _otherEntity ) {}
	virtual bool TestCollision( IEntity& _otherEntity ) const { return false; }

private:
	//forbid copy and assignment
	CEntityGalaxy( const CEntityGalaxy& );
	CEntityGalaxy& operator=( const CEntityGalaxy& );

	CRandom m_rand_pos;
	CRandom m_rand_mass;

	typedef std::vector< std::shared_ptr< CEntityParticle > > StarList;
	StarList m_collection;
	unsigned int m_non_fixed_stars;
};

CEntityGalaxy::CEntityGalaxy( unsigned int _id, const C2DVector& _initial_pos,  const CEntityParticle& _star, unsigned int _star_number, float _bounding_box_side ):
	IEntity( _id, std::shared_ptr< IMoveable >( new CMoveableParticle( _initial_pos ) ) ),
	m_rand_pos(),
	m_rand_mass(),
	m_non_fixed_stars( _star_number )
{
	m_rand_pos.SetRealBounds( - _bounding_box_side/2.0f, _bounding_box_side/2.0f );
	m_rand_mass.SetRealBounds( 1, 4);

	for( unsigned int i = 0; i < _star_number; ++i )
	{
		std::shared_ptr< CEntityParticle > new_star( new CEntityParticle( _star ) );

		//randomize position.
		C2DVector pos = _initial_pos + C2DVector( m_rand_pos.RandReal(), m_rand_pos.RandReal() );
		C2DVector initial_vel = C2DVector( m_rand_pos.RandReal() / 200.0f, m_rand_pos.RandReal() / 200.0f );

		//randomize mass
		float mass = m_rand_mass.RandReal();
		//adjust size accordingly (assuming constant density). TODO: randomize density and have britness change by it. Make them turn into black holes too
		new_star->SetScale( mass );

		new_star->Reposition( pos );
		new_star->Boost( pos + initial_vel);
		new_star->SetMass( mass );

		//set id
		new_star->SetId(i);

		//add a copy of _star to the collection
		this->m_collection.push_back( new_star );
	}

	//now add a supemassive black hole to be fixed in the center
	std::shared_ptr< CEntityParticle > super_massive_black_hole( new CEntityParticle( _star ) );

	super_massive_black_hole->SetScale( 10.0f );
	super_massive_black_hole->Reposition( _initial_pos );
	float mass = 10.0f;

	super_massive_black_hole->SetMass( mass );

	//set id
	super_massive_black_hole->SetId(_star_number);

	//add a copy of _star to the collection
	this->m_collection.push_back( super_massive_black_hole );
}

#ifdef USE_CUDA
void CEntityGalaxy::Update( const C2DVector& _external_force, float dt )
{
	//instantiate some arrays to pass to the cuda kernels
	float2* star_positions	= new float2[this->m_collection.size()];
	float* masses			= new float[this->m_collection.size()];
	float2* grav_forces	= new float2[this->m_collection.size()];

	// load the vectors
	for( size_t i = 0; i < this->m_collection.size(); ++i )
	{
		//convert C2DVectors to lighter float2s
		C2DVector cur_pos = this->m_collection[i]->GetPosition();

		star_positions[i].x = cur_pos.x;
		star_positions[i].y = cur_pos.y;
		masses[i] = this->m_collection[i]->GetMass();
	}

	// call cuda kernel
	compute_grav( star_positions, masses, grav_forces, this->m_collection.size() );

	for( size_t i = 0; i < m_non_fixed_stars; ++i )//skip the super massive black hole (id = m_non_fixed_stars)
	{
		float mass_i = this->m_collection[i]->GetMass();
		this->m_collection[i]->Update( _external_force + C2DVector( mass_i*grav_forces[i].x, mass_i*grav_forces[i].y ), dt );
	}

	delete[] star_positions;
	delete[] masses;
	delete[] grav_forces;
}
#else
/**
* Update computes the total gravity acting on each single star and calls update on each. Scales as O(N^2)
*/
void CEntityGalaxy::Update( const C2DVector& _external_force, float dt )
{
	std::vector< C2DVector > pairwise_forces( this->m_collection.size()*this->m_collection.size() ); // We actully only need half of the matrix (force_ij=-force_ji), with no trace (no self interaction), however this would lead to complicated look up

	//load the forces for each pair of star in the forces vector (N^2 operation)
	for( unsigned int i = 0; i < this->m_collection.size(); ++i )
	{
		for( unsigned int j = 0 ; j < i; ++j ) //keep j < i to compute only the upper half of the matrix. The matrix is antisimmetric anyway, no need to compute it all!
		{
			float mass_i = this->m_collection[i]->GetMass();
			C2DVector pos_i = this->m_collection[i]->GetPosition();

			float mass_j = this->m_collection[j]->GetMass();
			C2DVector pos_j = this->m_collection[j]->GetPosition();

			//compute gravity
			C2DVector r =  pos_j - pos_i; // vector from i to j

			float dist = r.GetLenght();
			const float min_dist = 50.0f; // to avoid infinities
			const float NEWTON_CONST = 0.2f;

			//force = G*m*M/ r^2
			C2DVector force_ij = NEWTON_CONST * mass_i * mass_j/ ( dist * dist * dist + min_dist ) * r; //r is not normalized, therefore we divide by dist^3

			unsigned int index_ij = i +  this->m_collection.size() * j; // col + rows_num*row
			unsigned int index_ji = j +  this->m_collection.size() * i; // col + rows_num*row

			pairwise_forces[index_ij] = force_ij;
			pairwise_forces[index_ji] = (-1) * force_ij; //save redundant information for easy and fast look-ups
		}
	}

	//now add forces for each particle and apply it ( order N^2 )
	for( unsigned int i = 0; i < m_non_fixed_stars; ++i )//skip massive black hole to make it stay in the center of the screen
	{
		C2DVector force_on_i = C2DVector( 0.0f, 0.0f);
		for( unsigned int j = 0 ; j < this->m_collection.size(); ++j ) // sum all the column of forces
		{
			if ( i != j )
			{
				unsigned int index_ij = i +  m_collection.size() * j; // col + rows_num*row
				force_on_i += pairwise_forces[index_ij];
			}
		}
		this->m_collection[i]->Update( _external_force + force_on_i, dt);
	}
}
#endif

void CEntityGalaxy::Draw() const
{
	for ( StarList::const_iterator cit = m_collection.begin(); cit != m_collection.end(); ++cit )
	{
		(*cit)->Draw();
	}
}

void CEntityGalaxy::HandleMouseButtonDown( std::shared_ptr<C2DVector> _coords )
{
	//onclick add another massive star like the #1 where we clicked
	std::shared_ptr< CEntityParticle > new_star( new CEntityParticle( *(this->m_collection[m_non_fixed_stars]) ) );

	new_star->Reposition( *_coords );

	//make it super massive!
	float mass = 10000.0f;

	new_star->SetMass( mass );

	//set id
	new_star->SetId( this->m_collection.size() );

	this->m_collection.push_back( new_star );
}
void CEntityGalaxy::HandleMouseButtonUp( std::shared_ptr<C2DVector> _coords )
{
	//remove the massive star just added
	this->m_collection.pop_back();
}

#endif