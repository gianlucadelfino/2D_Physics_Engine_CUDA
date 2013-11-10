#include <stdexcept>

#include "CEntityGalaxy.h"

CEntityGalaxy::CEntityGalaxy( unsigned int id_, const C2DVector& initial_pos_,  const CEntityParticle& star_, unsigned int star_number_, float _bounding_box_side, bool use_CUDA_ ):
	CEntity( id_, std::unique_ptr< IMoveable >( new CMoveableParticle( initial_pos_ ) ), NULL ),
	m_rand_pos(),
	m_rand_mass(),
	m_using_CUDA( use_CUDA_ ),
	m_original_star_num( star_number_ )
{
	//first check proper number of stars is given
	if ( this->m_original_star_num <= 1 )
	{
		throw std::runtime_error("ERROR: CEntityGalaxy must be given at least 2 stars!");
	}

	m_rand_pos.SetRealBounds( - _bounding_box_side/2.0f, _bounding_box_side/2.0f );
	m_rand_mass.SetRealBounds( 1, 4);

	for( unsigned int i = 0; i < this->m_original_star_num - 1; ++i ) //Mind the (-1), we want to add another massive star in the centre!
	{
		std::unique_ptr< CEntityParticle > new_star( new CEntityParticle( star_ ) );

		//randomize position.
		C2DVector pos = initial_pos_ + C2DVector( m_rand_pos.RandReal(), m_rand_pos.RandReal() );
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

		//add a copy of star_ to the collection
		this->m_collection.push_back( std::move( new_star ) );
	}

	//now add a supemassive black hole to be fixed in the center
	std::unique_ptr< CEntityParticle > super_massive_black_hole( new CEntityParticle( star_ ) );

	super_massive_black_hole->SetScale( 10.0f );
	super_massive_black_hole->Reposition( initial_pos_ );
	float mass = 10.0f;

	super_massive_black_hole->SetMass( mass );

	//set id
	super_massive_black_hole->SetId( star_number_ );

	//make super_massibe_black_hole static
	super_massive_black_hole->Block();

	//add a copy of star_ to the collection
	this->m_collection.push_back( std::move( super_massive_black_hole ) );
}

void CEntityGalaxy::SetUseCUDA( bool use_CUDA_ )
{
	this->m_using_CUDA = use_CUDA_;
}

void CEntityGalaxy::Update( const C2DVector& external_force_, float dt )
{
	if ( this->m_using_CUDA )
		this->UpdateCUDA( external_force_, dt );
	else
		this->UpdateCPU( external_force_, dt );
}

void CEntityGalaxy::UpdateCUDA( const C2DVector& external_force_, float dt )
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

	for( size_t i = 0; i < this->m_collection.size(); ++i )
	{
		float mass_i = this->m_collection[i]->GetMass();
		this->m_collection[i]->Update( external_force_ + C2DVector( mass_i*grav_forces[i].x, mass_i*grav_forces[i].y ), dt );
	}

	delete[] star_positions;
	delete[] masses;
	delete[] grav_forces;
}

/**
* Update computes the total gravity acting on each single star and calls update on each. Scales as O(N^2)
*/
void CEntityGalaxy::UpdateCPU( const C2DVector& external_force_, float dt )
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
	for( unsigned int i = 0; i < this->m_collection.size(); ++i )
	{
		C2DVector force_on_i = C2DVector( 0.0f, 0.0f);
		for( unsigned int j = 0 ; j < this->m_collection.size(); ++j ) // sum all the column of forces
		{
			if ( i != j )
			{
				unsigned int index_ij = i + this->m_collection.size() * j; // col + rows_num*row
				force_on_i += pairwise_forces[index_ij];
			}
		}
		this->m_collection[i]->Update( external_force_ + force_on_i, dt);
	}
}

void CEntityGalaxy::Draw() const
{
	for ( StarList::const_iterator cit = this->m_collection.begin(); cit != this->m_collection.end(); ++cit )
	{
		(*cit)->Draw();
	}
}

void CEntityGalaxy::HandleMouseButtonDown( std::shared_ptr<C2DVector> coords_ )
{
	//onclick add another massive star like the last one, in the place we clicked
	if ( this->m_collection.size() <= ( this->m_original_star_num ) ) //m_original_star_num is the current total of stars counting the black hole
	{
		std::unique_ptr< CEntityParticle > new_star( new CEntityParticle( *(this->m_collection[0]) ) ); //use the first star as a prototype

		new_star->Reposition( *coords_ );

		//make it super massive!
		float mass = 10000.0f;

		new_star->SetMass( mass );

		//set id
		new_star->SetId( this->m_collection.size() );
		new_star->HandleMouseButtonDown( coords_ );
		this->m_collection.push_back( std::move( new_star ) );
	}
}
void CEntityGalaxy::HandleMouseButtonUp( std::shared_ptr<C2DVector> coords_ )
{
	//remove the massive star just added
	if ( this->m_collection.size() > ( this->m_original_star_num ) ) //m_original_star_num is the current total of stars counting the black hole
		this->m_collection.pop_back();
}