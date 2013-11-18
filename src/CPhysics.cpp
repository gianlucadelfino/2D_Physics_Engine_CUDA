#include <cassert>

#include "CPhysics.h"

CPhysics::CPhysics( float mass_ )
	:m_mass(mass_),
	m_inverseMass(1/mass_),
	m_gravity_acc(C2DVector(0.0f, 9.8f))
{}

CPhysics::CPhysics( float mass_, const C2DVector& gravity_accel_ )
	:m_mass(mass_),
	m_inverseMass(1/mass_),
	m_gravity_acc(gravity_accel_)
{}

CPhysics::CPhysics( const CPhysics& other_ ):
	m_mass( other_.m_mass ),
	m_inverseMass( other_.m_inverseMass ),
	m_gravity_acc( other_.m_gravity_acc )
{}

/*
* Clone is needed to create a deep copy when IMoveable is used as pimpl
*/
std::unique_ptr< CPhysics > CPhysics::Clone() const
{
	std::unique_ptr< CPhysics > clone( this->DoClone() );
	//lets check that the derived class actually implemented clone and it does not come from a parent
	assert( typeid(*clone) == typeid(*this) && "DoClone incorrectly overridden");
	return clone;
}

std::unique_ptr< CPhysics > CPhysics::DoClone() const
{
	return std::unique_ptr< CPhysics >( new CPhysics( m_mass, this->m_gravity_acc ) );
}

CPhysics& CPhysics::operator=( const CPhysics& other_ )
{
	if ( &other_ != this )
	{
		this->m_mass = other_.m_mass;
		this->m_inverseMass = other_.m_inverseMass;
		this->m_gravity_acc = other_.m_gravity_acc;
	}
	return *this;
}

CPhysics::~CPhysics(){}

//mind the reference to the unique ptr because we want to modify it without sharing ownership!
//(remind that unique_ptr cant be passed by value without moving ownershipt and therfore destroying the original)
void CPhysics::Update( const C2DVector& external_force_, std::unique_ptr<IMoveable>& moveable_, float dt )
{
	this->Integrate( moveable_->pos, moveable_->old_pos, external_force_, dt );
}

/**
* Integrate increments the position of the entity by integrating the force acting on it
*/
void CPhysics::Integrate( C2DVector& pos, C2DVector& old_pos, const C2DVector& external_force_, float dt )
{
	//verlet
	C2DVector temp = pos;
	pos += (pos - old_pos) + this->m_inverseMass *( this->GetForce( pos ) + external_force_ )* dt * dt;
	old_pos = temp;
}