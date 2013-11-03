#ifndef CPHYSICS_H
#define CPHYSICS_H

#include <memory>
#include "C2DVector.h"
#include "IMoveable.h"

/**
* CPhysics defines the interface and basic implementation for entities that are bound to follow physics laws.
* It is intended to be used
*/
class CPhysics
{
public:
	CPhysics( float mass_ )
		:m_mass(mass_),
		m_inverseMass(1/mass_),
		m_gravity_acc(C2DVector(0.0f, 9.8f))
	{}

	CPhysics( float mass_, const C2DVector& gravity_accel_ )
		:m_mass(mass_),
		m_inverseMass(1/mass_),
		m_gravity_acc(gravity_accel_)
	{}

	CPhysics( const CPhysics& other_ ):
		m_mass( other_.m_mass ),
		m_inverseMass( other_.m_inverseMass ),
		m_gravity_acc( other_.m_gravity_acc )
	{}

	virtual std::unique_ptr< CPhysics > Clone() const
	{
		return std::unique_ptr< CPhysics >( new CPhysics( m_mass, this->m_gravity_acc ) );
	}

	CPhysics& operator=( const CPhysics& other_ )
	{
		if ( &other_ != this )
		{
			this->m_mass = other_.m_mass;
			this->m_inverseMass = other_.m_inverseMass;
			this->m_gravity_acc = other_.m_gravity_acc;
		}
		return *this;
	}

	virtual ~CPhysics(){}

	//mind the reference to the unique ptr because we want to modify it without sharing ownership!
	//(remind that unique_ptr cant be passed by value without moving ownershipt and therfore destroying the original)
	virtual void Update( const C2DVector& external_force_, std::unique_ptr<IMoveable>& moveable_, float dt )
	{
		this->Integrate( moveable_->pos, moveable_->old_pos, external_force_, dt );
	}

	virtual C2DVector GetForce( const C2DVector& pos ) const { return m_mass*m_gravity_acc; }

	float GetMass() const { return m_mass; }
	void SetMass( float mass_ ) { m_mass = mass_; }

	void SetGravity( const C2DVector& _grav ){ m_gravity_acc = _grav; }

protected:
	float m_mass;
	float m_inverseMass;
	C2DVector m_gravity_acc;

	void Integrate( C2DVector& pos, C2DVector& old_pos, const C2DVector& external_force_, float dt );
};

#endif