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
	CPhysics( float _mass )
		:m_mass(_mass),
		m_inverseMass(1/_mass),
		m_gravity_acc(C2DVector(0.0f, 9.8f))
	{}

	CPhysics( float _mass, const C2DVector& _gravity_accel )
		:m_mass(_mass),
		m_inverseMass(1/_mass),
		m_gravity_acc(_gravity_accel)
	{}

	CPhysics( const CPhysics& _other ):
		m_mass( _other.m_mass ),
		m_inverseMass( _other.m_inverseMass ),
		m_gravity_acc( _other.m_gravity_acc )
	{}

	virtual CPhysics* Clone() const
	{
		CPhysics* clone = new CPhysics( m_mass, this->m_gravity_acc );
		return clone;
	}

	CPhysics& operator=( const CPhysics& _other )
	{
		if ( &_other != this )
		{
			this->m_mass = _other.m_mass;
			this->m_inverseMass = _other.m_inverseMass;
			this->m_gravity_acc = _other.m_gravity_acc;
		}
		return *this;
	}

	virtual ~CPhysics(){}

	//mind the reference to the shared ptr because we want to modify it!
	virtual void Update( const C2DVector& _external_force, std::shared_ptr<IMoveable>& _moveable, float dt )
	{
		this->Integrate( _moveable->pos, _moveable->old_pos, _external_force, dt );
	}

	virtual C2DVector GetForce( const C2DVector& pos ) const { return m_mass*m_gravity_acc; }

	float GetMass() const { return m_mass; }
	void SetMass( float _mass ) { m_mass = _mass; }

	void SetGravity( const C2DVector& _grav ){ m_gravity_acc = _grav; }

protected:
	float m_mass;
	float m_inverseMass;
	C2DVector m_gravity_acc;

	void Integrate( C2DVector& pos, C2DVector& old_pos, const C2DVector& _external_force, float dt );
};

#endif