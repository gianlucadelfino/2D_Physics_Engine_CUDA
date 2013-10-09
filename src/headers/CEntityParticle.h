#ifndef CENTITYPARTICLE_H
#define CENTITYPARTICLE_H

#include <memory>
#include "IEntity.h"
#include "IDrawable.h"
#include "CPhysics.h"
#include "CMoveableParticle.h"

class CEntityParticle: public IEntity
{
public:
	CEntityParticle( const unsigned int _id, std::shared_ptr<IMoveable> _m, std::shared_ptr<IDrawable> _d, std::shared_ptr<CPhysics> _p);

	CEntityParticle( const CEntityParticle& e );
	CEntityParticle& operator=( const CEntityParticle& rhs);

	virtual ~CEntityParticle();

	virtual void Draw() const;
	virtual void Update( const C2DVector& _external_force, float dt );

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> _coords);
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> _coords );

	void AddDrawable( std::shared_ptr<IDrawable> _drawable );
	void AddPhysics( std::shared_ptr<CPhysics> _physics );

	//TODO: implement collision detection
	virtual void SolveCollision( IEntity& _otherEntity ) {}
	virtual bool TestCollision( IEntity& _otherEntity ) const { return false; }

	/* Accessors to pimpl properties  */
	//IMoveable
	void Reposition( const C2DVector& _new_position );
	void Boost( const C2DVector& _new_position );
	void Translate( const C2DVector& _shift );

	void Block() { m_is_static = true; }
	void Unblock() { m_is_static = false; }
	C2DVector GetPosition() const { return mp_moveable->pos; }
	void SetConstraint( std::shared_ptr<C2DVector> _origin, const C2DVector& _displacement );

	//IPhysics
	float GetMass() const;
	void SetMass( float _mass );

	//IDrawable
	void SetSize( const C2DVector& _size );
	void SetScale( float _scale );
private:
	std::shared_ptr<CPhysics>  mp_physics;
	std::shared_ptr<IDrawable> mp_drawable;
	bool m_is_static;
};

CEntityParticle::CEntityParticle( const unsigned int _id, std::shared_ptr<IMoveable> _m, std::shared_ptr<IDrawable> _d, std::shared_ptr<CPhysics> _p):
	IEntity( _id, _m),
	mp_physics( _p ),
	mp_drawable( _d ),
	m_is_static(false)
{}

/**
* For the copy constructor remind that both pimpl objects drawable and physics are NOT shared,
* so must clone 'em.
*/
CEntityParticle::CEntityParticle( const CEntityParticle& e ):
	IEntity( e ),
	mp_physics( e.mp_physics? e.mp_physics->Clone() : NULL ),
	mp_drawable( e.mp_drawable? e.mp_drawable->Clone() : NULL ),
	m_is_static(false)
{}

CEntityParticle& CEntityParticle::operator=( const CEntityParticle& rhs)
{
	if( this != &rhs )
	{
		IEntity::operator=( rhs );
		this->mp_physics  = rhs.mp_physics;
		this->mp_drawable = rhs.mp_drawable;
		this->m_is_static = rhs.m_is_static;
	}

	return *this;
}

void CEntityParticle::Draw() const
{
	if( mp_drawable )
	{
		mp_drawable->Draw( mp_moveable->pos );
	}
}
void CEntityParticle::Update( const C2DVector& _external_force, float dt )
{
	if( !m_is_static && mp_physics )
	{
		mp_physics->Update( _external_force, mp_moveable , dt);
		//impose local constraints
		mp_moveable->ImposeConstraints();
	}
}

void CEntityParticle::AddDrawable( std::shared_ptr<IDrawable> _drawable )
{
	mp_drawable = _drawable;
}

void CEntityParticle::AddPhysics( std::shared_ptr<CPhysics> _physics )
{
	mp_physics = _physics;
}

void CEntityParticle::HandleMouseButtonDown( std::shared_ptr<C2DVector> _coords )
{
	if ( mp_moveable->IsHit( *_coords ) )
	{
		this->SetConstraint( _coords, C2DVector( 0.0f, 0.0f ) );
	}
}

void CEntityParticle::HandleMouseButtonUp( std::shared_ptr<C2DVector> _coords )
{
	mp_moveable->UnsetConstraint();
}

/*
* Reposition moves the particle and sets its speed to zero
*/
void CEntityParticle::Reposition( const C2DVector& _new_position )
{
	if ( !m_is_static )
	{
		mp_moveable->Reposition( _new_position );
	}
}

/*
* Move offsets the particle which makes it gain speed
*/
void CEntityParticle::Boost( const C2DVector& _new_position )
{
	if ( mp_moveable && !m_is_static )
		mp_moveable->Boost( _new_position );
}

/*
* Translate moves particle and its velocity vector by a shift vector
*/
void CEntityParticle::Translate( const C2DVector& _shift )
{
	if ( mp_moveable && !m_is_static )
		mp_moveable->Translate( _shift );
}

void CEntityParticle::SetConstraint( std::shared_ptr<C2DVector> _origin, const C2DVector& _displacement )
{
	if ( mp_moveable)
		mp_moveable->SetConstraint( _origin, _displacement );
}

float CEntityParticle::GetMass() const
{
	float mass = 0.0f;
	if ( mp_physics )
		mass = mp_physics->GetMass();

	return mass;
}

void CEntityParticle::SetMass( float _mass )
{
	if ( mp_physics )
		mp_physics->SetMass( _mass );
}

void CEntityParticle::SetSize( const C2DVector& _size )
{
	if ( mp_drawable )
		mp_drawable->SetSize( _size );
}

void CEntityParticle::SetScale( float _scale)
{
	if ( mp_drawable )
		mp_drawable->SetScale( _scale );
}

CEntityParticle::~CEntityParticle()
{}

#endif