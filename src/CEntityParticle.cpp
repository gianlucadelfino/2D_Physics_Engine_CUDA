#include "CEntityParticle.h"

CEntityParticle::CEntityParticle( const unsigned int id_, std::unique_ptr<IMoveable> m_, std::unique_ptr<IDrawable> d_, std::unique_ptr<CPhysics> p_):
	CEntity( id_, std::move( m_ ), std::move( d_ ) ),
	mp_physics(  std::move( p_ ) ),
	m_is_static(false)
{}

/**
* For the copy constructor remind that both pimpl objects drawable and physics are NOT shared,
* so must clone 'em.
*/
CEntityParticle::CEntityParticle( const CEntityParticle& other_ ):
	CEntity( other_ ),
	m_is_static( other_.m_is_static )
{
	if ( other_.mp_physics )
		this->mp_physics = std::move( other_.mp_physics->Clone() );
	else
		this->mp_physics = NULL;
	if ( other_.mp_drawable )
		this->mp_drawable = std::move( other_.mp_drawable->Clone() );
	else
		this->mp_drawable = NULL;
}

CEntityParticle& CEntityParticle::operator=( const CEntityParticle& rhs)
{
	if( this != &rhs )
	{
		CEntity::operator=( rhs );
		if ( rhs.mp_physics )
			this->mp_physics  = std::move( rhs.mp_physics->Clone() );
		else
			this->mp_physics = NULL;
		if ( rhs.mp_drawable )
			this->mp_drawable = std::move( rhs.mp_drawable->Clone() );
		else
			this->mp_drawable = NULL;
		this->m_is_static = rhs.m_is_static;
	}

	return *this;
}

void CEntityParticle::Draw() const
{
	if( this->mp_drawable )
	{
		this->mp_drawable->Draw( mp_moveable->pos, C2DVector( 0.0f, 0.0f ) );
	}
}
void CEntityParticle::Update( const C2DVector& external_force_, float dt )
{
	if( !this->m_is_static && this->mp_physics && this->mp_physics )
	{
		this->mp_physics->Update( external_force_, mp_moveable , dt);
		//impose local constraints
		this->mp_moveable->ImposeConstraints();
	}
}

void CEntityParticle::AddDrawable( std::unique_ptr<IDrawable> drawable_ )
{
	this->mp_drawable = std::move( drawable_->Clone() );
}

void CEntityParticle::AddPhysics( std::unique_ptr<CPhysics> physics_ )
{
	this->mp_physics = std::move( physics_->Clone() );
}

void CEntityParticle::HandleMouseButtonDown( std::shared_ptr<C2DVector> coords_ )
{
	if ( this->mp_moveable->IsHit( *coords_ ) )
	{
		this->SetConstraint( coords_ );
	}
}

void CEntityParticle::HandleMouseButtonUp( std::shared_ptr<C2DVector> coords_ )
{
	this->mp_moveable->UnsetConstraint();
}

/*
* Reposition moves the particle and sets its speed to zero
*/
void CEntityParticle::Reposition( const C2DVector& new_position_ )
{
	if ( !this->m_is_static )
	{
		this->mp_moveable->Reposition( new_position_ );
	}
}

/*
* Move offsets the particle which makes it gain speed
*/
void CEntityParticle::Boost( const C2DVector& new_position_ )
{
	if ( this->mp_moveable && !this->m_is_static )
		this->mp_moveable->Boost( new_position_ );
}

/*
* Translate moves particle and its velocity vector by a shift vector
*/
void CEntityParticle::Translate( const C2DVector& shift_ )
{
	if ( this->mp_moveable && !this->m_is_static )
		this->mp_moveable->Translate( shift_ );
}

void CEntityParticle::SetConstraint( std::shared_ptr<C2DVector> constrainted_pos_ )
{
	if ( this->mp_moveable )
		this->mp_moveable->SetConstraint( constrainted_pos_ );
}

float CEntityParticle::GetMass() const
{
	float mass = 0.0f;
	if ( this->mp_physics )
		mass = this->mp_physics->GetMass();

	return mass;
}

void CEntityParticle::SetMass( float mass_ )
{
	if ( this->mp_physics )
		this->mp_physics->SetMass( mass_ );
}

void CEntityParticle::SetSize( const C2DVector& size_ )
{
	if ( this->mp_drawable )
		this->mp_drawable->SetSize( size_ );
}

void CEntityParticle::SetScale( float scale_ )
{
	if ( this->mp_drawable )
		this->mp_drawable->SetScale( scale_ );
}

CEntityParticle::~CEntityParticle()
{}