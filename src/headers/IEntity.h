#ifndef IENTITY_H
#define IENTITY_H

#include <memory>
#include "C2DVector.h"
#include "IMoveable.h"
#include "IDrawable.h"

/**
* This class defines the interface of the entities populating the game/simulation
*/
class IEntity
{
public:
	IEntity( unsigned int _id, std::shared_ptr< IMoveable > _moveable ):
		m_id( _id ),
		mp_moveable( _moveable )
	{}
	IEntity( const IEntity& _other ):
		m_id( _other.m_id ),
		mp_moveable(  _other.mp_moveable->Clone() ) //I need to deep copy the pointee, otherwise i am using the same Imovable!
	{}

	IEntity& operator=( const IEntity& rhs )
	{
		if ( this != &rhs )
		{
			this->m_id = rhs.m_id;
			mp_moveable = std::shared_ptr< IMoveable >( rhs.mp_moveable->Clone() ); //I need to deep copy the pointee, otherwise i am using the same Imovable!
		}
		return *this;
	}

	virtual ~IEntity() = 0;

	void SetId( unsigned int _id ) { m_id = _id; }
	unsigned int GetId() const { return m_id; }

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> _coords ) = 0;
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> _coords ) = 0;

	virtual void Update( const C2DVector& _external_force, float dt ) = 0;
	virtual void Draw() const = 0;

	virtual void SolveCollision( IEntity& otherEntity ) = 0;
	virtual bool TestCollision( IEntity& otherEntity ) const = 0;

protected:
	unsigned int m_id;
	std::shared_ptr< IMoveable > mp_moveable;
};

//virtual destructor definition, must be there. Inline to avoid the cost of a call to empty body..
inline IEntity::~IEntity() {}

#endif