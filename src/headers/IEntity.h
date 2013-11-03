#ifndef IENTITY_H
#define IENTITY_H

#include <memory>
#include "C2DVector.h"
#include "IMoveable.h"
#include "IDrawable.h"

/**
* This class defines the interface of the entities populating the game/simulation.
*/
class IEntity
{
public:
	/**
	* IEntity constructor
	* @param id_ the id of the entity
	* @param unique_ptr moveable_ of the IMoveable object, of which takes owneship (using move semantics)
	* @param shared_ptr drawable_ of which takes owneship (using move semantics)
	*/
	IEntity( unsigned int id_, std::unique_ptr< IMoveable > moveable_, std::unique_ptr< IDrawable > drawable_ ):
		m_id( id_ ),
		mp_moveable( std::move( moveable_ ) ), //transfer ownership with move semantics
		mp_drawable( std::move( drawable_ ) )  //transfer ownership with move semantics
	{}

	IEntity( const IEntity& other_ ):
		m_id( other_.m_id )
	{
		//I need to deep copy the pointee, otherwise i am using the same IMovable!
		if ( other_.mp_moveable )
			this->mp_moveable = std::move( other_.mp_moveable->Clone() );
		else
			this->mp_moveable = NULL;

		//I need to deep copy the pointee, otherwise i am using the same IDrawable!
		if ( other_.mp_drawable )
			this->mp_drawable = std::move( other_.mp_drawable->Clone() );
		else
			this->mp_drawable = NULL;
	}

	IEntity& operator=( const IEntity& rhs )
	{
		if ( this != &rhs )
		{
			this->m_id = rhs.m_id;
			if ( rhs.mp_moveable )
				this->mp_moveable = std::move( rhs.mp_moveable->Clone() ); //I need to deep copy the pointee, otherwise i am using the same IMovable!
			else
				this->mp_moveable = NULL;

			if ( rhs.mp_drawable )
				this->mp_drawable = std::move( rhs.mp_drawable->Clone() ); //I need to deep copy the pointee, otherwise i am using the same IDrawable
			else
				this->mp_drawable = NULL;
		}
		return *this;
	}

	virtual ~IEntity() {}

	void SetId( unsigned int id_ ) { m_id = id_; }
	unsigned int GetId() const { return m_id; }

	bool IsHit( const C2DVector& coords_ ) const
	{
		if ( this->mp_moveable )
			return this->mp_moveable->IsHit( coords_ );
		else
			return false;
	}

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> cursor_position_ ) {}
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> cursor_position_ ) {}

	virtual void Update( const C2DVector& external_force_, float dt ) {}
	virtual void Draw() const
	{
		if( this->mp_drawable && this->mp_moveable	)
			mp_drawable->Draw( this->mp_moveable->pos, this->mp_moveable->orientation );
	}
	virtual void SolveCollision( IEntity& otherEntity ) {}
	virtual bool TestCollision( IEntity& otherEntity ) const { return false; }

protected:
	unsigned int m_id;
	std::unique_ptr< IMoveable > mp_moveable;
	std::unique_ptr< IDrawable > mp_drawable;
};

#endif