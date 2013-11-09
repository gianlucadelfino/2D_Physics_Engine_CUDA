#ifndef CENTITY_H
#define CENTITY_H

#include <memory>
#include "C2DVector.h"
#include "IMoveable.h"
#include "IDrawable.h"

/**
* CEntity defines the interface of the entities populating the game/simulation.
*/
class CEntity
{
public:
	/**
	* CEntity constructor
	* @param id_ the id of the entity
	* @param unique_ptr moveable_ of the IMoveable object, of which takes owneship (using move semantics)
	* @param shared_ptr drawable_ of which takes owneship (using move semantics)
	*/
	CEntity( unsigned int id_, std::unique_ptr< IMoveable > moveable_, std::unique_ptr< IDrawable > drawable_ );
	CEntity( const CEntity& other_ );

	CEntity& operator=( const CEntity& rhs );

	virtual ~CEntity();

	void SetId( unsigned int id_ ) { m_id = id_; }
	unsigned int GetId() const { return m_id; }

	virtual bool IsHit( const C2DVector& coords_ ) const;

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> cursor_position_ ) {}
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> cursor_position_ ) {}

	/**
	* Update should be overridden and called if the Entity is subject to external forces and has an IMoveable
	*/
	virtual void Update( const C2DVector& external_force_, float dt ) {}

	/**
	* Draw renders the Entity if an IDrawable is available
	*/
	virtual void Draw() const;

protected:
	unsigned int m_id;
	std::unique_ptr< IMoveable > mp_moveable;
	std::unique_ptr< IDrawable > mp_drawable;
};

#endif