#ifndef CENTITYBUTTON_H
#define CENTITYBUTTON_H

#include <memory>
#include "IEntity.h"
#include "IDrawable.h"
#include "IMoveable.h"
#include "IScene.h"

class CWorld;

class CEntityButton : public IEntity
{
public:
	CEntityButton( unsigned int id_, std::unique_ptr< IMoveable > moveable_, std::unique_ptr<IDrawable> drawable_, CWorld& world_, std::unique_ptr< IScene > scene_to_switch_to_ );

	virtual void HandleMouseButtonDown( std::shared_ptr<C2DVector> cursor_position_ );
	virtual void HandleMouseButtonUp( std::shared_ptr<C2DVector> cursor_position_ );

	virtual bool IsHit( const C2DVector& coords_ ) const;

private:
	CWorld& mr_world;
	std::unique_ptr< IScene > mp_scene_to_switch_to;
};

#endif