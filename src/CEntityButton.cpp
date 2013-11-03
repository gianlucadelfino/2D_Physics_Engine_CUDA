#include "CEntityButton.h"
#include "CWorld.h"

CEntityButton::CEntityButton( unsigned int id_, std::unique_ptr< IMoveable > moveable_, std::unique_ptr<IDrawable> drawable_, CWorld& world_, std::unique_ptr< IScene > scene_to_switch_to_ ):
	IEntity( id_, std::move( moveable_ ), std::move( drawable_ ) ),
	mr_world( world_ ),
	mp_scene_to_switch_to( std::move( scene_to_switch_to_ ) )
{}

void CEntityButton::HandleMouseButtonDown( std::shared_ptr<C2DVector> cursor_position_ )
{}

void CEntityButton::HandleMouseButtonUp( std::shared_ptr<C2DVector> cursor_position_ )
{
	this->mr_world.ChangeScene( std::move( this->mp_scene_to_switch_to ) );
}

bool CEntityButton::IsHit( const C2DVector& coords_ ) const
{
	bool is_hit = false;
	if ( this->mp_moveable )
	{
		is_hit = this->mp_moveable->IsHit( coords_ );
	}
	return is_hit;
}