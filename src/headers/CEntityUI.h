#ifndef CENTITYUI_H
#define CENTITYUI_H

#include "IEntity.h"
#include "IMoveable.h"
#include "CGameStateManager.h"
#include <memory>

/**
* CEntityUI defines a piece of the User Interface of the simulation
*/
class CEntityUI: public IEntity
{
	CEntityUI( unsigned int _id, std::shared_ptr< IMoveable > _moveable, std::shared_ptr< CGameStateManager > _game_state_manager ):
		IEntity( _id, _moveable){}

	CEntityUI( , )
};

#endif