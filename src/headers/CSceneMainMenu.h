#ifndef CSCENEMAINMENU_H
#define CSCENEMAINMENU_H

#include <memory>
#include "SDL.h"

#include "CDrawableStar.h"
#include "CPhysics.h"
#include "C2DVector.h"
#include "IScene.h"
#include "CEntityGalaxy.h"
#include "CDrawableButton.h"
#include "CEntityButton.h"
#include "CMoveableButton.h"

class CWorld;

/**
* CSceneMainMenu defines the scene with the main menu.
*/
class CSceneMainMenu: public IScene
{
public:
	CSceneMainMenu( SDL_Surface* screen_, CWorld& world_ );
	CSceneMainMenu( const CSceneMainMenu& other_ );
	CSceneMainMenu& operator=( const CSceneMainMenu& rhs );

	virtual void Init();
};

#endif
