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

class CSceneMainMenu: public IScene
{
public:
	CSceneMainMenu( SDL_Surface* screen_, CWorld& world_ );
	virtual void Init();

private:
	//TODO: define proper copy constructor & assignment
	CSceneMainMenu( const CSceneMainMenu& );
	CSceneMainMenu& operator=( const CSceneMainMenu& );
};

#endif
