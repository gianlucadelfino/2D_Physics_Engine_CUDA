#ifndef CSCENECLOTH_H
#define CSCENECLOTH_H

#include <memory>
#include "SDL.h"

#include "CPhysics.h"
#include "C2DVector.h"
#include "IScene.h"
#include "CDrawableButton.h"
#include "CEntityButton.h"
#include "CMoveableButton.h"

class CWorld;

class CSceneCloth: public IScene
{
public:
	CSceneCloth( SDL_Surface* screen_, CWorld& world_ );
	virtual void Init();

private:
	//TODO: define proper copy constructor & assignment
	CSceneCloth( const CSceneCloth& );
	CSceneCloth& operator=( const CSceneCloth& );
};

#endif
