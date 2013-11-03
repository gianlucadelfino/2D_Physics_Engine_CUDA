#ifndef CSCENEGALAXY_H
#define CSCENEGALAXY_H

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

class CSceneGalaxy: public IScene
{
public:
	CSceneGalaxy( SDL_Surface* screen_, CWorld& world_, bool use_CUDA_ );
	virtual void Init();

private:
	//TODO: define proper copy constructor & assignment
	CSceneGalaxy( const CSceneGalaxy& );
	CSceneGalaxy& operator=( const CSceneGalaxy& );

	bool m_use_CUDA;
};

#endif
