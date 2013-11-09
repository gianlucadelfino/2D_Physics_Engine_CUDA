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

/**
* CSceneCloth defines the scene with the moving cloth.
*/
class CSceneCloth: public IScene
{
public:
	CSceneCloth( SDL_Surface* screen_, CWorld& world_ );
	virtual void Init();

private:
	//TODO: define proper copy constructor & assignment
	CSceneCloth( const CSceneCloth& );
	CSceneCloth& operator=( const CSceneCloth& );
	std::shared_ptr< CFont > m_font;
};

#endif
