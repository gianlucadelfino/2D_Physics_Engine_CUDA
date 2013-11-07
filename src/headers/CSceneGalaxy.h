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
	CSceneGalaxy( SDL_Surface* screen_, CWorld& world_, bool use_CUDA_, unsigned int stars_num_ );
	virtual void Init();

	~CSceneGalaxy();

private:
	//TODO: define proper copy constructor & assignment
	CSceneGalaxy( const CSceneGalaxy& );
	CSceneGalaxy& operator=( const CSceneGalaxy& );

	bool m_using_CUDA;
	unsigned int m_stars_num;
	std::shared_ptr< CFont > m_font;
	bool m_CUDA_capable_device_present;
};

#endif
