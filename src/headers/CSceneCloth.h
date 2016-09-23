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
class CSceneCloth : public IScene
{
public:
    CSceneCloth(SDL_Surface* screen_, CWorld& world_);
    CSceneCloth(const CSceneCloth& other_);
    CSceneCloth& operator=(const CSceneCloth& rhs);

    virtual void Init();

private:
    std::shared_ptr<CFont> mp_font;
};

#endif
