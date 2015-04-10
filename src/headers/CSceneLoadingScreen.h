#ifndef CSCENELOADINGSCREEN_H
#define CSCENELOADINGSCREEN_H

#include "SDL.h"

#include "IScene.h"

class CWorld;

/**
* CSceneLoadingScreen defines the scene for the Loading screen.
*/
class CSceneLoadingScreen: public IScene
{
public:
    CSceneLoadingScreen( SDL_Surface* screen_, CWorld& world_ );
    CSceneLoadingScreen( const CSceneLoadingScreen& other_ );
    CSceneLoadingScreen& operator=( const CSceneLoadingScreen& rhs );

    virtual void Init();
};

#endif
