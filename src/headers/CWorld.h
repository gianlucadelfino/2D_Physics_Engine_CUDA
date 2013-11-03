#ifndef CWORLD_H
#define CWORLD_H

#include <memory>
#include "SDL.h"
#include "SDL_events.h"
#include "IScene.h"

#include "CSceneGalaxy.h"

class IScene;

/**
* CWorld is a Finite-State-Machine that switches between Scenes ( aka "levels" or "simulations" )
*/
class CWorld
{
public:
	CWorld( SDL_Surface* screen );

	void Update( float dt ) const;
	void HandleEvent( const SDL_Event& event_ );
	void Draw() const;

	void ChangeScene( std::unique_ptr< IScene > new_scene_ );
	~CWorld();

private:
	std::unique_ptr< IScene > mp_cur_scene;
	SDL_Surface* mp_screen;
};

#endif