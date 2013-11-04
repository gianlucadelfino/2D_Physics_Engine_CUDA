#ifndef ISCENE_H
#define ISCENE_H

#include <vector>
#include <memory>
#include <string>
#include "CEntity.h"
#include "C2DVector.h"
#include "SDL.h"

class CWorld;

class IScene
{
public:
	IScene( SDL_Surface* screen_, CWorld& world_, Uint32 background_color_ );

	virtual void Init();

	virtual void Update( float dt );
	virtual void HandleEvent( CWorld& world_, const SDL_Event& event_ );
	void Draw() const;

	virtual ~IScene();
protected:
	SDL_Surface* mp_screen;
	CWorld& mr_world;
	Uint32 m_background_color;
	std::shared_ptr< C2DVector > m_mouse_coords;
	std::vector< std::unique_ptr< CEntity > > m_entities;
	std::vector< std::unique_ptr< CEntity > > m_UI_elements;
private:
	//TODO: define assignment and copy constructor
	IScene( const IScene& );
	IScene& operator=( const IScene& );
};

#endif
