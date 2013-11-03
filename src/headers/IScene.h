#ifndef ISCENE_H
#define ISCENE_H

#include <vector>
#include <memory>
#include <string>
#include "IEntity.h"
#include "C2DVector.h"
#include "SDL.h"

class CWorld;

class IScene
{
public:
	IScene( SDL_Surface* screen_, CWorld& world_ );

	virtual void Init();

	virtual void Update( float dt );
	virtual void HandleEvent( CWorld& world_, const SDL_Event& event_ );
	void Draw() const;

	void AddUIelement( std::unique_ptr< IEntity > UI_element_ );

	virtual ~IScene();
protected:
	SDL_Surface* mp_screen;
	CWorld& mr_world;
	std::vector< std::unique_ptr< IEntity > > m_entities;
	std::vector< std::unique_ptr< IEntity > > m_UI_elements;
private:
	//TODO: define assignment and copy constructor
	IScene( const IScene& );
	IScene& operator=( const IScene& );
};

#endif
