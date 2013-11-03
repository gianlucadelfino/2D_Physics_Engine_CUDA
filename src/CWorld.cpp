#include <iostream>
#include "SDL_TTF.h"
#include "CWorld.h"
#include "CSceneGalaxy.h"

CWorld::CWorld( SDL_Surface* screen_ ):
	mp_screen( screen_ )
{
	if ( !mp_screen )
	{
		std::cerr<< "Could not initalize World without a screen" << std::endl;
		exit(EXIT_FAILURE);
	}

	//init SDL_ttf
	if ( TTF_Init() == -1 )
	{
		std::cerr << "Could NOT initialize SDL_ttf.." << std::endl;
	}

	std::unique_ptr< CSceneGalaxy > galaxy_scene( new CSceneGalaxy( this->mp_screen,  *this, true ) ); //start in CUDA mode!

	this->mp_cur_scene = std::move( galaxy_scene );
	this->mp_cur_scene->Init();
}

void CWorld::Update( float dt ) const
{
	if ( this->mp_cur_scene )
		this->mp_cur_scene->Update( dt);
}

void CWorld::HandleEvent( const SDL_Event& event_ )
{
	if ( this->mp_cur_scene )
		this->mp_cur_scene->HandleEvent( *this, event_ );
}

void CWorld::Draw() const
{
	if ( this->mp_cur_scene )
		this->mp_cur_scene->Draw();
}

void CWorld::ChangeScene( std::unique_ptr< IScene > new_scene_ )
{
	this->mp_cur_scene = std::move( new_scene_ );
	this->mp_cur_scene->Init();
}

CWorld::~CWorld()
{
	//manually reset the scene pointer(calling its destruction), before we
	//deallocate SDL resources
	this->mp_cur_scene.reset(NULL);

	//NOW we can free resources
	TTF_Quit();
}