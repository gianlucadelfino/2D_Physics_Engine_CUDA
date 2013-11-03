#include <memory>
#include <vector>
#include <iostream>
#include "SDL.h"
#include "SDL_image.h"
#include "CSurface.h"

#include "CWorld.h"

#include "CEntityParticle.h"
#include "CEntityCloth.h"
#include "CEntityGalaxy.h"
#include "CMoveableParticle.h"
#include "CDrawableImage.h"
#include "CDrawableLink.h"
#include "CDrawableStar.h"

const unsigned int SCREEN_WIDTH = 1280;
const unsigned int SCREEN_HEIGHT = 720;
const unsigned int SCREEN_BPP = 16;
const unsigned int MS_PER_UPDATE = 32; // 30Hz

enum Simulation { Cloth, Galaxy };

//TODO: implement a button to dinamically swap between simulations.
//For now uncomment the wanted one
//#define CLOTH_SIMULATION
#define GALAXY_SIMULATION

int main ( int argc, char **argv )
{
	//init SDL
	if( !SDL_Init( SDL_INIT_EVERYTHING ) )
	{
		std::cerr << "Could NOT initialize SDL.." << std::endl;
	}

	SDL_WM_SetCaption( "2D Physics Engine. Gianluca.Delfino@gmail.com", NULL );

	SDL_Surface* screen( SDL_SetVideoMode( SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_BPP, SDL_SWSURFACE ) );
	if ( !screen )
	{
		std::cerr << "No screen." << std::endl;
		exit(EXIT_FAILURE);
	}

	SDL_Rect screen_rect;
	screen_rect.x = 0;
	screen_rect.y = 0;
	screen_rect.w = static_cast<Uint16>(screen->w);
	screen_rect.h = static_cast<Uint16>(screen->h);
	Uint32 color_white = SDL_MapRGB( screen->format, 255, 255, 255);
	Uint32 color_black = SDL_MapRGB( screen->format, 0	, 0, 0);

	CWorld world( screen );

	//instantiate entities container
	vector< shared_ptr< IEntity > > entities;

#ifdef CLOTH_SIMULATION
	//build the cloth
	unique_ptr<CPhysics> phys_seam( new CPhysics( 1.0f ) );
	unique_ptr<IMoveable> seam_moveable( new CMoveableParticle(50.0f, 50.0f) );
	unique_ptr<CDrawableLink> cloth_drawable( new CDrawableLink( screen ) );

	CEntityParticle seam( 1, move(seam_moveable), NULL, move(phys_seam) );
	shared_ptr< IEntity > cloth( new CEntityCloth( 1, C2DVector( 300.0f,20.0f), move( cloth_drawable ), seam, 40 ) ); // id, pos, IDrawable, seam prototype, size
	//entities.push_back( cloth );
#endif

#ifdef GALAXY_SIMULATION
	//build prototype star for the galaxy
	unique_ptr<CDrawableStar> star_drawable( new CDrawableStar( screen ) );
	unique_ptr<CPhysics> star_physics( new CPhysics( 1.0f, C2DVector( 0.0f, 0.0f ) ) );
	star_physics->SetGravity( C2DVector( 0.0f, 0.0f ) );
	unique_ptr<CMoveableParticle> star_moveable( new CMoveableParticle() );
	CEntityParticle star( 1, move(star_moveable), move(star_drawable), move(star_physics));

	//create galaxy
#ifdef USE_CUDA
	shared_ptr<IEntity> galaxy( new CEntityGalaxy( 2, C2DVector( 600.0f, 300.0f), star, 100, 500, true ) ); // id, pos, star prototype, stars number (max 9216), size
#else
	shared_ptr<IEntity> galaxy( new CEntityGalaxy( 2, C2DVector( 600.0f, 300.0f), star, 1024 , 500, false ) ); // id, pos, star, stars number, size
#endif
	//entities.push_back( galaxy );
#endif

	bool quit = false;
	SDL_Event event;

	//keep track of the lag if the game is slowing down (update only, without rendering, to keep up)
	int lag = 0;
	int prev_ticks = SDL_GetTicks();

	//need the mouse coords
	shared_ptr< C2DVector > mouse_coords( new C2DVector() );

	while( !quit )
	{
		//update dt
		int cur_ticks = SDL_GetTicks();
		int dt = cur_ticks - prev_ticks;
		prev_ticks = cur_ticks;

		//add the time it took last frame to the lag, then we know how many times
		//we have to cycle over MS_PER_UPDATE to catch up with real time
		lag += dt;

		while ( lag >= MS_PER_UPDATE )
		{
			//event handling
			while( SDL_PollEvent( &event ) )
			{
				world.HandleEvent( event );
				switch( event.type )
				{
				case SDL_QUIT:
					quit = true;
					break;
					//case SDL_KEYDOWN:
					//	//phyisicsON = false;
					//	break;
					//case SDL_MOUSEMOTION:
					//	mouse_coords->x = static_cast<float>(event.motion.x);
					//	mouse_coords->y = static_cast<float>(event.motion.y);
					//	break;
					//case SDL_MOUSEBUTTONDOWN:
					//	for( vector< shared_ptr< IEntity > >::iterator it = entities.begin(); it != entities.end(); ++it )
					//	{
					//		(*it)->HandleMouseButtonDown( mouse_coords );
					//	}
					//	break;
					//case SDL_MOUSEBUTTONUP:
					//	for( vector< shared_ptr< IEntity > >::iterator it = entities.begin(); it != entities.end(); ++it )
					//	{
					//		(*it)->HandleMouseButtonUp( mouse_coords );
					//	}
					//	break;
				}
			}
			world.Update( static_cast<float>(MS_PER_UPDATE)*0.01f );
			for( vector< shared_ptr< IEntity > >::iterator it = entities.begin(); it != entities.end(); ++it )
			{
				//Arbitrary factor in front of the time, It would be better to use seconds ( so it should be "*0.001").
				//But then is too slow because we use 1px = 1mt
				(*it)->Update( C2DVector( 0.0f, 0.0f ), static_cast<float>(MS_PER_UPDATE)*0.01f );
			}

			lag -= MS_PER_UPDATE;
		}

		//draw stuff, background first!
#ifdef CLOTH_SIMULATION
		//SDL_FillRect( screen, &screen_rect, color_white );
#endif
#ifdef GALAXY_SIMULATION
		//SDL_FillRect( screen, &screen_rect, color_black );
#endif
		world.Draw();
		for( vector< shared_ptr< IEntity > >::iterator it = entities.begin(); it != entities.end(); ++it )
		{
			(*it)->Draw();
		}

		//update Screen
		SDL_Flip( screen );

		//sleep if it took less than MS_PER_UPDATE
		int delay = cur_ticks + MS_PER_UPDATE - SDL_GetTicks();
		if ( delay > 0 ) SDL_Delay( delay );
	}

	SDL_Quit();

#ifdef USE_CUDA
	cudaDeviceReset();
#endif
	return 0;
}