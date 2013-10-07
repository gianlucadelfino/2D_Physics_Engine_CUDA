#include "SDL.h"
#include "SDL_image.h"
#include "CSurface.h"
#include <memory>
#include <vector>
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

int main ( int argc, char **argv )
{
	if( !SDL_Init( SDL_INIT_EVERYTHING ) )
	{
		//error code
	}

	SDL_WM_SetCaption( "Magic Mine ", NULL );

	SDL_Surface* screen = NULL;
	screen = SDL_SetVideoMode( SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_BPP, SDL_SWSURFACE );
	if ( !screen )
	{
		cerr << "no screen" << endl;
	}

	SDL_Rect screen_rect;
	screen_rect.x = 0;
	screen_rect.y = 0;
	screen_rect.w = static_cast<Uint16>(screen->w);
	screen_rect.h = static_cast<Uint16>(screen->h);
	Uint32 color_white = SDL_MapRGB( screen->format, 255, 255, 255);
	Uint32 color_black = SDL_MapRGB( screen->format, 0	, 0, 0);

	//instatiate the pimpls
	shared_ptr<CDrawableLink> cloth_drawable( new CDrawableLink(screen) );

	//instantiate entities container
	vector< shared_ptr< IEntity > > entities;

	//build the cloth
	shared_ptr<CPhysics> phys_seam( new CPhysics( 1.0f ) );
	shared_ptr<IMoveable> seam_moveable( new CMoveableParticle(50.0f, 50.0f) );

	CEntityParticle seam( 1, seam_moveable, NULL, phys_seam );
	shared_ptr< IEntity > cloth( new CEntityCloth( 1, C2DVector( 300.0f,20.0f), cloth_drawable, seam, 10 ) ); //mass = 1, side_length = 40
	//entities.push_back( cloth );

	//build prototype star for the galaxy
	shared_ptr<CDrawableStar> star_drawable( new CDrawableStar( screen ) );
	shared_ptr<CPhysics> star_physics( new CPhysics( 1.0f, C2DVector( 0.0f, 0.0f ) ) );
	star_physics->SetGravity( C2DVector( 0.0f, 0.0f ) );
	shared_ptr<CMoveableParticle> star_moveable( new CMoveableParticle() );
	CEntityParticle star( 1, star_moveable, star_drawable, star_physics);

	//create galaxy
#ifdef CUDA
	shared_ptr<IEntity> galaxy( new CEntityGalaxy( 2, C2DVector( 600.0f, 300.0f), star, 1024*8, 500 ) ); // id, pos, star, stars number (max 9432), size
#else
	shared_ptr<IEntity> galaxy( new CEntityGalaxy( 2, C2DVector( 600.0f, 300.0f), star, 1280 , 500 ) ); // id, pos, star, stars number, size
#endif
	entities.push_back( galaxy );

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
			while( SDL_PollEvent( &event) )
			{
				switch( event.type )
				{
				case SDL_QUIT:
					quit = true;
					break;
				case SDL_KEYDOWN:
					//phyisicsON = false;
					break;
				case SDL_MOUSEMOTION:
					mouse_coords->x = static_cast<float>(event.motion.x);
					mouse_coords->y = static_cast<float>(event.motion.y);
					break;
				case SDL_MOUSEBUTTONDOWN:
					for( vector< shared_ptr< IEntity > >::iterator it = entities.begin(); it != entities.end(); ++it )
					{
						(*it)->HandleMouseButtonDown( mouse_coords );
					}
					break;
				case SDL_MOUSEBUTTONUP:
					for( vector< shared_ptr< IEntity > >::iterator it = entities.begin(); it != entities.end(); ++it )
					{
						(*it)->HandleMouseButtonUp( mouse_coords );
					}
					break;
				}
			}

			for( vector< shared_ptr< IEntity > >::iterator it = entities.begin(); it != entities.end(); ++it )
			{
				//Arbitrary factor in front of the time, It would be better to use seconds ( so it should be "*0.001").
				//But then is too slow because we use 1px = 1mt
				(*it)->Update( C2DVector( 0.0f, 0.0f ), static_cast<float>(MS_PER_UPDATE)*0.01f );
			}

			lag -= MS_PER_UPDATE;
		}

		//draw stuff
		//SDL_FillRect( screen, &screen_rect, color_white);
		SDL_FillRect( screen, &screen_rect, color_black);
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

	//delete surf_purpleGem;
	SDL_Quit();

#ifdef CUDA
	cudaDeviceReset();
#endif
	return 0;
}