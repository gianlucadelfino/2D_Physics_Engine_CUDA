#include <memory>
#include <iostream>
#include "SDL.h"
#include "CWorld.h"

const unsigned int SCREEN_WIDTH = 1280;
const unsigned int SCREEN_HEIGHT = 720;
const unsigned int SCREEN_BPP = 16;
const unsigned int MS_PER_UPDATE = 32; // 30Hz

int main ( int argc, char **argv )
{
	//init SDL
	if( SDL_Init( SDL_INIT_EVERYTHING ) ) //returns 0 if everything is ok!
	{
		std::cerr << "Could NOT initialize SDL. Error: " << SDL_GetError() << std::endl;
		exit(EXIT_FAILURE);
	}

	SDL_WM_SetCaption( "2D Physics Engine. Gianluca.Delfino@gmail.com", NULL );

	SDL_Surface* screen( SDL_SetVideoMode( SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_BPP, SDL_SWSURFACE ) );
	if ( !screen )
	{
		std::cerr << "No screen." << std::endl;
		exit(EXIT_FAILURE);
	}

	//instantiate World
	CWorld world( screen );

	bool quit = false;
	SDL_Event event;

	//keep track of the lag if the game is slowing down (update only, without rendering, to keep up)
	int lag = 0;
	int prev_ticks = SDL_GetTicks();

	//Main Loop
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
				if( event.type == SDL_QUIT )
					quit = true;
			}
			//update World (constant in front of time is arbitrary to set "distances")
			world.Update( static_cast<float>(MS_PER_UPDATE)*0.01f );

			lag -= MS_PER_UPDATE;
		}

		//draw world
		world.Draw();

		//update Screen
		SDL_Flip( screen );

		//sleep if it took less than MS_PER_UPDATE
		int delay = cur_ticks + MS_PER_UPDATE - SDL_GetTicks();
		if ( delay > 0 ) SDL_Delay( delay );
	}

	SDL_Quit();

	return 0;
}