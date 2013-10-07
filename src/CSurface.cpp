#include "CSurface.h"
#include "SDL.h"
#include "SDL_image.h"
#include <string>

#include <iostream>
using namespace std;

CSurface::~CSurface()
{
	Destroy();
}

void CSurface::Destroy()
{
	SDL_FreeSurface( m_surface );
	m_surface = NULL;
}

void CSurface::ApplySurface( int x, int y, SDL_Surface* destination )
{
	SDL_Rect offset;
	offset.x = x;
	offset.y = y;

	SDL_BlitSurface( m_surface, NULL, destination, &offset );
}

SDL_Surface* CSurface::load_image( string filename )
{
	SDL_Surface* loadedImage = NULL;
	SDL_Surface* optimizedImage = NULL;

	loadedImage = IMG_Load( filename.c_str() );

	if( loadedImage )
	{
		//create optimized version
		optimizedImage = SDL_DisplayFormatAlpha( loadedImage );

		//free old memory
		SDL_FreeSurface( loadedImage );
	}else
	{
		cerr << "Could not load " << filename << endl;
	}

	return optimizedImage;
}