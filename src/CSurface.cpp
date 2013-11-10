#include <stdexcept>
#include <string>

#include "CSurface.h"
#include "SDL.h"
#include "SDL_image.h"

CSurface::CSurface() : mp_surface( NULL ){}
CSurface::CSurface( SDL_Surface* surface_ ) : mp_surface( surface_ ){}

CSurface::~CSurface()
{
	this->Destroy();
}

void CSurface::Destroy()
{
	SDL_FreeSurface( this->mp_surface );
	this->mp_surface = NULL;
}

void CSurface::ApplySurface( int x, int y, SDL_Surface* destination )
{
	SDL_Rect offset;
	offset.x = static_cast<Sint16>(x);
	offset.y = static_cast<Sint16>(y);

	SDL_BlitSurface( this->mp_surface, NULL, destination, &offset );
}

void CSurface::LoadSurface( SDL_Surface* new_surface_ )
{
	this->Destroy();
	this->mp_surface = new_surface_;
}

void CSurface::load_image( std::string filename )
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
		this->Destroy();
		this->mp_surface = optimizedImage;
	}else
	{
		throw std::runtime_error("Could not load surface file: " + filename );
	}
}