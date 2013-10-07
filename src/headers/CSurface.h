/**
* RAII class for images
*/
#ifndef CSURFACE_H
#define CSURFACE_H

#include "SDL.h"
#include "SDL_image.h"
#include <string>
using namespace std;

class CSurface
{
public:
	explicit CSurface( string filename ) : m_surface( load_image( filename ) ){}
	~CSurface();
	SDL_Surface* GetSurface() const{ return m_surface; }
	void ApplySurface( int x, int y, SDL_Surface* destination );
	void Destroy();

private:
	CSurface( const CSurface& );
	CSurface& operator=( const CSurface& );

	SDL_Surface* load_image( string filename );
	SDL_Surface* m_surface;
};

#endif