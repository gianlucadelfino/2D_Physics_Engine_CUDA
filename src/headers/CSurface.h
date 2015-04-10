/**
* RAII class for images
*/
#ifndef CSURFACE_H
#define CSURFACE_H

#include "SDL.h"
#include "SDL_image.h"
#include <string>
using namespace std;

/**
* CSurface is a RAII class to handle an SDL surface.
*/
class CSurface
{
public:
    CSurface();
    explicit CSurface( SDL_Surface* surface_ );
    ~CSurface();
    SDL_Surface* GetSurface() const{ return mp_surface; }
    void LoadSurface( SDL_Surface* new_surface_ );
    void ApplySurface( int x, int y, SDL_Surface* destination );
    void load_image( string filename );
    void Destroy();

private:
    CSurface( const CSurface& );
    CSurface& operator=( const CSurface& );

    SDL_Surface* mp_surface;
};

#endif
