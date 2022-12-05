/**
* RAII class for images
*/
#ifndef CSURFACE_H
#define CSURFACE_H

#include "SDL.h"
#include "SDL_image.h"
#include <string>

/**
* CSurface is a RAII class to handle an SDL surface.
*/
class CSurface
{
public:
    CSurface();
    explicit CSurface(SDL_Surface* surface_);
    ~CSurface();
    SDL_Surface* GetSurface() const
    {
        return _surface;
    }
    void LoadSurface(SDL_Surface* new_surface_);
    void ApplySurface(int x, int y, SDL_Surface* destination);
    void Destroy();

private:
    CSurface(const CSurface&);
    CSurface& operator=(const CSurface&);

    SDL_Surface* _surface;
};

#endif
