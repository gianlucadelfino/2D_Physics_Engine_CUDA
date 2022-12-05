#include <stdexcept>
#include <string>

#include "CSurface.h"
#include "SDL.h"
#include "SDL_image.h"

CSurface::CSurface() : _surface(nullptr)
{
}
CSurface::CSurface(SDL_Surface* surface_) : _surface(surface_)
{
}

CSurface::~CSurface()
{
    Destroy();
}

void CSurface::Destroy()
{
    SDL_FreeSurface(_surface);
    _surface = nullptr;
}

void CSurface::ApplySurface(int x, int y, SDL_Surface* destination)
{
    SDL_Rect offset;
    offset.x = static_cast<Sint16>(x);
    offset.y = static_cast<Sint16>(y);

    SDL_BlitSurface(_surface, nullptr, destination, &offset);
}

void CSurface::LoadSurface(SDL_Surface* new_surface_)
{
    Destroy();
    _surface = new_surface_;
}
