#include <stdexcept>
#include <string>

#include "SDL.h"
#include "SDL_image.h"
#include "surface_handler.h"

surface_handler::surface_handler(SDL_Surface* surface_) : _surface(surface_) {}

surface_handler::~surface_handler() { destroy(); }

void surface_handler::destroy()
{
  SDL_FreeSurface(_surface);
  _surface = nullptr;
}

void surface_handler::load_surface(SDL_Surface* new_surface_)
{
  destroy();
  _surface = new_surface_;
}
