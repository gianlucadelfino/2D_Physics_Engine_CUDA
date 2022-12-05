/**
 * RAII class for images
 */
#ifndef CSURFACE_H
#define CSURFACE_H

#include "SDL.h"
#include "SDL_image.h"
#include <string>

/**
 * surface_handler is a RAII class to handle an SDL surface.
 */
class surface_handler
{
public:
  surface_handler();
  explicit surface_handler(SDL_Surface* surface_);
  ~surface_handler();
  SDL_Surface* GetSurface() const { return _surface; }
  void load_surface(SDL_Surface* new_surface_);
  void destroy();

private:
  surface_handler(const surface_handler&);
  surface_handler& operator=(const surface_handler&);

  SDL_Surface* _surface;
};

#endif
