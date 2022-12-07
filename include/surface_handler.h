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
  surface_handler() = default;
  explicit surface_handler(SDL_Surface* surface_);
  ~surface_handler();
  SDL_Surface* get() const { return _surface; }
  void load_surface(SDL_Surface* new_surface_);

  surface_handler(const surface_handler&) = delete;
  surface_handler& operator=(const surface_handler&) = delete;

private:
  void destroy();

  SDL_Surface* _surface{};
};

#endif
