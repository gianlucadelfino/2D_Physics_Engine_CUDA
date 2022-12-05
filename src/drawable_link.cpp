#include "drawable_link.h"
#include "SDL.h"
// #include "SDL_gfxPrimitives.h"
#include "drawable_base.h"
#include <memory>

drawable_link::drawable_link(SDL_Renderer* renderer_) : drawable_base(renderer_) {}

std::unique_ptr<drawable_base> drawable_link::do_clone() const
{
  std::unique_ptr<drawable_link> clone = std::make_unique<drawable_link>(_renderer);
  clone->_scale = _scale;
  clone->_dimensions = _dimensions;

  return clone;
}

void drawable_link::draw(const vec2& pos_, const vec2& origin_) const
{
  // draw line
  SDL_SetRenderDrawColor(_renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderDrawLine(_renderer, origin_.x, origin_.y, pos_.x, pos_.y);
}
