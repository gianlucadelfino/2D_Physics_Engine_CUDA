#include "drawable_link.h"
#include "SDL.h"
// #include "SDL_gfxPrimitives.h"
#include "drawable_base.h"
#include <memory>

std::unique_ptr<drawable_base> drawable_link::do_clone() const
{
  std::unique_ptr<drawable_link> clone = std::make_unique<drawable_link>();
  clone->_scale = _scale;
  clone->_dimensions = _dimensions;

  return clone;
}

void drawable_link::draw(SDL_Renderer* renderer_, const vec2& pos_, const vec2& origin_) const
{
  // draw line
  SDL_SetRenderDrawColor(renderer_, 0, 0, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderDrawLine(renderer_, origin_.x, origin_.y, pos_.x, pos_.y);
}
