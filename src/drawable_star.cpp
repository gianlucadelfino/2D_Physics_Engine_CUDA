#include "SDL.h"
// #include "SDL_gfxPrimitives.h"

#include "drawable_base.h"
#include "drawable_star.h"
#include "vec2.h"
#include <SDL_rect.h>

std::unique_ptr<drawable_base> drawable_star::do_clone() const
{
  std::unique_ptr<drawable_star> clone = std::make_unique<drawable_star>();
  clone->_scale = _scale;
  clone->_dimensions = _dimensions;

  return clone;
}

void drawable_star::draw(SDL_Renderer* renderer_,
                         const vec2& pos_,
                         const vec2& /*orientation_*/) const
{
  const SDL_Rect star{static_cast<int>(pos_.x),
                      static_cast<int>(pos_.y),
                      static_cast<int>(_scale * 0.4f),
                      static_cast<int>(_scale * 0.4f)};
  SDL_SetRenderDrawColor(renderer_, 255, 255, 255, 255);
  SDL_RenderFillRect(renderer_, &star);
}
