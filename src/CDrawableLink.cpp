#include "CDrawableLink.h"
#include "SDL.h"
// #include "SDL_gfxPrimitives.h"
#include "IDrawable.h"
#include <memory>

CDrawableLink::CDrawableLink(SDL_Renderer* renderer_) : IDrawable(renderer_) {}

std::unique_ptr<IDrawable> CDrawableLink::DoClone() const
{
  std::unique_ptr<CDrawableLink> clone = std::make_unique<CDrawableLink>(_renderer);
  clone->_scale = _scale;
  clone->_dimensions = _dimensions;

  return clone;
}

void CDrawableLink::Draw(const C2DVector& pos_, const C2DVector& origin_) const
{
  // draw line
  SDL_SetRenderDrawColor(_renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderDrawLine(_renderer, origin_.x, origin_.y, pos_.x, pos_.y);
}
