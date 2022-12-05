#include "SDL.h"
// #include "SDL_gfxPrimitives.h"

#include "C2DVector.h"
#include "CDrawableStar.h"
#include "IDrawable.h"
#include <SDL_rect.h>

CDrawableStar::CDrawableStar(SDL_Renderer* renderer_) : IDrawable(renderer_) {}

std::unique_ptr<IDrawable> CDrawableStar::DoClone() const
{
  std::unique_ptr<CDrawableStar> clone = std::make_unique<CDrawableStar>(_renderer);
  clone->_scale = _scale;
  clone->_dimensions = _dimensions;

  return clone;
}

void CDrawableStar::Draw(const C2DVector& pos_, const C2DVector& /*orientation_*/) const
{
  const SDL_Rect star{static_cast<int>(pos_.x),
                      static_cast<int>(pos_.y),
                      static_cast<int>(_scale * 0.4f),
                      static_cast<int>(_scale * 0.4f)};
  SDL_SetRenderDrawColor(_renderer, 255, 255, 255, 255);
  SDL_RenderFillRect(_renderer, &star);
}
