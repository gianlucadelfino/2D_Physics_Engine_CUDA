#ifndef CDRAWABLESTAR_H
#define CDRAWABLESTAR_H

#include "C2DVector.h"
#include "IDrawable.h"
#include "SDL.h"

/**
 * CDrawableStar defines how stars are drawn on screen.
 */
class CDrawableStar : public IDrawable
{
public:
  explicit CDrawableStar(SDL_Renderer* renderer_);

  virtual void Draw(const C2DVector& pos_, const C2DVector& orientation_) const;

private:
  virtual std::unique_ptr<IDrawable> DoClone() const;
};

#endif
