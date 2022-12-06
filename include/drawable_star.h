#ifndef CDRAWABLESTAR_H
#define CDRAWABLESTAR_H

#include "SDL.h"
#include "drawable_base.h"
#include "vec2.h"

/**
 * drawable_star defines how stars are drawn on screen.
 */
class drawable_star : public drawable_base
{
public:
  drawable_star() = default;

  virtual void draw(SDL_Renderer* renderer_,
                    const vec2& pos_,
                    const vec2& orientation_) const override;

private:
  virtual std::unique_ptr<drawable_base> do_clone() const override;
};

#endif
