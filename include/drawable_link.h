#ifndef CDRAWABLELINK_H
#define CDRAWABLELINK_H

#include "SDL.h"
#include "drawable_base.h"
#include "vec2.h"

/**
 * drawable_link defines how the cloth links are drawn on screen.
 */
class drawable_link : public drawable_base
{
public:
  explicit drawable_link(SDL_Renderer* renderer_);

  virtual void draw(const vec2& pos_, const vec2& origin_) const;

private:
  virtual std::unique_ptr<drawable_base> do_clone() const;
};

#endif
