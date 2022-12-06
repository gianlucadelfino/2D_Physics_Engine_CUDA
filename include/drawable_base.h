#ifndef IDRAWABLE_H
#define IDRAWABLE_H

#include "SDL.h"
#include "surface_handler.h"
#include "vec2.h"
#include <memory>

/**
 * drawable_base defines the interface for everything that needs to be drawn on
 * screen.
 */
class drawable_base
{
public:
  drawable_base() = default;
  // copy constructor and assignment operator
  drawable_base(const drawable_base& other_);
  drawable_base& operator=(const drawable_base& other_);

  std::unique_ptr<drawable_base> clone() const; // non virtual to check that it
                                                // has been overridden (see "c++
                                                // coding standards" elem 54)

  virtual void draw(SDL_Renderer* renderer_,
                    const vec2& position_,
                    const vec2& orientation_) const = 0;

  virtual ~drawable_base();

  void set_size(const vec2& _dimensions);
  void set_scale(float scale_);

protected:
  vec2 _dimensions;
  float _scale{};

private:
  virtual std::unique_ptr<drawable_base> do_clone() const = 0;
};

#endif
