#ifndef CMOVEABLEBUTTON_H
#define CMOVEABLEBUTTON_H

#include "moveable_base.h"
#include "vec2.h"
#include <memory>

/**
 * moveable_button defines the moveable_base functions for a UI button.
 */
class moveable_button : public moveable_base
{
public:
  moveable_button(const vec2& initial_pos_, const vec2& size_);

  moveable_button(const moveable_button& other_);
  moveable_button& operator=(const moveable_button& other_);

  virtual bool is_hit(const vec2& coords_) const;

private:
  virtual std::unique_ptr<moveable_base> do_clone() const;

  vec2 _size;
};
#endif
