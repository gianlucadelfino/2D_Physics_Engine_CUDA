#include <memory>

#include "moveable_base.h"
#include "moveable_particle.h"
#include "vec2.h"

moveable_particle::moveable_particle() : moveable_base(0.0f, 0.0f) {}
moveable_particle::moveable_particle(float x_, float y_) : moveable_base(x_, y_) {}
moveable_particle::moveable_particle(const vec2& initial_pos_) : moveable_base(initial_pos_) {}

moveable_particle::moveable_particle(const moveable_particle& other_) : moveable_base(other_) {}

moveable_particle& moveable_particle::operator=(const moveable_particle& other_)
{
  if (&other_ != this)
  {
    moveable_base::operator=(other_);
  }
  return *this;
}

std::unique_ptr<moveable_base> moveable_particle::do_clone() const
{
  return std::make_unique<moveable_particle>(*this);
}

bool moveable_particle::is_hit(const vec2& coords_) const
{
  // check if coords_ is in within the bounding box
  const bool check_x =
      coords_.x < (pos.x + boundingbox_half_side) && coords_.x > (pos.x - boundingbox_half_side);
  const bool check_y =
      coords_.y < (pos.y + boundingbox_half_side) && coords_.y > (pos.y - boundingbox_half_side);

  return check_x && check_y;
}
