#include <memory>

#include "moveable_base.h"
#include "moveable_button.h"
#include "vec2.h"

moveable_button::moveable_button(const vec2& initial_pos_, const vec2& size_)
    : moveable_base(initial_pos_), _size(size_)
{
}

moveable_button::moveable_button(const moveable_button& other_) : moveable_base(other_) {}

moveable_button& moveable_button::operator=(const moveable_button& other_)
{
  if (&other_ != this)
  {
    moveable_base::operator=(other_);
  }
  return *this;
}
std::unique_ptr<moveable_base> moveable_button::do_clone() const
{
  return std::make_unique<moveable_button>(*this);
}

bool moveable_button::is_hit(const vec2& coords_) const
{
  // check if coords_ is in within the bounding box
  const bool check_x = coords_.x < (pos.x + _size.x) && coords_.x > pos.x;
  const bool check_y = coords_.y < (pos.y + _size.y) && coords_.y > pos.y;

  return check_x && check_y;
}
