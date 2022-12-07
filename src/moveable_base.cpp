#include <cassert>
#include <memory>

#include "moveable_base.h"
#include "vec2.h"

moveable_base::moveable_base(float x_, float y_) : pos(x_, y_), prev_pos(x_, y_), orientation(0, 0)
{
}

moveable_base::moveable_base(const vec2& pos_) : pos(pos_), prev_pos(pos_), orientation(0, 0) {}
moveable_base::moveable_base(const vec2& pos_, const vec2& old_pos_, const vec2& orientation_)
    : pos(pos_), prev_pos(old_pos_), orientation(orientation_)
{
}

// copy constructor and assignment operator
moveable_base::moveable_base(const moveable_base& other_)
    : pos(other_.pos), prev_pos(other_.prev_pos), orientation(other_.orientation)
{
}

moveable_base& moveable_base::operator=(const moveable_base& other_)
{
  if (&other_ != this)
  {
    pos = other_.pos;
    prev_pos = other_.prev_pos;
    orientation = other_.orientation;
  }
  return *this;
}

/*
 * clone is needed to create a deep copy when moveable_base is used as pimpl
 */
std::unique_ptr<moveable_base> moveable_base::clone() const
{
  std::unique_ptr<moveable_base> clone(do_clone());
  // lets check that the derived class actually implemented clone and it does
  // not come from a parent
  assert(typeid(*clone) == typeid(*this) && "do_clone incorrectly overridden");
  return clone;
}

/*
 * boost offsets the particle which makes it gain speed
 */
void moveable_base::boost(const vec2& new_position_) { pos = new_position_; }

/*
 * reposition moves the particle and sets its speed to zero
 */
void moveable_base::reposition(const vec2& new_position_)
{
  pos = new_position_;
  prev_pos = new_position_;
}

/*
 * translate moves particle and its velocity vector by a shift vector
 */
void moveable_base::translate(const vec2& shift_)
{
  pos += shift_;
  prev_pos += shift_;
}

// need this to use in copy constructors!

moveable_base::~moveable_base() {}

void moveable_base::impose_constraints()
{
  if (_constraint)
  {
    // remind that constraint origin is a pointer because the constraint can
    // move ( e.g. mouse )
    pos = *_constraint;
  }
}

void moveable_base::set_constraint(std::shared_ptr<vec2> constrainted_pos_)
{
  _constraint = constrainted_pos_;
}

void moveable_base::unset_constraints() { _constraint.reset(); }
