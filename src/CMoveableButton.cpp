#include <memory>

#include "C2DVector.h"
#include "CMoveableButton.h"
#include "IMoveable.h"

CMoveableButton::CMoveableButton(const C2DVector& initial_pos_, const C2DVector& size_)
    : IMoveable(initial_pos_), _size(size_)
{
}

CMoveableButton::CMoveableButton(const CMoveableButton& other_) : IMoveable(other_) {}

CMoveableButton& CMoveableButton::operator=(const CMoveableButton& other_)
{
  if (&other_ != this)
  {
    IMoveable::operator=(other_);
  }
  return *this;
}
std::unique_ptr<IMoveable> CMoveableButton::DoClone() const
{
  return std::make_unique<CMoveableButton>(*this);
}

bool CMoveableButton::IsHit(const C2DVector& coords_) const
{
  // check if coords_ is in within the bounding box
  const bool check_x = coords_.x < (pos.x + _size.x) && coords_.x > pos.x;
  const bool check_y = coords_.y < (pos.y + _size.y) && coords_.y > pos.y;

  return check_x && check_y;
}
