#include <memory>

#include "C2DVector.h"
#include "CEntity.h"
#include "IDrawable.h"
#include "IMoveable.h"

CEntity::CEntity(unsigned int id_,
                 std::unique_ptr<IMoveable> moveable_,
                 std::unique_ptr<IDrawable> drawable_)
    : _id(id_), _moveable(std::move(moveable_)), _drawable(std::move(drawable_))
{
}

CEntity::CEntity(const CEntity& other_) : _id(other_._id)
{
  // I need to deep copy the pointee, otherwise i am using the same IMovable!
  if (other_._moveable)
    _moveable = other_._moveable->Clone();
  else
    _moveable = nullptr;

  // I need to deep copy the pointee, otherwise i am using the same IDrawable!
  if (other_._drawable)
    _drawable = other_._drawable->Clone();
  else
    _drawable = nullptr;
}

CEntity& CEntity::operator=(const CEntity& rhs)
{
  if (this != &rhs)
  {
    _id = rhs._id;
    if (rhs._moveable)
    {
      // I need to deep copy the pointee, otherwise i am using the same
      // IMovable!
      _moveable = rhs._moveable->Clone();
    }
    else
      _moveable = nullptr;

    if (rhs._drawable)
    {
      // I need to deep copy the pointee, otherwise i am using the same
      // IDrawable
      _drawable = rhs._drawable->Clone();
    }
    else
    {
      _drawable = nullptr;
    }
  }
  return *this;
}

CEntity::~CEntity() {}

bool CEntity::IsHit(const C2DVector& coords_) const
{
  if (_moveable)
    return _moveable->IsHit(coords_);
  else
    return false;
}

void CEntity::Draw() const
{
  if (_drawable && _moveable)
    _drawable->Draw(_moveable->pos, _moveable->orientation);
}
