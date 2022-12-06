#include <memory>

#include "drawable_base.h"
#include "entity_base.h"
#include "moveable_base.h"
#include "vec2.h"

entity_base::entity_base(unsigned int id_,
                         std::unique_ptr<moveable_base> moveable_,
                         std::unique_ptr<drawable_base> drawable_)
    : _id(id_), _moveable(std::move(moveable_)), _drawable(std::move(drawable_))
{
}

entity_base::entity_base(const entity_base& other_) : _id(other_._id)
{
  // I need to deep copy the pointee, otherwise i am using the same IMovable!
  if (other_._moveable)
    _moveable = other_._moveable->clone();
  else
    _moveable = nullptr;

  // I need to deep copy the pointee, otherwise i am using the same drawable_base!
  if (other_._drawable)
    _drawable = other_._drawable->clone();
  else
    _drawable = nullptr;
}

entity_base& entity_base::operator=(const entity_base& rhs)
{
  if (this != &rhs)
  {
    _id = rhs._id;
    if (rhs._moveable)
    {
      // I need to deep copy the pointee, otherwise i am using the same
      // IMovable!
      _moveable = rhs._moveable->clone();
    }
    else
      _moveable = nullptr;

    if (rhs._drawable)
    {
      // I need to deep copy the pointee, otherwise i am using the same
      // drawable_base
      _drawable = rhs._drawable->clone();
    }
    else
    {
      _drawable = nullptr;
    }
  }
  return *this;
}

entity_base::~entity_base() {}

bool entity_base::is_hit(const vec2& coords_) const
{
  if (_moveable)
    return _moveable->is_hit(coords_);
  else
    return false;
}

void entity_base::draw(SDL_Renderer* renderer_) const
{
  if (_drawable && _moveable)
  {
    _drawable->draw(renderer_, _moveable->pos, _moveable->orientation);
  }
}
