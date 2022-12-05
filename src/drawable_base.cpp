#include <cassert>

#include "drawable_base.h"

drawable_base::drawable_base(SDL_Renderer* renderer_)
    : _renderer(renderer_), _dimensions(), _scale(1)
{
}

drawable_base::drawable_base(const drawable_base& other_)
    : _renderer(other_._renderer), _dimensions(other_._dimensions), _scale(other_._scale)
{
}

drawable_base& drawable_base::operator=(const drawable_base& other_)
{
  if (this != &other_)
  {
    _renderer = other_._renderer;
    _dimensions = other_._dimensions;
    _scale = other_._scale;
  }
  return *this;
}

std::unique_ptr<drawable_base> drawable_base::clone() const
{
  std::unique_ptr<drawable_base> clone(do_clone());
  // lets check that the derived class actually implemented clone and it does
  // not come from a parent
  assert(typeid(*clone) == typeid(*this) && "do_clone incorrectly overridden");
  return clone;
}

void drawable_base::set_size(const vec2& dimensions_) { _dimensions = dimensions_; }
void drawable_base::set_scale(float scale_) { _scale = scale_; }

drawable_base::~drawable_base() {}
