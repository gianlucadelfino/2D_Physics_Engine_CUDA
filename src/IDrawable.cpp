#include <cassert>

#include "IDrawable.h"

IDrawable::IDrawable(SDL_Renderer* renderer_)
    : _renderer(renderer_), _dimensions(), _scale(1)
{
}

IDrawable::IDrawable(const IDrawable& other_)
    : _renderer(other_._renderer),
      _dimensions(other_._dimensions),
      _scale(other_._scale)
{
}

IDrawable& IDrawable::operator=(const IDrawable& other_)
{
    if (this != &other_)
    {
        _renderer = other_._renderer;
        _dimensions = other_._dimensions;
        _scale = other_._scale;
    }
    return *this;
}

std::unique_ptr<IDrawable> IDrawable::Clone() const
{
    std::unique_ptr<IDrawable> clone(DoClone());
    // lets check that the derived class actually implemented clone and it does
    // not come from a parent
    assert(typeid(*clone) == typeid(*this) && "DoClone incorrectly overridden");
    return clone;
}

void IDrawable::SetSize(const C2DVector& dimensions_)
{
    _dimensions = dimensions_;
}
void IDrawable::SetScale(float scale_)
{
    _scale = scale_;
}

IDrawable::~IDrawable()
{
}
