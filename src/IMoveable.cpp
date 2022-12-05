#include <memory>
#include <cassert>

#include "IMoveable.h"
#include "C2DVector.h"

IMoveable::IMoveable(float x_, float y_)
    : pos(x_, y_), prev_pos(x_, y_), orientation(0, 0)
{
}

IMoveable::IMoveable(const C2DVector& pos_)
    : pos(pos_), prev_pos(pos_), orientation(0, 0)
{
}
IMoveable::IMoveable(const C2DVector& pos_,
                     const C2DVector& old_pos_,
                     const C2DVector& orientation_)
    : pos(pos_), prev_pos(old_pos_), orientation(orientation_)
{
}

// copy constructor and assignment operator
IMoveable::IMoveable(const IMoveable& other_)
    : pos(other_.pos), prev_pos(other_.prev_pos), orientation(other_.orientation)
{
}

IMoveable& IMoveable::operator=(const IMoveable& other_)
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
* Clone is needed to create a deep copy when IMoveable is used as pimpl
*/
std::unique_ptr<IMoveable> IMoveable::Clone() const
{
    std::unique_ptr<IMoveable> clone(DoClone());
    // lets check that the derived class actually implemented clone and it does
    // not come from a parent
    assert(typeid(*clone) == typeid(*this) && "DoClone incorrectly overridden");
    return clone;
}

/*
* Boost offsets the particle which makes it gain speed
*/
void IMoveable::Boost(const C2DVector& new_position_)
{
    pos = new_position_;
}

/*
* Reposition moves the particle and sets its speed to zero
*/
void IMoveable::Reposition(const C2DVector& new_position_)
{
    pos = new_position_;
    prev_pos = new_position_;
}

/*
* Translate moves particle and its velocity vector by a shift vector
*/
void IMoveable::Translate(const C2DVector& shift_)
{
    pos += shift_;
    prev_pos += shift_;
}

//need this to use in copy constructors!

IMoveable::~IMoveable()
{
}

void IMoveable::ImposeConstraints()
{
    if (_constraint)
    {
        // remind that constraint origin is a pointer because the constraint can
        // move ( e.g. mouse )
        pos = *_constraint;
    }
}

void IMoveable::SetConstraint(std::shared_ptr<C2DVector> constrainted_pos_)
{
    _constraint = constrainted_pos_;
}

void IMoveable::UnsetConstraint()
{
    _constraint.reset();
}
