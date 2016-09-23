#include <memory>

#include "CMoveableParticle.h"
#include "IMoveable.h"
#include "C2DVector.h"

CMoveableParticle::CMoveableParticle() : IMoveable(0.0f, 0.0f)
{
}
CMoveableParticle::CMoveableParticle(float x_, float y_) : IMoveable(x_, y_)
{
}
CMoveableParticle::CMoveableParticle(const C2DVector& initial_pos_)
    : IMoveable(initial_pos_)
{
}

CMoveableParticle::CMoveableParticle(const CMoveableParticle& other_)
    : IMoveable(other_)
{
}

CMoveableParticle& CMoveableParticle::operator=(const CMoveableParticle& other_)
{
    if (&other_ != this)
    {
        IMoveable::operator=(other_);
    }
    return *this;
}

std::unique_ptr<IMoveable> CMoveableParticle::DoClone() const
{
    return std::unique_ptr<IMoveable>(new CMoveableParticle(*this));
}

bool CMoveableParticle::IsHit(const C2DVector& coords_) const
{
    // check if coords_ is in within the bounding box
    bool check_x = coords_.x < (pos.x + boundingbox_half_side) &&
                   coords_.x > (pos.x - boundingbox_half_side);
    bool check_y = coords_.y < (pos.y + boundingbox_half_side) &&
                   coords_.y > (pos.y - boundingbox_half_side);

    return check_x && check_y;
}
