#ifndef CMOVEABLEPARTICLE_H
#define CMOVEABLEPARTICLE_H

#include <memory>
#include "IMoveable.h"
#include "C2DVector.h"

/**
* CMoveableParticle defines the IMoveable functions for a physical particle
*/
class CMoveableParticle : public IMoveable
{
public:
    CMoveableParticle();
    CMoveableParticle(float x_, float y_);
    CMoveableParticle(const C2DVector& initial_pos_);

    CMoveableParticle(const CMoveableParticle& other_);

    CMoveableParticle& operator=(const CMoveableParticle& other_);

    virtual bool IsHit(const C2DVector& coords_) const;

private:
    virtual std::unique_ptr<IMoveable> DoClone() const;

    static const int boundingbox_half_side = 20;
};
#endif
