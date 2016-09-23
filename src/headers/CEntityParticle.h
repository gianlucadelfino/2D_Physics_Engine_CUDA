#ifndef CENTITYPARTICLE_H
#define CENTITYPARTICLE_H

#include <memory>
#include "CEntity.h"
#include "IDrawable.h"
#include "CPhysics.h"
#include "CMoveableParticle.h"

/**
* CEntityParticle defines the CEntity that is actually a physical particle,
* which reacts to forces acting on it.
*/
class CEntityParticle : public CEntity
{
public:
    CEntityParticle(const unsigned int id_,
                    std::unique_ptr<IMoveable> m_,
                    std::unique_ptr<IDrawable> d_,
                    std::unique_ptr<CPhysics> p_);

    CEntityParticle(const CEntityParticle& e);
    CEntityParticle& operator=(const CEntityParticle& rhs);

    virtual ~CEntityParticle();

    virtual void Draw() const;
    virtual void Update(const C2DVector& external_force_, float dt);

    virtual void HandleMouseButtonDown(std::shared_ptr<C2DVector> coords_);
    virtual void HandleMouseButtonUp(std::shared_ptr<C2DVector> coords_);

    void AddDrawable(std::unique_ptr<IDrawable> drawable_);
    void AddPhysics(std::unique_ptr<CPhysics> physics_);

    /* Accessors to pimpl properties  */
    // IMoveable
    void Reposition(const C2DVector& new_position_);
    void Boost(const C2DVector& new_position_);
    void Translate(const C2DVector& shift_);

    void Block()
    {
        m_is_static = true;
    }
    void Unblock()
    {
        m_is_static = false;
    }
    C2DVector GetPosition() const
    {
        return mp_moveable->pos;
    }
    void SetConstraint(std::shared_ptr<C2DVector> constrainted_pos_);

    // IPhysics
    float GetMass() const;
    void SetMass(float mass_);

    // IDrawable
    void SetSize(const C2DVector& size_);
    void SetScale(float scale_);

private:
    std::unique_ptr<CPhysics> mp_physics;
    bool m_is_static;
};

#endif
