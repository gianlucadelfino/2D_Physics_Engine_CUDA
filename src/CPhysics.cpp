#include <cassert>

#include "CPhysics.h"

CPhysics::CPhysics(float mass_)
    : _mass(mass_), _inverseMass(1 / mass_), _gravity_acc(C2DVector(0.0f, 9.8f))
{
}

CPhysics::CPhysics(float mass_, const C2DVector& gravity_accel_)
    : _mass(mass_), _inverseMass(1 / mass_), _gravity_acc(gravity_accel_)
{
}

CPhysics::CPhysics(const CPhysics& other_)
    : _mass(other_._mass), _inverseMass(other_._inverseMass), _gravity_acc(other_._gravity_acc)
{
}

/*
 * Clone is needed to create a deep copy when IMoveable is used as pimpl
 */
std::unique_ptr<CPhysics> CPhysics::Clone() const
{
  std::unique_ptr<CPhysics> clone(DoClone());
  // lets check that the derived class actually implemented clone and it does
  // not come from a parent
  assert(typeid(*clone) == typeid(*this) && "DoClone incorrectly overridden");
  return clone;
}

std::unique_ptr<CPhysics> CPhysics::DoClone() const
{
  return std::make_unique<CPhysics>(_mass, _gravity_acc);
}

CPhysics& CPhysics::operator=(const CPhysics& other_)
{
  if (&other_ != this)
  {
    _mass = other_._mass;
    _inverseMass = other_._inverseMass;
    _gravity_acc = other_._gravity_acc;
  }
  return *this;
}

CPhysics::~CPhysics() {}

// mind the reference to the unique ptr because we want to modify it without sharing ownership!
//(remind that unique_ptr cant be passed by value without moving ownershipt and
// therfore destroying the original)
void CPhysics::Update(const C2DVector& external_force_,
                      std::unique_ptr<IMoveable>& moveable_,
                      float dt)
{
  Integrate(moveable_->pos, moveable_->prev_pos, external_force_, dt);
}

/**
 * Integrate increments the position of the entity by integrating the force
 * acting on it
 */
void CPhysics::Integrate(C2DVector& pos,
                         C2DVector& prev_pos,
                         const C2DVector& external_force_,
                         float dt)
{
  // verlet
  C2DVector temp = pos;
  pos += (pos - prev_pos) + _inverseMass * (GetForce(pos) + external_force_) * dt * dt;
  prev_pos = temp;
}
