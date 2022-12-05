#include <cassert>

#include "physics_base.h"

physics_base::physics_base(float mass_)
    : _mass(mass_), _inverseMass(1 / mass_), _gravity_acc(vec2(0.0f, 9.8f))
{
}

physics_base::physics_base(float mass_, const vec2& gravity_accel_)
    : _mass(mass_), _inverseMass(1 / mass_), _gravity_acc(gravity_accel_)
{
}

physics_base::physics_base(const physics_base& other_)
    : _mass(other_._mass), _inverseMass(other_._inverseMass), _gravity_acc(other_._gravity_acc)
{
}

/*
 * clone is needed to create a deep copy when moveable_base is used as pimpl
 */
std::unique_ptr<physics_base> physics_base::clone() const
{
  std::unique_ptr<physics_base> clone(do_clone());
  // lets check that the derived class actually implemented clone and it does
  // not come from a parent
  assert(typeid(*clone) == typeid(*this) && "do_clone incorrectly overridden");
  return clone;
}

std::unique_ptr<physics_base> physics_base::do_clone() const
{
  return std::make_unique<physics_base>(_mass, _gravity_acc);
}

physics_base& physics_base::operator=(const physics_base& other_)
{
  if (&other_ != this)
  {
    _mass = other_._mass;
    _inverseMass = other_._inverseMass;
    _gravity_acc = other_._gravity_acc;
  }
  return *this;
}

physics_base::~physics_base() {}

// mind the reference to the unique ptr because we want to modify it without sharing ownership!
//(remind that unique_ptr cant be passed by value without moving ownershipt and
// therfore destroying the original)
void physics_base::update(const vec2& external_force_,
                          std::unique_ptr<moveable_base>& moveable_,
                          float dt)
{
  integrate(moveable_->pos, moveable_->prev_pos, external_force_, dt);
}

/**
 * integrate increments the position of the entity by integrating the force
 * acting on it
 */
void physics_base::integrate(vec2& pos, vec2& prev_pos, const vec2& external_force_, float dt)
{
  // verlet
  vec2 temp = pos;
  pos += (pos - prev_pos) + _inverseMass * (get_force(pos) + external_force_) * dt * dt;
  prev_pos = temp;
}
