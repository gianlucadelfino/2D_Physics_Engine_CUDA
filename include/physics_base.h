#ifndef CPHYSICS_H
#define CPHYSICS_H

#include <memory>

#include "moveable_base.h"
#include "vec2.h"

/**
 * physics_base defines the interface and basic implementation for entities that are
 * bound to follow physics laws.
 */
class physics_base
{
public:
  explicit physics_base(float mass_);

  physics_base(float mass_, const vec2& gravity_accel_);

  physics_base(const physics_base& other_);

  physics_base& operator=(const physics_base& other_);

  virtual ~physics_base();

  /**
   * clone is needed to create a deep copy when moveable_base is used as pimpl
   */
  std::unique_ptr<physics_base> clone() const; // non virtual to check that it has
                                               // been overridden (see "c++ coding
                                               // standards" elem 54)

  // mind the reference to the unique ptr because we want to modify it without
  // sharing ownership!
  // (remind that unique_ptr cant be passed by value without moving ownership
  // and therfore destroying the original)
  virtual void update(const vec2& external_force_,
                      std::unique_ptr<moveable_base>& moveable_,
                      float dt);

  virtual vec2 get_force(const vec2& /*pos*/) const { return _mass * _gravity_acc; }

  float get_mass() const { return _mass; }
  void set_mass(float mass_) { _mass = mass_; }

  void set_gravity(const vec2& _grav) { _gravity_acc = _grav; }

protected:
  virtual std::unique_ptr<physics_base> do_clone() const;
  void integrate(vec2& pos, vec2& prev_pos, const vec2& external_force_, float dt);

  float _mass;
  float _inverseMass;
  vec2 _gravity_acc;
};

#endif
