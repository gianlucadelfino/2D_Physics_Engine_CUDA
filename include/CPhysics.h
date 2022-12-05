#ifndef CPHYSICS_H
#define CPHYSICS_H

#include <memory>

#include "C2DVector.h"
#include "IMoveable.h"

/**
 * CPhysics defines the interface and basic implementation for entities that are
 * bound to follow physics laws.
 */
class CPhysics
{
public:
  explicit CPhysics(float mass_);

  CPhysics(float mass_, const C2DVector& gravity_accel_);

  CPhysics(const CPhysics& other_);

  CPhysics& operator=(const CPhysics& other_);

  virtual ~CPhysics();

  /**
   * Clone is needed to create a deep copy when IMoveable is used as pimpl
   */
  std::unique_ptr<CPhysics> Clone() const; // non virtual to check that it has
                                           // been overridden (see "c++ coding
                                           // standards" elem 54)

  // mind the reference to the unique ptr because we want to modify it without
  // sharing ownership!
  // (remind that unique_ptr cant be passed by value without moving ownership
  // and therfore destroying the original)
  virtual void Update(const C2DVector& external_force_,
                      std::unique_ptr<IMoveable>& moveable_,
                      float dt);

  virtual C2DVector GetForce(const C2DVector& /*pos*/) const { return _mass * _gravity_acc; }

  float GetMass() const { return _mass; }
  void SetMass(float mass_) { _mass = mass_; }

  void SetGravity(const C2DVector& _grav) { _gravity_acc = _grav; }

protected:
  virtual std::unique_ptr<CPhysics> DoClone() const;
  void Integrate(C2DVector& pos, C2DVector& prev_pos, const C2DVector& external_force_, float dt);

  float _mass;
  float _inverseMass;
  C2DVector _gravity_acc;
};

#endif
