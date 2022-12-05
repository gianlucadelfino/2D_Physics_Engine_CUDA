#ifndef IMOVEABLE_H
#define IMOVEABLE_H

#include <memory>

#include "vec2.h"

/**
 * moveable_base. Anything on screen that can move should use an moveable_base, i.e. have
 * a composite object that inherits from moveable_base.
 */
class moveable_base
{
public:
  moveable_base(float x_, float y_);
  moveable_base(const vec2& pos_);
  moveable_base(const vec2& pos_, const vec2& old_pos_, const vec2& orientation_);

  // copy constructor and assignment operator
  moveable_base(const moveable_base& other_);
  moveable_base& operator=(const moveable_base& other_);

  /**
   * boost offsets the particle which makes it gain speed
   */
  void boost(const vec2& new_position_);

  /**
   * reposition moves the particle and sets its speed to zero
   */
  void reposition(const vec2& new_position_);

  /**
   * translate moves particle and its velocity vector by a shift vector
   */
  void translate(const vec2& shift_);

  /**
   * clone is needed to create a deep copy when moveable_base is used as pimpl
   */
  std::unique_ptr<moveable_base> clone() const; // non virtual to check that it
                                                // has been overridden (see "c++
                                                // coding standards" elem 54)

  virtual ~moveable_base();

  /**
   * ImposeConstraints solves the constraints imposed on the position/velocity
   * of the objects and applies them
   */
  virtual void ImposeConstraints();

  /**
   * set_constraint adds the pointer to the position the current instance is
   * supposed to stick to
   */
  virtual void set_constraint(std::shared_ptr<vec2> constrainted_pos_);

  /**
   * UnsetConstraint frees the moveable_base from a set constraint
   */
  virtual void UnsetConstraint();

  /**
   * is_hit receives a set of coordinates and returns true if hit (default to
   * false)
   * @param coords_ the position to test if is hitting the moveable_base
   */
  virtual bool is_hit(const vec2& /*coords_*/) const { return false; }

  vec2 pos;
  vec2 prev_pos;
  vec2 orientation;

protected:
  // constraints. The origin needs to be a pointer to be dynamic
  std::shared_ptr<vec2> _constraint;

private:
  virtual std::unique_ptr<moveable_base> do_clone() const = 0;
};
#endif
