#ifndef CENTITY_H
#define CENTITY_H

#include "drawable_base.h"
#include "moveable_base.h"
#include "vec2.h"
#include <memory>

/**
 * entity_base defines the interface of the entities populating the game/simulation.
 */
class entity_base
{
public:
  /**
   * entity_base constructor
   * @param id_ the id of the entity
   * @param moveable_ unique_ptr of the moveable_base object, of which takes
   *   owneship (using move semantics)
   * @param drawable_ shared_ptr of which takes owneship (using move semantics)
   */
  entity_base(unsigned int id_,
              std::unique_ptr<moveable_base> moveable_,
              std::unique_ptr<drawable_base> drawable_);
  entity_base(const entity_base& other_);

  entity_base& operator=(const entity_base& rhs);

  virtual ~entity_base();

  void SetId(unsigned int id_) { _id = id_; }
  unsigned int GetId() const { return _id; }

  virtual bool is_hit(const vec2& coords_) const;

  virtual void handle_mouse_buttondown([[maybe_unused]] std::shared_ptr<vec2> cursor_position_) {}
  virtual void handle_mouse_buttonup([[maybe_unused]] std::shared_ptr<vec2> cursor_position_) {}

  /**
   * update should be overridden and called if the Entity is subject to
   * external forces and has an moveable_base
   */
  virtual void update(const vec2& /*external_force_*/, float /*dt*/) {}

  /**
   * draw renders the Entity if an drawable_base is available
   */
  virtual void draw() const;

protected:
  unsigned int _id;
  std::unique_ptr<moveable_base> _moveable;
  std::unique_ptr<drawable_base> _drawable;
};

#endif
