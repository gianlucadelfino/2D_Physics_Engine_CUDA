#ifndef CENTITYPARTICLE_H
#define CENTITYPARTICLE_H

#include "drawable_base.h"
#include "entity_base.h"
#include "moveable_particle.h"
#include "physics_base.h"
#include <memory>

/**
 * entity_particle defines the entity_base that is actually a physical particle,
 * which reacts to forces acting on it.
 */
class entity_particle : public entity_base
{
public:
  entity_particle(const unsigned int id_,
                  std::unique_ptr<moveable_base> _,
                  std::unique_ptr<drawable_base> d_,
                  std::unique_ptr<physics_base> p_);

  entity_particle(const entity_particle& e);
  entity_particle& operator=(const entity_particle& rhs);

  virtual ~entity_particle();

  virtual void draw(SDL_Renderer*) const;
  virtual void update(const vec2& external_force_, float dt);

  virtual void handle_mouse_buttondown(std::shared_ptr<vec2> coords_);
  virtual void handle_mouse_buttonup(std::shared_ptr<vec2> coords_);

  void add_drawable(std::unique_ptr<drawable_base> drawable_);
  void add_physics(std::unique_ptr<physics_base> physics_);

  /* Accessors to pimpl properties  */
  // moveable_base
  void reposition(const vec2& new_position_);
  void boost(const vec2& new_position_);
  void translate(const vec2& shift_);

  void block() { _is_static = true; }
  void unblock() { _is_static = false; }
  vec2 get_position() const { return _moveable->pos; }
  void set_constraint(std::shared_ptr<vec2> constrainted_pos_);

  // IPhysics
  float get_mass() const;
  void set_mass(float mass_);

  // drawable_base
  void set_size(const vec2& size_);
  void set_scale(float scale_);

private:
  std::unique_ptr<physics_base> _physics;
  bool _is_static;
};

#endif
