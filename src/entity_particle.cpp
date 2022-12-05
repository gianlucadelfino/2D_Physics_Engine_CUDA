#include "entity_particle.h"

entity_particle::entity_particle(const unsigned int id_,
                                 std::unique_ptr<moveable_base> _,
                                 std::unique_ptr<drawable_base> d_,
                                 std::unique_ptr<physics_base> p_)
    : entity_base(id_, std::move(_), std::move(d_)), _physics(std::move(p_)), _is_static(false)
{
}

/**
 * For the copy constructor remind that both pimpl objects drawable and physics
 * are NOT shared,
 * so must clone 'em.
 */
entity_particle::entity_particle(const entity_particle& other_)
    : entity_base(other_), _is_static(other_._is_static)
{
  if (other_._physics)
    _physics = std::move(other_._physics->clone());
  else
    _physics = nullptr;
  if (other_._drawable)
    _drawable = std::move(other_._drawable->clone());
  else
    _drawable = nullptr;
}

entity_particle& entity_particle::operator=(const entity_particle& rhs)
{
  if (this != &rhs)
  {
    entity_base::operator=(rhs);
    if (rhs._physics)
      _physics = std::move(rhs._physics->clone());
    else
      _physics = nullptr;
    if (rhs._drawable)
      _drawable = std::move(rhs._drawable->clone());
    else
      _drawable = nullptr;
    _is_static = rhs._is_static;
  }

  return *this;
}

void entity_particle::draw() const
{
  if (_drawable)
  {
    _drawable->draw(_moveable->pos, vec2(0.0f, 0.0f));
  }
}
void entity_particle::update(const vec2& external_force_, float dt)
{
  if (!_is_static && _physics && _physics)
  {
    _physics->update(external_force_, _moveable, dt);
    // impose local constraints
    _moveable->ImposeConstraints();
  }
}

void entity_particle::add_drawable(std::unique_ptr<drawable_base> drawable_)
{
  _drawable = std::move(drawable_->clone());
}

void entity_particle::add_physics(std::unique_ptr<physics_base> physics_)
{
  _physics = std::move(physics_->clone());
}

void entity_particle::handle_mouse_buttondown(std::shared_ptr<vec2> coords_)
{
  if (_moveable->is_hit(*coords_))
  {
    set_constraint(coords_);
  }
}

void entity_particle::handle_mouse_buttonup(std::shared_ptr<vec2> /*coords_*/)
{
  _moveable->UnsetConstraint();
}

/*
 * reposition moves the particle and sets its speed to zero
 */
void entity_particle::reposition(const vec2& new_position_)
{
  if (!_is_static)
  {
    _moveable->reposition(new_position_);
  }
}

/*
 * Move offsets the particle which makes it gain speed
 */
void entity_particle::boost(const vec2& new_position_)
{
  if (_moveable && !_is_static)
    _moveable->boost(new_position_);
}

/*
 * translate moves particle and its velocity vector by a shift vector
 */
void entity_particle::translate(const vec2& shift_)
{
  if (_moveable && !_is_static)
    _moveable->translate(shift_);
}

void entity_particle::set_constraint(std::shared_ptr<vec2> constrainted_pos_)
{
  if (_moveable)
    _moveable->set_constraint(constrainted_pos_);
}

float entity_particle::get_mass() const
{
  float mass = 0.0f;
  if (_physics)
    mass = _physics->get_mass();

  return mass;
}

void entity_particle::set_mass(float mass_)
{
  if (_physics)
    _physics->set_mass(mass_);
}

void entity_particle::set_size(const vec2& size_)
{
  if (_drawable)
    _drawable->set_size(size_);
}

void entity_particle::set_scale(float scale_)
{
  if (_drawable)
    _drawable->set_scale(scale_);
}

entity_particle::~entity_particle() {}
