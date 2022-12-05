#include "CEntityParticle.h"

CEntityParticle::CEntityParticle(const unsigned int id_,
                                 std::unique_ptr<IMoveable> _,
                                 std::unique_ptr<IDrawable> d_,
                                 std::unique_ptr<CPhysics> p_)
    : CEntity(id_, std::move(_), std::move(d_)), _physics(std::move(p_)), _is_static(false)
{
}

/**
 * For the copy constructor remind that both pimpl objects drawable and physics
 * are NOT shared,
 * so must clone 'em.
 */
CEntityParticle::CEntityParticle(const CEntityParticle& other_)
    : CEntity(other_), _is_static(other_._is_static)
{
  if (other_._physics)
    _physics = std::move(other_._physics->Clone());
  else
    _physics = nullptr;
  if (other_._drawable)
    _drawable = std::move(other_._drawable->Clone());
  else
    _drawable = nullptr;
}

CEntityParticle& CEntityParticle::operator=(const CEntityParticle& rhs)
{
  if (this != &rhs)
  {
    CEntity::operator=(rhs);
    if (rhs._physics)
      _physics = std::move(rhs._physics->Clone());
    else
      _physics = nullptr;
    if (rhs._drawable)
      _drawable = std::move(rhs._drawable->Clone());
    else
      _drawable = nullptr;
    _is_static = rhs._is_static;
  }

  return *this;
}

void CEntityParticle::Draw() const
{
  if (_drawable)
  {
    _drawable->Draw(_moveable->pos, C2DVector(0.0f, 0.0f));
  }
}
void CEntityParticle::Update(const C2DVector& external_force_, float dt)
{
  if (!_is_static && _physics && _physics)
  {
    _physics->Update(external_force_, _moveable, dt);
    // impose local constraints
    _moveable->ImposeConstraints();
  }
}

void CEntityParticle::AddDrawable(std::unique_ptr<IDrawable> drawable_)
{
  _drawable = std::move(drawable_->Clone());
}

void CEntityParticle::AddPhysics(std::unique_ptr<CPhysics> physics_)
{
  _physics = std::move(physics_->Clone());
}

void CEntityParticle::HandleMouseButtonDown(std::shared_ptr<C2DVector> coords_)
{
  if (_moveable->IsHit(*coords_))
  {
    SetConstraint(coords_);
  }
}

void CEntityParticle::HandleMouseButtonUp(std::shared_ptr<C2DVector> /*coords_*/)
{
  _moveable->UnsetConstraint();
}

/*
 * Reposition moves the particle and sets its speed to zero
 */
void CEntityParticle::Reposition(const C2DVector& new_position_)
{
  if (!_is_static)
  {
    _moveable->Reposition(new_position_);
  }
}

/*
 * Move offsets the particle which makes it gain speed
 */
void CEntityParticle::Boost(const C2DVector& new_position_)
{
  if (_moveable && !_is_static)
    _moveable->Boost(new_position_);
}

/*
 * Translate moves particle and its velocity vector by a shift vector
 */
void CEntityParticle::Translate(const C2DVector& shift_)
{
  if (_moveable && !_is_static)
    _moveable->Translate(shift_);
}

void CEntityParticle::SetConstraint(std::shared_ptr<C2DVector> constrainted_pos_)
{
  if (_moveable)
    _moveable->SetConstraint(constrainted_pos_);
}

float CEntityParticle::GetMass() const
{
  float mass = 0.0f;
  if (_physics)
    mass = _physics->GetMass();

  return mass;
}

void CEntityParticle::SetMass(float mass_)
{
  if (_physics)
    _physics->SetMass(mass_);
}

void CEntityParticle::SetSize(const C2DVector& size_)
{
  if (_drawable)
    _drawable->SetSize(size_);
}

void CEntityParticle::SetScale(float scale_)
{
  if (_drawable)
    _drawable->SetScale(scale_);
}

CEntityParticle::~CEntityParticle() {}
