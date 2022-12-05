#include "CEntityButton.h"
#include "CWorld.h"

CEntityButton::CEntityButton(unsigned int id_,
                             std::unique_ptr<IMoveable> moveable_,
                             std::unique_ptr<IDrawable> drawable_,
                             CWorld& world_,
                             std::unique_ptr<IScene> scene_to_switch_to_)
    : CEntity(id_, std::move(moveable_), std::move(drawable_)),
      _world(world_),
      _scene_to_switch_to(std::move(scene_to_switch_to_))
{
}

void CEntityButton::HandleMouseButtonDown(
    std::shared_ptr<C2DVector> /*cursor_position_*/)
{
}

void CEntityButton::HandleMouseButtonUp(
    std::shared_ptr<C2DVector> /*cursor_position_*/)
{
    _world.ChangeScene(std::move(_scene_to_switch_to));
}

bool CEntityButton::IsHit(const C2DVector& coords_) const
{
    bool is_hit = false;
    if (_moveable)
    {
        is_hit = _moveable->IsHit(coords_);
    }
    return is_hit;
}
