#include "entity_button.h"
#include "world_manager.h"

entity_button::entity_button(unsigned int id_,
                             std::unique_ptr<moveable_base> moveable_,
                             std::unique_ptr<drawable_base> drawable_,
                             world_manager& world_,
                             std::unique_ptr<scene_base> scene_to_switch_to_)
    : entity_base(id_, std::move(moveable_), std::move(drawable_)),
      _world(world_),
      _scene_to_switch_to(std::move(scene_to_switch_to_))
{
}

void entity_button::handle_mouse_buttondown(std::shared_ptr<vec2> /*cursor_position_*/) {}

void entity_button::handle_mouse_buttonup(std::shared_ptr<vec2> /*cursor_position_*/)
{
  _world.change_scene(std::move(_scene_to_switch_to));
}

bool entity_button::is_hit(const vec2& coords_) const
{
  bool is_hit = false;
  if (_moveable)
  {
    is_hit = _moveable->is_hit(coords_);
  }
  return is_hit;
}
