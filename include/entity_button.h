#ifndef CENTITYBUTTON_H
#define CENTITYBUTTON_H

#include "drawable_base.h"
#include "entity_base.h"
#include "moveable_base.h"
#include "scene_base.h"
#include <memory>

class world_manager;

/**
 * entity_button defines the entity_base to be used as a UI clickable Button.
 */
class entity_button : public entity_base
{
public:
  entity_button(unsigned int id_,
                std::unique_ptr<moveable_base> moveable_,
                std::unique_ptr<drawable_base> drawable_,
                world_manager& world_,
                std::unique_ptr<scene_base> scene_to_switch_to_);

  virtual void handle_mouse_buttondown(std::shared_ptr<vec2> cursor_position_);
  virtual void handle_mouse_buttonup(std::shared_ptr<vec2> cursor_position_);

  virtual bool is_hit(const vec2& coords_) const;

private:
  world_manager& _world;
  std::unique_ptr<scene_base> _scene_to_switch_to;
};

#endif
