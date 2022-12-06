#ifndef CSCENECLOTH_H
#define CSCENECLOTH_H

#include "SDL.h"
#include <memory>

#include "drawable_button.h"
#include "entity_button.h"
#include "moveable_button.h"
#include "physics_base.h"
#include "scene_base.h"
#include "vec2.h"

class world_manager;

/**
 * scene_cloth defines the scene with the moving cloth.
 */
class scene_cloth : public scene_base
{
public:
  scene_cloth(world_manager& world_);
  scene_cloth(const scene_cloth& other_);
  scene_cloth& operator=(const scene_cloth& rhs);

  virtual void init();

private:
  std::shared_ptr<font_handler> _font;
};

#endif
