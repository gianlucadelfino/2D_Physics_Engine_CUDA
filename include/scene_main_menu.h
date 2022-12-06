#ifndef CSCENEMAINMENU_H
#define CSCENEMAINMENU_H

#include "SDL.h"
#include <memory>

#include "drawable_button.h"
#include "drawable_star.h"
#include "entity_button.h"
#include "entity_galaxy.h"
#include "moveable_button.h"
#include "physics_base.h"
#include "scene_base.h"
#include "vec2.h"

class world_manager;

/**
 * scene_main_menu defines the scene with the main menu.
 */
class scene_main_menu : public scene_base
{
public:
  explicit scene_main_menu(world_manager& world_);
  scene_main_menu(const scene_main_menu& other_);
  scene_main_menu& operator=(const scene_main_menu& rhs);

  virtual void init();
};

#endif
