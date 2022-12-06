#ifndef CWORLD_H
#define CWORLD_H

#include "SDL.h"
#include "SDL_events.h"
#include "scene_base.h"
#include <memory>

#include "scene_galaxy.h"

class scene_base;

/**
 * world_manager is a Finite-State-Machine that switches between Scenes ( aka "levels"
 * or "simulations" )
 */
class world_manager
{
public:
  world_manager();

  void update(float dt) const;
  void handle_event(const SDL_Event& event_);
  void draw(SDL_Renderer*) const;

  void change_scene(std::unique_ptr<scene_base> new_scene_);
  ~world_manager();

private:
  /*forbid copy and assignment*/
  world_manager(const world_manager&);
  world_manager& operator=(const world_manager&);

  std::unique_ptr<scene_base> _cur_scene;
};

#endif
