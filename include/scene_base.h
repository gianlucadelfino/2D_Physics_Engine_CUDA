#ifndef ISCENE_H
#define ISCENE_H

#include "SDL.h"
#include "entity_base.h"
#include "vec2.h"
#include <memory>
#include <string>
#include <vector>

class world_manager;

/**
 * scene_base defines the interface for all the scenes that the world_manager contain.
 */
class scene_base
{
public:
  scene_base(world_manager& world_, SDL_Color bgr_color_);
  scene_base(const scene_base& other_);
  scene_base& operator=(const scene_base& rhs);

  /**
   * init loads the scene elements. Should be called before calling update on
   * it.
   */
  virtual void init();

  /**
   * update takes care of updating all the IEnities in the scene
   */
  virtual void update(float dt);
  /**
   * handle_event dispatches the events (mouse/keyboars) to the UI and the other
   * IEntities of the scene
   */
  virtual void handle_event(world_manager& world_, const SDL_Event& event_);
  /**
   * draw renders all the IEntities of the scene
   */
  void draw(SDL_Renderer*) const;

  virtual ~scene_base();

protected:
  world_manager& _world;
  SDL_Color _background_color;
  std::shared_ptr<vec2> _mouse_coords;
  std::vector<std::unique_ptr<entity_base>> _entities;
  std::vector<std::unique_ptr<entity_base>> _UI_elements;
};

#endif
