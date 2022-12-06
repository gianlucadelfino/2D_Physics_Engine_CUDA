#ifndef CSCENEGALAXY_H
#define CSCENEGALAXY_H

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
 * scene_galaxy defines the scene with the galaxy.
 */
class scene_galaxy : public scene_base
{
public:
  scene_galaxy(world_manager& world_, bool use_CUDA_, unsigned int stars_num_);
  scene_galaxy(const scene_galaxy& other_);
  scene_galaxy& operator=(const scene_galaxy& rhs);

  virtual void init();

  ~scene_galaxy();

private:
  bool _using_CUDA;
  unsigned int _stars_num;
  std::shared_ptr<font_handler> _font;
  bool _CUDA_capable_device_present;
};

#endif
