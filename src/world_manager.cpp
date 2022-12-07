#include <SDL_render.h>
#include <iostream>
#include <thread>

#include "SDL_ttf.h"
#include "scene_galaxy.h"
#include "scene_main_menu.h"
#include "world_manager.h"

world_manager::world_manager()
{
  // init SDL_ttf
  if (TTF_Init() == -1)
  {
    std::cerr << "Could NOT initialize SDL_ttf.." << std::endl;
  }
  // add the Main Menu as first scene
  change_scene(std::make_unique<scene_main_menu>(*this));
}

void world_manager::update(float dt) const
{
  if (_cur_scene)
    _cur_scene->update(dt);
}

void world_manager::handle_event(const SDL_Event& event_)
{
  if (_cur_scene)
    _cur_scene->handle_event(*this, event_);
}

void world_manager::draw(SDL_Renderer* renderer_) const
{
  if (_cur_scene)
    _cur_scene->draw(renderer_);
}

void world_manager::change_scene(std::unique_ptr<scene_base> new_scene_)
{
  new_scene_->init();
  _cur_scene = std::move(new_scene_);
}

world_manager::~world_manager()
{
  // manually reset the scene pointer(calling its destruction), before we
  // deallocate SDL resources
  _cur_scene.reset();

  // NOW we can free resources
  TTF_Quit();
}
