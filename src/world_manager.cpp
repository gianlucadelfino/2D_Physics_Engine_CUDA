#include <iostream>
#include <thread>

#include "SDL_ttf.h"
#include "scene_galaxy.h"
#include "scene_main_menu.h"
#include "world_manager.h"

world_manager::world_manager(SDL_Renderer* renderer_) : _renderer(renderer_)
{
  if (!_renderer)
  {
    std::cerr << "Could not initalize World without a screen" << std::endl;
    exit(EXIT_FAILURE);
  }

  // init SDL_ttf
  if (TTF_Init() == -1)
  {
    std::cerr << "Could NOT initialize SDL_ttf.." << std::endl;
  }
  // add the Main Menu as first scene
  std::unique_ptr<scene_base> main_menu = std::make_unique<scene_main_menu>(_renderer, *this);

  _cur_scene = std::move(main_menu);
  _cur_scene->init();
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

void world_manager::draw() const
{
  if (_cur_scene)
    _cur_scene->draw();
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
