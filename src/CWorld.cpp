#include <iostream>
#include <thread>

#include "CSceneGalaxy.h"
#include "CSceneMainMenu.h"
#include "CWorld.h"
#include "SDL_ttf.h"

CWorld::CWorld(SDL_Renderer* renderer_) : _renderer(renderer_)
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
  std::unique_ptr<IScene> main_menu = std::make_unique<CSceneMainMenu>(_renderer, *this);

  _cur_scene = std::move(main_menu);
  _cur_scene->Init();
}

void CWorld::Update(float dt) const
{
  if (_cur_scene)
    _cur_scene->Update(dt);
}

void CWorld::HandleEvent(const SDL_Event& event_)
{
  if (_cur_scene)
    _cur_scene->HandleEvent(*this, event_);
}

void CWorld::Draw() const
{
  if (_cur_scene)
    _cur_scene->Draw();
}

void CWorld::ChangeScene(std::unique_ptr<IScene> new_scene_)
{
  new_scene_->Init();
  _cur_scene = std::move(new_scene_);
}

CWorld::~CWorld()
{
  // manually reset the scene pointer(calling its destruction), before we
  // deallocate SDL resources
  _cur_scene.reset();

  // NOW we can free resources
  TTF_Quit();
}
