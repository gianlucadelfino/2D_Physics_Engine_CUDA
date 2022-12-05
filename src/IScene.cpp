#include "IScene.h"
#include "CWorld.h"

IScene::IScene(SDL_Renderer* renderer_, CWorld& world_, SDL_Color bgr_color_)
    : _renderer(renderer_),
      _world(&world_),
      _background_color(bgr_color_),
      _mouse_coords(std::make_shared<C2DVector>())
{
}

IScene::IScene(const IScene& other_)
    : _renderer(other_._renderer),
      _world(other_._world),
      _background_color(other_._background_color),
      _mouse_coords(other_._mouse_coords)
{
}

IScene& IScene::operator=(const IScene& rhs)
{
  if (this != &rhs)
  {
    _renderer = rhs._renderer;
    _world = rhs._world;
    _background_color = rhs._background_color;
    _mouse_coords = rhs._mouse_coords;
  }
  return *this;
}

void IScene::Init() {}

void IScene::Update(float dt)
{
  // update the "Entities"
  for (std::unique_ptr<CEntity>& it : _entities)
  {
    it->Update(C2DVector(0.0f, 0.0f), dt);
  }
}

void IScene::HandleEvent(CWorld& /*world_*/, const SDL_Event& event_)
{
  bool UI_element_hit = false;

  switch (event_.type)
  {
  case SDL_KEYDOWN:
    break;
  case SDL_MOUSEMOTION:
    _mouse_coords->x = static_cast<float>(event_.motion.x);
    _mouse_coords->y = static_cast<float>(event_.motion.y);
    break;
  case SDL_MOUSEBUTTONDOWN:
    // see if I hit a UI element first, if so, dont look at the entities!
    for (std::unique_ptr<CEntity>& it : _UI_elements)
    {
      if (it->IsHit(*_mouse_coords))
      {
        it->HandleMouseButtonDown(_mouse_coords);
        UI_element_hit = true;
        break;
      }
    }
    // now propagate the click if to the entities if I didnt hit a UI
    // element
    if (!UI_element_hit)
    {
      for (std::unique_ptr<CEntity>& it : _entities)
      {
        it->HandleMouseButtonDown(_mouse_coords);
      }
    }
    break;
  case SDL_MOUSEBUTTONUP:
    // see if I hit a UI element first, if so, dont look at the entities!
    for (std::unique_ptr<CEntity>& it : _UI_elements)
    {
      if (it->IsHit(*_mouse_coords))
      {
        it->HandleMouseButtonUp(_mouse_coords);
        UI_element_hit = true;
        break;
      }
    }
    // now propagate the click if to the entities if I didnt hit a UI
    // element
    if (!UI_element_hit)
    {
      for (std::unique_ptr<CEntity>& it : _entities)
      {
        it->HandleMouseButtonUp(_mouse_coords);
      }
    }
    break;
  }
}

void IScene::Draw() const
{
  SDL_RenderClear(_renderer);
  // Draw Background
  SDL_SetRenderDrawColor(
      _renderer, _background_color.r, _background_color.g, _background_color.b, SDL_ALPHA_OPAQUE);

  // Draw the "Entities"
  for (const std::unique_ptr<CEntity>& cit : _entities)
  {
    cit->Draw();
  }
  // Draw the HUD/UI last (on top)
  for (const std::unique_ptr<CEntity>& cit : _UI_elements)
  {
    cit->Draw();
  }
}

IScene::~IScene() {}
