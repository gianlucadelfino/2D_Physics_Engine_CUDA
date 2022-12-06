#include "scene_base.h"
#include "world_manager.h"

scene_base::scene_base(world_manager& world_, SDL_Color bgr_color_)
    : _world(world_), _background_color(bgr_color_), _mouse_coords(std::make_shared<vec2>())
{
}

scene_base::scene_base(const scene_base& other_)
    : _world(other_._world),
      _background_color(other_._background_color),
      _mouse_coords(other_._mouse_coords)
{
}

scene_base& scene_base::operator=(const scene_base& rhs)
{
  if (this != &rhs)
  {
    _background_color = rhs._background_color;
    _mouse_coords = rhs._mouse_coords;
  }
  return *this;
}

void scene_base::init() {}

void scene_base::update(float dt)
{
  // update the "Entities"
  for (std::unique_ptr<entity_base>& it : _entities)
  {
    it->update(vec2(0.0f, 0.0f), dt);
  }
}

void scene_base::handle_event(world_manager& /*world_*/, const SDL_Event& event_)
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
    for (std::unique_ptr<entity_base>& it : _UI_elements)
    {
      if (it->is_hit(*_mouse_coords))
      {
        it->handle_mouse_buttondown(_mouse_coords);
        UI_element_hit = true;
        break;
      }
    }
    // now propagate the click if to the entities if I didnt hit a UI
    // element
    if (!UI_element_hit)
    {
      for (std::unique_ptr<entity_base>& it : _entities)
      {
        it->handle_mouse_buttondown(_mouse_coords);
      }
    }
    break;
  case SDL_MOUSEBUTTONUP:
    // see if I hit a UI element first, if so, dont look at the entities!
    for (std::unique_ptr<entity_base>& it : _UI_elements)
    {
      if (it->is_hit(*_mouse_coords))
      {
        it->handle_mouse_buttonup(_mouse_coords);
        UI_element_hit = true;
        break;
      }
    }
    // now propagate the click if to the entities if I didnt hit a UI
    // element
    if (!UI_element_hit)
    {
      for (std::unique_ptr<entity_base>& it : _entities)
      {
        it->handle_mouse_buttonup(_mouse_coords);
      }
    }
    break;
  }
}

void scene_base::draw(SDL_Renderer* renderer_) const
{
  SDL_RenderClear(renderer_);
  // draw Background
  SDL_SetRenderDrawColor(
      renderer_, _background_color.r, _background_color.g, _background_color.b, SDL_ALPHA_OPAQUE);

  // draw the "Entities"
  for (const std::unique_ptr<entity_base>& cit : _entities)
  {
    cit->draw(renderer_);
  }
  // draw the HUD/UI last (on top)
  for (const std::unique_ptr<entity_base>& cit : _UI_elements)
  {
    cit->draw(renderer_);
  }
}

scene_base::~scene_base() {}
