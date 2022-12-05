#include "scene_cloth.h"
#include "entity_cloth.h"
#include "scene_main_menu.h"
#include <memory>

scene_cloth::scene_cloth(SDL_Renderer* renderer_, world_manager& world_)
    : scene_base(renderer_, world_, {255, 255, 255, 0}),
      _font(std::make_unique<font_handler>("assets/pcseniorSmall.ttf", 20))
{
  // build cloth scene
  std::unique_ptr<physics_base> phys_seam = std::make_unique<physics_base>(1.0f);
  std::unique_ptr<moveable_base> sea_moveable = std::make_unique<moveable_particle>(50.0f, 50.0f);
  std::unique_ptr<drawable_link> cloth_drawable = std::make_unique<drawable_link>(renderer_);

  entity_particle seam(1, std::move(sea_moveable), nullptr, std::move(phys_seam));
  std::unique_ptr<entity_base> cloth =
      std::make_unique<entity_cloth>(1,
                                     vec2(400.0f, 10.0f),
                                     std::move(cloth_drawable),
                                     seam,
                                     40); // id, pos, drawable_base, seam prototype, size
  // add simulation Elements:
  _entities.push_back(std::move(cloth));
}

scene_cloth::scene_cloth(const scene_cloth& other_) : scene_base(other_), _font(other_._font) {}

scene_cloth& scene_cloth::operator=(const scene_cloth& rhs)
{
  if (this != &rhs)
  {
    scene_base::operator=(rhs);
    _font = rhs._font;
  }
  return *this;
}

void scene_cloth::init()
{
  // add UI elements to Scene
  SDL_Color black_color = {0, 0, 0, 0};
  SDL_Color button_label_color = {255, 255, 255, 0};

  std::unique_ptr<moveable_button> main_menu_moveable_button =
      std::make_unique<moveable_button>(vec2(50.0f, 600.0f), vec2(200.0f, 22.0f));
  std::unique_ptr<drawable_button> main_menu_button_drawable = std::make_unique<drawable_button>(
      _font, _renderer, "MAIN MENU", vec2(200.0f, 22.0f), black_color, button_label_color);

  std::unique_ptr<entity_base> galaxy_si_switch_button =
      std::make_unique<entity_button>(1,
                                      std::move(main_menu_moveable_button),
                                      std::move(main_menu_button_drawable),
                                      *_world,
                                      std::make_unique<scene_main_menu>(_renderer, *_world));
  _UI_elements.push_back(std::move(galaxy_si_switch_button));

  // instructions label (first half)
  SDL_Color white_color = {255, 255, 255, SDL_ALPHA_OPAQUE};
  SDL_Color label_color = {0, 0, 0, 0};

  std::unique_ptr<moveable_button> instructionsfirst_label_moveable =
      std::make_unique<moveable_button>(vec2(900.0f, 300.0f), vec2(220.0f, 22.0f));
  std::unique_ptr<drawable_button> instructionsfirst_label_drawable =
      std::make_unique<drawable_button>(
          _font, _renderer, "click and hold to", vec2(220.0f, 22.0f), white_color, label_color);

  std::unique_ptr<entity_base> instructionsfirst_label = std::make_unique<entity_base>(
      1, std::move(instructionsfirst_label_moveable), std::move(instructionsfirst_label_drawable));
  _UI_elements.push_back(std::move(instructionsfirst_label));
  // instructions label (second half)
  std::unique_ptr<moveable_button> instructionssecond_label_moveable =
      std::make_unique<moveable_button>(vec2(900.0f, 330.0f), vec2(220.0f, 22.0f));
  std::unique_ptr<drawable_button> instructionssecond_label_drawable =
      std::make_unique<drawable_button>(
          _font, _renderer, "drag the cloth", vec2(220.0f, 22.0f), white_color, label_color);

  std::unique_ptr<entity_base> instructionssecond_label =
      std::make_unique<entity_base>(1,
                                    std::move(instructionssecond_label_moveable),
                                    std::move(instructionssecond_label_drawable));
  _UI_elements.push_back(std::move(instructionssecond_label));
}
