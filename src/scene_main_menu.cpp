#include <string>

#include "cuda_utils.cuh"
#include "moveable_button.h"
#include "scene_cloth.h"
#include "scene_galaxy.h"
#include "scene_main_menu.h"

scene_main_menu::scene_main_menu(world_manager& world_) : scene_base(world_, {0, 0, 0, 0}) {}

scene_main_menu::scene_main_menu(const scene_main_menu& other_) : scene_base(other_) {}

scene_main_menu& scene_main_menu::operator=(const scene_main_menu& rhs)
{
  if (this != &rhs)
  {
    scene_base::operator=(rhs);
  }
  return *this;
}

void scene_main_menu::init()
{
  // add UI elements to Scene
  SDL_Color white_color = {255, 255, 255, SDL_ALPHA_OPAQUE};
  SDL_Color black_color = {0, 0, 0, 0};
  SDL_Color button_label_color = {0, 0, 0, 0};

  // TITLE label
  std::shared_ptr<font_handler> title_font =
      std::make_shared<font_handler>("assets/pcseniorSmall.ttf", 30);
  SDL_Color title_color = {255, 255, 255, 0};
  std::unique_ptr<moveable_button> title_moveable =
      std::make_unique<moveable_button>(vec2(400.0f, 100.0f), vec2(300.0f, 31.0f));
  std::unique_ptr<drawable_button> title_drawable = std::make_unique<drawable_button>(
      title_font, "SIMULATOR 2000", vec2(450.0f, 90.0f), black_color, title_color);

  std::unique_ptr<entity_base> title =
      std::make_unique<entity_base>(1, std::move(title_moveable), std::move(title_drawable));
  _UI_elements.push_back(std::move(title));

  // switch to CLOTH simulation button
  std::shared_ptr<font_handler> button_font =
      std::make_shared<font_handler>("assets/pcseniorSmall.ttf", 20);

  std::unique_ptr<moveable_button> cloth_moveable_button =
      std::make_unique<moveable_button>(vec2(500.0f, 300.0f), vec2(200.0f, 22.0f));
  std::unique_ptr<drawable_button> cloth_button_drawable = std::make_unique<drawable_button>(
      button_font, "CLOTH", vec2(200.0f, 22.0f), white_color, button_label_color);

  std::unique_ptr<entity_base> cloth_si_switch_button =
      std::make_unique<entity_button>(1,
                                      std::move(cloth_moveable_button),
                                      std::move(cloth_button_drawable),
                                      _world,
                                      std::make_unique<scene_cloth>(_world));
  _UI_elements.push_back(std::move(cloth_si_switch_button));

  // switch to galaxy simulation button
  std::unique_ptr<moveable_button> galaxy_button =
      std::make_unique<moveable_button>(vec2(500.0f, 400.0f), vec2(200.0f, 22.0f));
  std::unique_ptr<drawable_button> galaxy_button_drawable = std::make_unique<drawable_button>(
      button_font, "GALAXY", vec2(200.0f, 22.0f), white_color, button_label_color);

  /*The (-1) on the initial_stars_num accounts for the possible presence of
  the extra star when holding down the button.
  Like so, the CUDA kernel will run the same amount of blocks, even if we add
  another star, and there is no performance hit.*/
  unsigned int initial_stars_num = 8 * 1024 - 1; // assume CUDA
  bool start_in_cuda_mode = true;
  // check CUDA Availability
  if (!cuda_utils::is_cuda_device_available())
  {
    initial_stars_num = 512;
    start_in_cuda_mode = false;
  }

  std::unique_ptr<entity_base> galaxy_si_switch_button = std::make_unique<entity_button>(
      1,
      std::move(galaxy_button),
      std::move(galaxy_button_drawable),
      _world,
      std::make_unique<scene_galaxy>(_world, start_in_cuda_mode, initial_stars_num));
  _UI_elements.push_back(std::move(galaxy_si_switch_button));

  // credits label
  std::unique_ptr<moveable_button> credits_moveable =
      std::make_unique<moveable_button>(vec2(800.0f, 660.0f), vec2(300.0f, 22.0f));
  std::unique_ptr<drawable_button> credits_drawable = std::make_unique<drawable_button>(
      button_font, "by Gianluca Delfino", vec2(300.0f, 22.0f), black_color, title_color);

  std::unique_ptr<entity_base> credits =
      std::make_unique<entity_base>(1, std::move(credits_moveable), std::move(credits_drawable));
  _UI_elements.push_back(std::move(credits));
}
