#include <string>

#include "cuda_utils.cuh"

#include "scene_galaxy.h"
#include "scene_main_menu.h"

scene_galaxy::scene_galaxy(world_manager& world_, bool use_CUDA_, unsigned int stars_nu_)
    : scene_base(world_, {0, 0, 0, 0}),
      _using_CUDA(use_CUDA_),
      _stars_num(stars_nu_),
      _font(std::make_unique<font_handler>("assets/pcseniorSmall.ttf", 20)),
      _CUDA_capable_device_present(cuda_utils::is_cuda_device_available())
{
  // check CUDA Availability
  if (!_CUDA_capable_device_present)
  {
    if (_using_CUDA)
    {
      // add warning label
      SDL_Color black_color = {0, 0, 0, 0};
      SDL_Color label_color = {255, 255, 255, 0};
      std::unique_ptr<drawable_button> stars_nu_label_drawable =
          std::make_unique<drawable_button>(_font,
                                            "CUDA compatible device not found, still using CPU!",
                                            vec2(220.0f, 22.0f),
                                            black_color,
                                            label_color);
      std::unique_ptr<moveable_button> stars_nu_label_moveable =
          std::make_unique<moveable_button>(vec2(250.0f, 660.0f), vec2(220.0f, 22.0f));

      std::unique_ptr<entity_base> still_using_CPU_label = std::make_unique<entity_base>(
          1, std::move(stars_nu_label_moveable), std::move(stars_nu_label_drawable));
      _UI_elements.push_back(std::move(still_using_CPU_label));
    }
    _using_CUDA = false;
  }

  // build prototype star for the galaxy
  std::unique_ptr<drawable_star> star_drawable = std::make_unique<drawable_star>();
  std::unique_ptr<physics_base> star_physics =
      std::make_unique<physics_base>(1.0f, vec2(0.0f, 0.0f));
  star_physics->set_gravity(vec2(0.0f, 0.0f));
  std::unique_ptr<moveable_particle> star_moveable = std::make_unique<moveable_particle>();
  entity_particle star(
      1, std::move(star_moveable), std::move(star_drawable), std::move(star_physics));

  // create galaxy
  std::unique_ptr<entity_base> galaxy =
      std::make_unique<entity_galaxy>(2,
                                      vec2(600.0f, 300.0f),
                                      star,
                                      _stars_num,
                                      500,
                                      _using_CUDA); // id, pos, star prototype,
                                                    // stars number (max around
                                                    // 1024*9), size

  // add simulation Elements:
  _entities.push_back(std::move(galaxy));
}

scene_galaxy::scene_galaxy(const scene_galaxy& other_)
    : scene_base(other_),
      _using_CUDA(other_._using_CUDA),
      _stars_num(other_._stars_num),
      _font(other_._font),
      _CUDA_capable_device_present(other_._CUDA_capable_device_present)
{
}

scene_galaxy& scene_galaxy::operator=(const scene_galaxy& rhs)
{
  if (this != &rhs)
  {
    scene_base::operator=(rhs);
    _using_CUDA = rhs._using_CUDA;
    _stars_num = rhs._stars_num;
    _font = rhs._font;
    _CUDA_capable_device_present = rhs._CUDA_capable_device_present;
  }
  return *this;
}

void scene_galaxy::init()
{
  // add UI elements to Scene

  // Toggle CUDA/CPU
  std::string CUDA_CPU_switch_label = _using_CUDA ? "SWITCH TO CPU" : "SWITCH TO CUDA";
  SDL_Color white_color = {255, 255, 255, SDL_ALPHA_OPAQUE};
  SDL_Color button_label_color = {0, 0, 0, 0};

  std::unique_ptr<moveable_button> CUDA_CPU_moveable_button =
      std::make_unique<moveable_button>(vec2(50.0f, 600.0f), vec2(320.0f, 22.0f));
  std::unique_ptr<drawable_button> CUDA_CPU_button_drawable = std::make_unique<drawable_button>(
      _font, CUDA_CPU_switch_label, vec2(320.0f, 22.0f), white_color, button_label_color);

  /*start from less stars if we are switching to CPU. The starting number of
  stars should be high ONLY if we clicked on "SWITCH TO CUDA"
  and we actually have a CUDA capable device. In all other instances we'll be
  using the CPU anyway, so better start with little number of stars.
  The (-1) on the CUDA side accounts for the possible presence of the extra
  star when holding down the button. Like so, the CUDA kernel will
  run the same amount of blocks, even if we add another star, and there is no
  performance hit.*/
  unsigned int starting_stars_num =
      !_using_CUDA && _CUDA_capable_device_present ? (8 * 1024 - 1) : 512;
  std::unique_ptr<entity_base> CUDA_CPU_switch_button = std::make_unique<entity_button>(
      1,
      std::move(CUDA_CPU_moveable_button),
      std::move(CUDA_CPU_button_drawable),
      _world,
      std::make_unique<scene_galaxy>(_world, !_using_CUDA, starting_stars_num));
  _UI_elements.push_back(std::move(CUDA_CPU_switch_button));

  // switch to Main Menu button
  std::unique_ptr<moveable_button> main_menu_moveable_button =
      std::make_unique<moveable_button>(vec2(400.0f, 600.0f), vec2(200.0f, 22.0f));
  std::unique_ptr<drawable_button> main_menu_button_drawable = std::make_unique<drawable_button>(
      _font, "MAIN MENU", vec2(200.0f, 22.0f), white_color, button_label_color);

  std::unique_ptr<entity_base> cloth_si_switch_button =
      std::make_unique<entity_button>(1,
                                      std::move(main_menu_moveable_button),
                                      std::move(main_menu_button_drawable),
                                      _world,
                                      std::make_unique<scene_main_menu>(_world));
  _UI_elements.push_back(std::move(cloth_si_switch_button));

  // more stars buttons
  const unsigned int increment_num = _using_CUDA ? 5024 : 128;
  std::unique_ptr<moveable_button> more_stars_moveable_button =
      std::make_unique<moveable_button>(vec2(630.0f, 600.0f), vec2(220.0f, 22.0f));
  std::unique_ptr<drawable_button> more_stars_button_drawable = std::make_unique<drawable_button>(
      _font, "MORE STARS", vec2(220.0f, 22.0f), white_color, button_label_color);

  std::unique_ptr<entity_base> more_stars_button = std::make_unique<entity_button>(
      1,
      std::move(more_stars_moveable_button),
      std::move(more_stars_button_drawable),
      _world,
      std::make_unique<scene_galaxy>(_world, _using_CUDA, _stars_num + increment_num));
  _UI_elements.push_back(std::move(more_stars_button));

  // less stars buttons (appears only if there are more stars than increment_num)
  if (_stars_num > increment_num)
  {
    std::unique_ptr<drawable_button> less_stars_button_drawable = std::make_unique<drawable_button>(
        _font, "LESS STARS", vec2(220.0f, 22.0f), white_color, button_label_color);
    std::unique_ptr<moveable_button> less_stars_moveable_button =
        std::make_unique<moveable_button>(vec2(870.0f, 600.0f), vec2(220.0f, 22.0f));

    std::unique_ptr<entity_base> less_stars_button = std::make_unique<entity_button>(
        1,
        std::move(less_stars_moveable_button),
        std::move(less_stars_button_drawable),
        _world,
        std::make_unique<scene_galaxy>(_world, _using_CUDA, _stars_num - increment_num));
    _UI_elements.push_back(std::move(less_stars_button));
  }
  // stars number label
  SDL_Color black_color = {0, 0, 0, 0};
  SDL_Color label_color = {255, 255, 255, 0};
  std::unique_ptr<moveable_button> stars_nu_label_moveable =
      std::make_unique<moveable_button>(vec2(20.0f, 20.0f), vec2(220.0f, 22.0f));
  std::unique_ptr<drawable_button> stars_nu_label_drawable =
      std::make_unique<drawable_button>(_font,
                                        "Stars in simulation: " + std::to_string(_stars_num),
                                        vec2(220.0f, 22.0f),
                                        black_color,
                                        label_color);

  std::unique_ptr<entity_base> stars_nu_label = std::make_unique<entity_base>(
      1, std::move(stars_nu_label_moveable), std::move(stars_nu_label_drawable));
  _UI_elements.push_back(std::move(stars_nu_label));

  // instructions label (first part)
  std::unique_ptr<moveable_button> instructionsfirst_label_moveable =
      std::make_unique<moveable_button>(vec2(850.0f, 240.0f), vec2(220.0f, 22.0f));
  std::unique_ptr<drawable_button> instructionsfirst_label_drawable =
      std::make_unique<drawable_button>(
          _font, "Each star gravitates", vec2(220.0f, 22.0f), black_color, label_color);

  std::unique_ptr<entity_base> instructionsfirst_label = std::make_unique<entity_base>(
      1, std::move(instructionsfirst_label_moveable), std::move(instructionsfirst_label_drawable));
  _UI_elements.push_back(std::move(instructionsfirst_label));

  // instructions label (second part)
  std::unique_ptr<moveable_button> instructionssecond_label_moveable =
      std::make_unique<moveable_button>(vec2(850.0f, 270.0f), vec2(220.0f, 22.0f));
  std::unique_ptr<drawable_button> instructionssecond_label_drawable =
      std::make_unique<drawable_button>(
          _font, "with all the others.", vec2(220.0f, 22.0f), black_color, label_color);

  std::unique_ptr<entity_base> instructionssecond_label =
      std::make_unique<entity_base>(1,
                                    std::move(instructionssecond_label_moveable),
                                    std::move(instructionssecond_label_drawable));
  _UI_elements.push_back(std::move(instructionssecond_label));

  // instructions label (third part)
  std::unique_ptr<moveable_button> instructions_third_label_moveable =
      std::make_unique<moveable_button>(vec2(850.0f, 300.0f), vec2(220.0f, 22.0f));
  std::unique_ptr<drawable_button> instructions_third_label_drawable =
      std::make_unique<drawable_button>(
          _font, "Click and hold to", vec2(220.0f, 22.0f), black_color, label_color);

  std::unique_ptr<entity_base> instructions_third_label =
      std::make_unique<entity_base>(1,
                                    std::move(instructions_third_label_moveable),
                                    std::move(instructions_third_label_drawable));
  _UI_elements.push_back(std::move(instructions_third_label));

  // instructions label (fourth half)
  std::unique_ptr<moveable_button> instructions_fourth_label_moveable =
      std::make_unique<moveable_button>(vec2(850.0f, 330.0f), vec2(220.0f, 22.0f));
  std::unique_ptr<drawable_button> instructions_fourth_label_drawable =
      std::make_unique<drawable_button>(
          _font, "attract them.", vec2(220.0f, 22.0f), black_color, label_color);

  std::unique_ptr<entity_base> instructions_fourth_label =
      std::make_unique<entity_base>(1,
                                    std::move(instructions_fourth_label_moveable),
                                    std::move(instructions_fourth_label_drawable));
  _UI_elements.push_back(std::move(instructions_fourth_label));
}

scene_galaxy::~scene_galaxy()
{
  if (_using_CUDA)
    cudaDeviceReset();
}
