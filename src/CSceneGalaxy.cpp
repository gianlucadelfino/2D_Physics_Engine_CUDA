#include <string>

#include "CUDA_utils.cuh"

#include "CSceneGalaxy.h"
#include "CSceneMainMenu.h"

CSceneGalaxy::CSceneGalaxy(SDL_Renderer* renderer_,
                           CWorld& world_,
                           bool use_CUDA_,
                           unsigned int stars_nu_)
    : IScene(renderer_, world_, {0, 0, 0, 0}),
      _using_CUDA(use_CUDA_),
      _stars_num(stars_nu_),
      _font(std::make_unique<CFont>("assets/pcseniorSmall.ttf", 20)),
      _CUDA_capable_device_present(CUDA_utils::IsCUDACompatibleDeviceAvailable())
{
  // check CUDA Availability
  if (!_CUDA_capable_device_present)
  {
    if (_using_CUDA)
    {
      // add warning label
      SDL_Color black_color = {0, 0, 0, 0};
      SDL_Color label_color = {255, 255, 255, 0};
      std::unique_ptr<CDrawableButton> stars_nu_label_drawable =
          std::make_unique<CDrawableButton>(_font,
                                            _renderer,
                                            "CUDA compatible device not found, still using CPU!",
                                            C2DVector(220.0f, 22.0f),
                                            black_color,
                                            label_color);
      std::unique_ptr<CMoveableButton> stars_nu_label_moveable =
          std::make_unique<CMoveableButton>(C2DVector(250.0f, 660.0f), C2DVector(220.0f, 22.0f));

      std::unique_ptr<CEntity> still_using_CPU_label = std::make_unique<CEntity>(
          1, std::move(stars_nu_label_moveable), std::move(stars_nu_label_drawable));
      _UI_elements.push_back(std::move(still_using_CPU_label));
    }
    _using_CUDA = false;
  }

  // build prototype star for the galaxy
  std::unique_ptr<CDrawableStar> star_drawable = std::make_unique<CDrawableStar>(renderer_);
  std::unique_ptr<CPhysics> star_physics = std::make_unique<CPhysics>(1.0f, C2DVector(0.0f, 0.0f));
  star_physics->SetGravity(C2DVector(0.0f, 0.0f));
  std::unique_ptr<CMoveableParticle> star_moveable = std::make_unique<CMoveableParticle>();
  CEntityParticle star(
      1, std::move(star_moveable), std::move(star_drawable), std::move(star_physics));

  // create galaxy
  std::unique_ptr<CEntity> galaxy =
      std::make_unique<CEntityGalaxy>(2,
                                      C2DVector(600.0f, 300.0f),
                                      star,
                                      _stars_num,
                                      500,
                                      _using_CUDA); // id, pos, star prototype,
                                                    // stars number (max around
                                                    // 1024*9), size

  // add simulation Elements:
  _entities.push_back(std::move(galaxy));
}

CSceneGalaxy::CSceneGalaxy(const CSceneGalaxy& other_)
    : IScene(other_),
      _using_CUDA(other_._using_CUDA),
      _stars_num(other_._stars_num),
      _font(other_._font),
      _CUDA_capable_device_present(other_._CUDA_capable_device_present)
{
}

CSceneGalaxy& CSceneGalaxy::operator=(const CSceneGalaxy& rhs)
{
  if (this != &rhs)
  {
    IScene::operator=(rhs);
    _using_CUDA = rhs._using_CUDA;
    _stars_num = rhs._stars_num;
    _font = rhs._font;
    _CUDA_capable_device_present = rhs._CUDA_capable_device_present;
  }
  return *this;
}

void CSceneGalaxy::Init()
{
  // add UI elements to Scene

  // Toggle CUDA/CPU
  std::string CUDA_CPU_switch_label = _using_CUDA ? "SWITCH TO CPU" : "SWITCH TO CUDA";
  SDL_Color white_color = {255, 255, 255, SDL_ALPHA_OPAQUE};
  SDL_Color button_label_color = {0, 0, 0, 0};

  std::unique_ptr<CMoveableButton> CUDA_CPU_moveable_button =
      std::make_unique<CMoveableButton>(C2DVector(50.0f, 600.0f), C2DVector(320.0f, 22.0f));
  std::unique_ptr<CDrawableButton> CUDA_CPU_button_drawable =
      std::make_unique<CDrawableButton>(_font,
                                        _renderer,
                                        CUDA_CPU_switch_label,
                                        C2DVector(320.0f, 22.0f),
                                        white_color,
                                        button_label_color);

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
  std::unique_ptr<CEntity> CUDA_CPU_switch_button = std::make_unique<CEntityButton>(
      1,
      std::move(CUDA_CPU_moveable_button),
      std::move(CUDA_CPU_button_drawable),
      *_world,
      std::make_unique<CSceneGalaxy>(_renderer, *_world, !_using_CUDA, starting_stars_num));
  _UI_elements.push_back(std::move(CUDA_CPU_switch_button));

  // switch to Main Menu button
  std::unique_ptr<CMoveableButton> main_menu_moveable_button =
      std::make_unique<CMoveableButton>(C2DVector(400.0f, 600.0f), C2DVector(200.0f, 22.0f));
  std::unique_ptr<CDrawableButton> main_menu_button_drawable = std::make_unique<CDrawableButton>(
      _font, _renderer, "MAIN MENU", C2DVector(200.0f, 22.0f), white_color, button_label_color);

  std::unique_ptr<CEntity> cloth_si_switch_button =
      std::make_unique<CEntityButton>(1,
                                      std::move(main_menu_moveable_button),
                                      std::move(main_menu_button_drawable),
                                      *_world,
                                      std::make_unique<CSceneMainMenu>(_renderer, *_world));
  _UI_elements.push_back(std::move(cloth_si_switch_button));

  // more stars buttons
  unsigned int increment_num = _using_CUDA ? 1024 : 128;
  std::unique_ptr<CMoveableButton> more_stars_moveable_button =
      std::make_unique<CMoveableButton>(C2DVector(630.0f, 600.0f), C2DVector(220.0f, 22.0f));
  std::unique_ptr<CDrawableButton> more_stars_button_drawable = std::make_unique<CDrawableButton>(
      _font, _renderer, "MORE STARS", C2DVector(220.0f, 22.0f), white_color, button_label_color);

  std::unique_ptr<CEntity> more_stars_button = std::make_unique<CEntityButton>(
      1,
      std::move(more_stars_moveable_button),
      std::move(more_stars_button_drawable),
      *_world,
      std::make_unique<CSceneGalaxy>(_renderer, *_world, _using_CUDA, _stars_num + increment_num));
  _UI_elements.push_back(std::move(more_stars_button));

  // less stars buttons (appears only if there are more stars than
  // increment_num)
  if (_stars_num > increment_num)
  {
    std::unique_ptr<CDrawableButton> less_stars_button_drawable = std::make_unique<CDrawableButton>(
        _font, _renderer, "LESS STARS", C2DVector(220.0f, 22.0f), white_color, button_label_color);
    std::unique_ptr<CMoveableButton> less_stars_moveable_button =
        std::make_unique<CMoveableButton>(C2DVector(870.0f, 600.0f), C2DVector(220.0f, 22.0f));

    std::unique_ptr<CEntity> less_stars_button = std::make_unique<CEntityButton>(
        1,
        std::move(less_stars_moveable_button),
        std::move(less_stars_button_drawable),
        *_world,
        std::make_unique<CSceneGalaxy>(
            _renderer, *_world, _using_CUDA, _stars_num - increment_num));
    _UI_elements.push_back(std::move(less_stars_button));
  }
  // stars number label
  SDL_Color black_color = {0, 0, 0, 0};
  SDL_Color label_color = {255, 255, 255, 0};
  std::unique_ptr<CMoveableButton> stars_nu_label_moveable =
      std::make_unique<CMoveableButton>(C2DVector(20.0f, 20.0f), C2DVector(220.0f, 22.0f));
  std::unique_ptr<CDrawableButton> stars_nu_label_drawable =
      std::make_unique<CDrawableButton>(_font,
                                        _renderer,
                                        "Stars in simulation: " + std::to_string(_stars_num),
                                        C2DVector(220.0f, 22.0f),
                                        black_color,
                                        label_color);

  std::unique_ptr<CEntity> stars_nu_label = std::make_unique<CEntity>(
      1, std::move(stars_nu_label_moveable), std::move(stars_nu_label_drawable));
  _UI_elements.push_back(std::move(stars_nu_label));

  // instructions label (first part)
  std::unique_ptr<CMoveableButton> instructionsfirst_label_moveable =
      std::make_unique<CMoveableButton>(C2DVector(850.0f, 240.0f), C2DVector(220.0f, 22.0f));
  std::unique_ptr<CDrawableButton> instructionsfirst_label_drawable =
      std::make_unique<CDrawableButton>(_font,
                                        _renderer,
                                        "Each star gravitates",
                                        C2DVector(220.0f, 22.0f),
                                        black_color,
                                        label_color);

  std::unique_ptr<CEntity> instructionsfirst_label = std::make_unique<CEntity>(
      1, std::move(instructionsfirst_label_moveable), std::move(instructionsfirst_label_drawable));
  _UI_elements.push_back(std::move(instructionsfirst_label));

  // instructions label (second part)
  std::unique_ptr<CMoveableButton> instructionssecond_label_moveable =
      std::make_unique<CMoveableButton>(C2DVector(850.0f, 270.0f), C2DVector(220.0f, 22.0f));
  std::unique_ptr<CDrawableButton> instructionssecond_label_drawable =
      std::make_unique<CDrawableButton>(_font,
                                        _renderer,
                                        "with all the others.",
                                        C2DVector(220.0f, 22.0f),
                                        black_color,
                                        label_color);

  std::unique_ptr<CEntity> instructionssecond_label =
      std::make_unique<CEntity>(1,
                                std::move(instructionssecond_label_moveable),
                                std::move(instructionssecond_label_drawable));
  _UI_elements.push_back(std::move(instructionssecond_label));

  // instructions label (third part)
  std::unique_ptr<CMoveableButton> instructions_third_label_moveable =
      std::make_unique<CMoveableButton>(C2DVector(850.0f, 300.0f), C2DVector(220.0f, 22.0f));
  std::unique_ptr<CDrawableButton> instructions_third_label_drawable =
      std::make_unique<CDrawableButton>(_font,
                                        _renderer,
                                        "Click and hold to",
                                        C2DVector(220.0f, 22.0f),
                                        black_color,
                                        label_color);

  std::unique_ptr<CEntity> instructions_third_label =
      std::make_unique<CEntity>(1,
                                std::move(instructions_third_label_moveable),
                                std::move(instructions_third_label_drawable));
  _UI_elements.push_back(std::move(instructions_third_label));

  // instructions label (fourth half)
  std::unique_ptr<CMoveableButton> instructions_fourth_label_moveable =
      std::make_unique<CMoveableButton>(C2DVector(850.0f, 330.0f), C2DVector(220.0f, 22.0f));
  std::unique_ptr<CDrawableButton> instructions_fourth_label_drawable =
      std::make_unique<CDrawableButton>(
          _font, _renderer, "attract them.", C2DVector(220.0f, 22.0f), black_color, label_color);

  std::unique_ptr<CEntity> instructions_fourth_label =
      std::make_unique<CEntity>(1,
                                std::move(instructions_fourth_label_moveable),
                                std::move(instructions_fourth_label_drawable));
  _UI_elements.push_back(std::move(instructions_fourth_label));
}

CSceneGalaxy::~CSceneGalaxy()
{
  if (_using_CUDA)
    cudaDeviceReset();
}
