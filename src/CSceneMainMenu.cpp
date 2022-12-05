#include <string>

#include "CSceneCloth.h"
#include "CSceneGalaxy.h"
#include "CSceneMainMenu.h"
#include "CUDA_utils.cuh"

CSceneMainMenu::CSceneMainMenu(SDL_Renderer* renderer_, CWorld& world_)
    : IScene(renderer_, world_, {0, 0, 0, 0})
{
}

CSceneMainMenu::CSceneMainMenu(const CSceneMainMenu& other_) : IScene(other_) {}

CSceneMainMenu& CSceneMainMenu::operator=(const CSceneMainMenu& rhs)
{
  if (this != &rhs)
  {
    IScene::operator=(rhs);
  }
  return *this;
}

void CSceneMainMenu::Init()
{
  // add UI elements to Scene
  SDL_Color white_color = {255, 255, 255, SDL_ALPHA_OPAQUE};
  SDL_Color black_color = {0, 0, 0, 0};
  SDL_Color button_label_color = {0, 0, 0, 0};

  // TITLE label
  std::shared_ptr<CFont> title_font = std::make_shared<CFont>("assets/pcseniorSmall.ttf", 30);
  SDL_Color title_color = {255, 255, 255, 0};
  std::unique_ptr<CMoveableButton> title_moveable =
      std::make_unique<CMoveableButton>(C2DVector(400.0f, 50.0f), C2DVector(300.0f, 31.0f));
  std::unique_ptr<CDrawableButton> title_drawable = std::make_unique<CDrawableButton>(
      title_font, _renderer, "SIMULINO 2000", C2DVector(200.0f, 31.0f), black_color, title_color);

  std::unique_ptr<CEntity> title =
      std::make_unique<CEntity>(1, std::move(title_moveable), std::move(title_drawable));
  _UI_elements.push_back(std::move(title));

  // switch to CLOTH simulation button
  std::shared_ptr<CFont> button_font = std::make_shared<CFont>("assets/pcseniorSmall.ttf", 20);

  std::unique_ptr<CMoveableButton> cloth_moveable_button =
      std::make_unique<CMoveableButton>(C2DVector(500.0f, 300.0f), C2DVector(200.0f, 22.0f));
  std::unique_ptr<CDrawableButton> cloth_button_drawable = std::make_unique<CDrawableButton>(
      button_font, _renderer, "CLOTH", C2DVector(200.0f, 22.0f), white_color, button_label_color);

  std::unique_ptr<CEntity> cloth_si_switch_button =
      std::make_unique<CEntityButton>(1,
                                      std::move(cloth_moveable_button),
                                      std::move(cloth_button_drawable),
                                      *_world,
                                      std::make_unique<CSceneCloth>(_renderer, *_world));
  _UI_elements.push_back(std::move(cloth_si_switch_button));

  // switch to galaxy simulation button
  std::unique_ptr<CMoveableButton> moveable_button =
      std::make_unique<CMoveableButton>(C2DVector(500.0f, 400.0f), C2DVector(200.0f, 22.0f));
  std::unique_ptr<CDrawableButton> galaxy_button_drawable = std::make_unique<CDrawableButton>(
      button_font, _renderer, "GALAXY", C2DVector(200.0f, 22.0f), white_color, button_label_color);

  /*The (-1) on the initial_stars_num accounts for the possible presence of
  the extra star when holding down the button.
  Like so, the CUDA kernel will run the same amount of blocks, even if we add
  another star, and there is no performance hit.*/
  unsigned int initial_stars_num = 8 * 1024 - 1; // assume CUDA
  bool start_in_cuda_mode = true;
  // check CUDA Availability
  if (!CUDA_utils::IsCUDACompatibleDeviceAvailable())
  {
    initial_stars_num = 512;
    start_in_cuda_mode = false;
  }

  std::unique_ptr<CEntity> galaxy_si_switch_button = std::make_unique<CEntityButton>(
      1,
      std::move(moveable_button),
      std::move(galaxy_button_drawable),
      *_world,
      std::make_unique<CSceneGalaxy>(_renderer, *_world, start_in_cuda_mode, initial_stars_num));
  _UI_elements.push_back(std::move(galaxy_si_switch_button));

  // credits label
  std::unique_ptr<CMoveableButton> credits_moveable =
      std::make_unique<CMoveableButton>(C2DVector(800.0f, 660.0f), C2DVector(300.0f, 22.0f));
  std::unique_ptr<CDrawableButton> credits_drawable =
      std::make_unique<CDrawableButton>(button_font,
                                        _renderer,
                                        "by Gianluca Delfino",
                                        C2DVector(300.0f, 22.0f),
                                        black_color,
                                        title_color);

  std::unique_ptr<CEntity> credits =
      std::make_unique<CEntity>(1, std::move(credits_moveable), std::move(credits_drawable));
  _UI_elements.push_back(std::move(credits));
}
