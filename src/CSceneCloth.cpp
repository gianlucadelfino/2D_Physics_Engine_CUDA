#include "CSceneCloth.h"
#include "CEntityCloth.h"
#include "CSceneMainMenu.h"
#include <memory>

CSceneCloth::CSceneCloth(SDL_Renderer* renderer_, CWorld& world_)
    : IScene(renderer_, world_, {255, 255, 255, 0}),
      _font(std::make_unique<CFont>("assets/pcseniorSmall.ttf", 20))
{
  // build cloth scene
  std::unique_ptr<CPhysics> phys_seam = std::make_unique<CPhysics>(1.0f);
  std::unique_ptr<IMoveable> sea_moveable = std::make_unique<CMoveableParticle>(50.0f, 50.0f);
  std::unique_ptr<CDrawableLink> cloth_drawable = std::make_unique<CDrawableLink>(renderer_);

  CEntityParticle seam(1, std::move(sea_moveable), nullptr, std::move(phys_seam));
  std::unique_ptr<CEntity> cloth =
      std::make_unique<CEntityCloth>(1,
                                     C2DVector(400.0f, 10.0f),
                                     std::move(cloth_drawable),
                                     seam,
                                     40); // id, pos, IDrawable, seam prototype, size
  // add simulation Elements:
  _entities.push_back(std::move(cloth));
}

CSceneCloth::CSceneCloth(const CSceneCloth& other_) : IScene(other_), _font(other_._font) {}

CSceneCloth& CSceneCloth::operator=(const CSceneCloth& rhs)
{
  if (this != &rhs)
  {
    IScene::operator=(rhs);
    _font = rhs._font;
  }
  return *this;
}

void CSceneCloth::Init()
{
  // add UI elements to Scene
  SDL_Color black_color = {0, 0, 0, 0};
  SDL_Color button_label_color = {255, 255, 255, 0};

  std::unique_ptr<CMoveableButton> main_menu_moveable_button =
      std::make_unique<CMoveableButton>(C2DVector(50.0f, 600.0f), C2DVector(200.0f, 22.0f));
  std::unique_ptr<CDrawableButton> main_menu_button_drawable = std::make_unique<CDrawableButton>(
      _font, _renderer, "MAIN MENU", C2DVector(200.0f, 22.0f), black_color, button_label_color);

  std::unique_ptr<CEntity> galaxy_si_switch_button =
      std::make_unique<CEntityButton>(1,
                                      std::move(main_menu_moveable_button),
                                      std::move(main_menu_button_drawable),
                                      *_world,
                                      std::make_unique<CSceneMainMenu>(_renderer, *_world));
  _UI_elements.push_back(std::move(galaxy_si_switch_button));

  // instructions label (first half)
  SDL_Color white_color = {255, 255, 255, SDL_ALPHA_OPAQUE};
  SDL_Color label_color = {0, 0, 0, 0};

  std::unique_ptr<CMoveableButton> instructionsfirst_label_moveable =
      std::make_unique<CMoveableButton>(C2DVector(900.0f, 300.0f), C2DVector(220.0f, 22.0f));
  std::unique_ptr<CDrawableButton> instructionsfirst_label_drawable =
      std::make_unique<CDrawableButton>(_font,
                                        _renderer,
                                        "click and hold to",
                                        C2DVector(220.0f, 22.0f),
                                        white_color,
                                        label_color);

  std::unique_ptr<CEntity> instructionsfirst_label = std::make_unique<CEntity>(
      1, std::move(instructionsfirst_label_moveable), std::move(instructionsfirst_label_drawable));
  _UI_elements.push_back(std::move(instructionsfirst_label));
  // instructions label (second half)
  std::unique_ptr<CMoveableButton> instructionssecond_label_moveable =
      std::make_unique<CMoveableButton>(C2DVector(900.0f, 330.0f), C2DVector(220.0f, 22.0f));
  std::unique_ptr<CDrawableButton> instructionssecond_label_drawable =
      std::make_unique<CDrawableButton>(
          _font, _renderer, "drag the cloth", C2DVector(220.0f, 22.0f), white_color, label_color);

  std::unique_ptr<CEntity> instructionssecond_label =
      std::make_unique<CEntity>(1,
                                std::move(instructionssecond_label_moveable),
                                std::move(instructionssecond_label_drawable));
  _UI_elements.push_back(std::move(instructionssecond_label));
}
