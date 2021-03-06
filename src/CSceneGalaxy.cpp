#include <string>

#include "CUDA_utils.cuh"

#include "CSceneGalaxy.h"
#include "CSceneMainMenu.h"

CSceneGalaxy::CSceneGalaxy(SDL_Surface* screen_,
                           CWorld& world_,
                           bool use_CUDA_,
                           unsigned int stars_num_)
    : IScene(screen_, world_, SDL_MapRGB(screen_->format, 0, 0, 0)),
      m_using_CUDA(use_CUDA_),
      m_stars_num(stars_num_),
      mp_font(new CFont("pcseniorSmall.ttf", 20)),
      m_CUDA_capable_device_present(
          CUDA_utils::IsCUDACompatibleDeviceAvailable())
{
    // check CUDA Availability
    if (!this->m_CUDA_capable_device_present)
    {
        if (this->m_using_CUDA)
        {
            // add warning label
            Uint32 black_color = SDL_MapRGB(this->mp_screen->format, 0, 0, 0);
            SDL_Color label_color = {255, 255, 255, 0};
            std::unique_ptr<CDrawableButton> stars_num_label_drawable(
                new CDrawableButton(
                    this->mp_font,
                    this->mp_screen,
                    "CUDA compatible device not found, still using CPU!",
                    C2DVector(220.0f, 22.0f),
                    black_color,
                    label_color));
            std::unique_ptr<CMoveableButton> stars_num_label_moveable(
                new CMoveableButton(C2DVector(250.0f, 660.0f),
                                    C2DVector(220.0f, 22.0f)));

            std::unique_ptr<CEntity> still_using_CPU_label(
                new CEntity(1,
                            std::move(stars_num_label_moveable),
                            std::move(stars_num_label_drawable)));
            this->m_UI_elements.push_back(std::move(still_using_CPU_label));
        }
        this->m_using_CUDA = false;
    }

    // build prototype star for the galaxy
    std::unique_ptr<CDrawableStar> star_drawable(new CDrawableStar(screen_));
    std::unique_ptr<CPhysics> star_physics(
        new CPhysics(1.0f, C2DVector(0.0f, 0.0f)));
    star_physics->SetGravity(C2DVector(0.0f, 0.0f));
    std::unique_ptr<CMoveableParticle> star_moveable(new CMoveableParticle());
    CEntityParticle star(1,
                         std::move(star_moveable),
                         std::move(star_drawable),
                         std::move(star_physics));

    // create galaxy
    std::unique_ptr<CEntity> galaxy(
        new CEntityGalaxy(2,
                          C2DVector(600.0f, 300.0f),
                          star,
                          this->m_stars_num,
                          500,
                          this->m_using_CUDA)); // id, pos, star prototype,
                                                // stars number (max around
                                                // 1024*9), size

    // add simulation Elements:
    this->m_entities.push_back(std::move(galaxy));
}

CSceneGalaxy::CSceneGalaxy(const CSceneGalaxy& other_)
    : IScene(other_),
      m_using_CUDA(other_.m_using_CUDA),
      m_stars_num(other_.m_stars_num),
      mp_font(other_.mp_font),
      m_CUDA_capable_device_present(other_.m_CUDA_capable_device_present)
{
}

CSceneGalaxy& CSceneGalaxy::operator=(const CSceneGalaxy& rhs)
{
    if (this != &rhs)
    {
        IScene::operator=(rhs);
        this->m_using_CUDA = rhs.m_using_CUDA;
        this->m_stars_num = rhs.m_stars_num;
        this->mp_font = rhs.mp_font;
        this->m_CUDA_capable_device_present = rhs.m_CUDA_capable_device_present;
    }
    return *this;
}

void CSceneGalaxy::Init()
{
    // add UI elements to Scene

    // Toggle CUDA/CPU
    std::string CUDA_CPU_switch_label =
        this->m_using_CUDA ? "SWITCH TO CPU" : "SWITCH TO CUDA";
    Uint32 white_color = SDL_MapRGB(this->mp_screen->format, 255, 255, 255);
    SDL_Color button_label_color = {0, 0, 0, 0};

    std::unique_ptr<CMoveableButton> CUDA_CPU_moveable_button(
        new CMoveableButton(C2DVector(50.0f, 600.0f),
                            C2DVector(320.0f, 22.0f)));
    std::unique_ptr<CDrawableButton> CUDA_CPU_button_drawable(
        new CDrawableButton(this->mp_font,
                            this->mp_screen,
                            CUDA_CPU_switch_label,
                            C2DVector(320.0f, 22.0f),
                            white_color,
                            button_label_color));

    /*start from less stars if we are switching to CPU. The starting number of
    stars should be high ONLY if we clicked on "SWITCH TO CUDA"
    and we actually have a CUDA capable device. In all other instances we'll be
    using the CPU anyway, so better start with little number of stars.
    The (-1) on the CUDA side accounts for the possible presence of the extra
    star when holding down the button. Like so, the CUDA kernel will
    run the same amount of blocks, even if we add another star, and there is no
    performance hit.*/
    unsigned int starting_stars_num =
        !this->m_using_CUDA && this->m_CUDA_capable_device_present ?
            (8 * 1024 - 1) :
            512;
    std::unique_ptr<CEntity> CUDA_CPU_switch_button(
        new CEntityButton(1,
                          std::move(CUDA_CPU_moveable_button),
                          std::move(CUDA_CPU_button_drawable),
                          *this->mr_world,
                          std::move(std::unique_ptr<IScene>(
                              new CSceneGalaxy(this->mp_screen,
                                               *this->mr_world,
                                               !this->m_using_CUDA,
                                               starting_stars_num)))));
    this->m_UI_elements.push_back(std::move(CUDA_CPU_switch_button));

    // switch to Main Menu button
    std::unique_ptr<CMoveableButton> main_menu_moveable_button(
        new CMoveableButton(C2DVector(400.0f, 600.0f),
                            C2DVector(200.0f, 22.0f)));
    std::unique_ptr<CDrawableButton> main_menu_button_drawable(
        new CDrawableButton(this->mp_font,
                            this->mp_screen,
                            "MAIN MENU",
                            C2DVector(200.0f, 22.0f),
                            white_color,
                            button_label_color));

    std::unique_ptr<CEntity> cloth_sim_switch_button(
        new CEntityButton(1,
                          std::move(main_menu_moveable_button),
                          std::move(main_menu_button_drawable),
                          *this->mr_world,
                          std::move(std::unique_ptr<IScene>(new CSceneMainMenu(
                              this->mp_screen, *this->mr_world)))));
    this->m_UI_elements.push_back(std::move(cloth_sim_switch_button));

    // more stars buttons
    unsigned int increment_num = this->m_using_CUDA ? 1024 : 128;
    std::unique_ptr<CMoveableButton> more_stars_moveable_button(
        new CMoveableButton(C2DVector(630.0f, 600.0f),
                            C2DVector(220.0f, 22.0f)));
    std::unique_ptr<CDrawableButton> more_stars_button_drawable(
        new CDrawableButton(this->mp_font,
                            this->mp_screen,
                            "MORE STARS",
                            C2DVector(220.0f, 22.0f),
                            white_color,
                            button_label_color));

    std::unique_ptr<CEntity> more_stars_button(
        new CEntityButton(1,
                          std::move(more_stars_moveable_button),
                          std::move(more_stars_button_drawable),
                          *this->mr_world,
                          std::move(std::unique_ptr<IScene>(new CSceneGalaxy(
                              this->mp_screen,
                              *this->mr_world,
                              this->m_using_CUDA,
                              this->m_stars_num + increment_num)))));
    this->m_UI_elements.push_back(std::move(more_stars_button));

    // less stars buttons (appears only if there are more stars than
    // increment_num)
    if (this->m_stars_num > increment_num)
    {
        std::unique_ptr<CDrawableButton> less_stars_button_drawable(
            new CDrawableButton(this->mp_font,
                                this->mp_screen,
                                "LESS STARS",
                                C2DVector(220.0f, 22.0f),
                                white_color,
                                button_label_color));
        std::unique_ptr<CMoveableButton> less_stars_moveable_button(
            new CMoveableButton(C2DVector(870.0f, 600.0f),
                                C2DVector(220.0f, 22.0f)));

        std::unique_ptr<CEntity> less_stars_button(new CEntityButton(
            1,
            std::move(less_stars_moveable_button),
            std::move(less_stars_button_drawable),
            *this->mr_world,
            std::move(std::unique_ptr<IScene>(
                new CSceneGalaxy(this->mp_screen,
                                 *this->mr_world,
                                 this->m_using_CUDA,
                                 this->m_stars_num - increment_num)))));
        this->m_UI_elements.push_back(std::move(less_stars_button));
    }
    // stars number label
    Uint32 black_color = SDL_MapRGB(this->mp_screen->format, 0, 0, 0);
    SDL_Color label_color = {255, 255, 255, 0};
    std::unique_ptr<CMoveableButton> stars_num_label_moveable(
        new CMoveableButton(C2DVector(20.0f, 20.0f), C2DVector(220.0f, 22.0f)));
    std::unique_ptr<CDrawableButton> stars_num_label_drawable(
        new CDrawableButton(this->mp_font,
                            this->mp_screen,
                            "Stars in simulation: " +
                                std::to_string(this->m_stars_num),
                            C2DVector(220.0f, 22.0f),
                            black_color,
                            label_color));

    std::unique_ptr<CEntity> stars_num_label(
        new CEntity(1,
                    std::move(stars_num_label_moveable),
                    std::move(stars_num_label_drawable)));
    this->m_UI_elements.push_back(std::move(stars_num_label));

    // instructions label (first part)
    std::unique_ptr<CMoveableButton> instructionsfirst_label_moveable(
        new CMoveableButton(C2DVector(850.0f, 240.0f),
                            C2DVector(220.0f, 22.0f)));
    std::unique_ptr<CDrawableButton> instructionsfirst_label_drawable(
        new CDrawableButton(this->mp_font,
                            this->mp_screen,
                            "Each star gravitates",
                            C2DVector(220.0f, 22.0f),
                            black_color,
                            label_color));

    std::unique_ptr<CEntity> instructionsfirst_label(
        new CEntity(1,
                    std::move(instructionsfirst_label_moveable),
                    std::move(instructionsfirst_label_drawable)));
    this->m_UI_elements.push_back(std::move(instructionsfirst_label));

    // instructions label (second part)
    std::unique_ptr<CMoveableButton> instructionssecond_label_moveable(
        new CMoveableButton(C2DVector(850.0f, 270.0f),
                            C2DVector(220.0f, 22.0f)));
    std::unique_ptr<CDrawableButton> instructionssecond_label_drawable(
        new CDrawableButton(this->mp_font,
                            this->mp_screen,
                            "with all the others.",
                            C2DVector(220.0f, 22.0f),
                            black_color,
                            label_color));

    std::unique_ptr<CEntity> instructionssecond_label(
        new CEntity(1,
                    std::move(instructionssecond_label_moveable),
                    std::move(instructionssecond_label_drawable)));
    this->m_UI_elements.push_back(std::move(instructionssecond_label));

    // instructions label (third part)
    std::unique_ptr<CMoveableButton> instructions_third_label_moveable(
        new CMoveableButton(C2DVector(850.0f, 300.0f),
                            C2DVector(220.0f, 22.0f)));
    std::unique_ptr<CDrawableButton> instructions_third_label_drawable(
        new CDrawableButton(this->mp_font,
                            this->mp_screen,
                            "Click and hold to",
                            C2DVector(220.0f, 22.0f),
                            black_color,
                            label_color));

    std::unique_ptr<CEntity> instructions_third_label(
        new CEntity(1,
                    std::move(instructions_third_label_moveable),
                    std::move(instructions_third_label_drawable)));
    this->m_UI_elements.push_back(std::move(instructions_third_label));

    // instructions label (fourth half)
    std::unique_ptr<CMoveableButton> instructions_fourth_label_moveable(
        new CMoveableButton(C2DVector(850.0f, 330.0f),
                            C2DVector(220.0f, 22.0f)));
    std::unique_ptr<CDrawableButton> instructions_fourth_label_drawable(
        new CDrawableButton(this->mp_font,
                            this->mp_screen,
                            "attract them.",
                            C2DVector(220.0f, 22.0f),
                            black_color,
                            label_color));

    std::unique_ptr<CEntity> instructions_fourth_label(
        new CEntity(1,
                    std::move(instructions_fourth_label_moveable),
                    std::move(instructions_fourth_label_drawable)));
    this->m_UI_elements.push_back(std::move(instructions_fourth_label));
}

CSceneGalaxy::~CSceneGalaxy()
{
    if (this->m_using_CUDA)
        cudaDeviceReset();
}
