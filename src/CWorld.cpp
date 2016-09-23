#include <iostream>
#include <thread>

#include "SDL_TTF.h"
#include "CWorld.h"
#include "CSceneMainMenu.h"
#include "CSceneLoadingScreen.h"
#include "CSceneGalaxy.h"

CWorld::CWorld(SDL_Surface* screen_) : mp_screen(screen_)
{
    if (!mp_screen)
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
    std::unique_ptr<IScene> main_menu(
        new CSceneMainMenu(this->mp_screen, *this));

    this->mp_cur_scene = std::move(main_menu);
    this->mp_cur_scene->Init();
}

void CWorld::Update(float dt) const
{
    if (this->mp_cur_scene)
        this->mp_cur_scene->Update(dt);
}

void CWorld::HandleEvent(const SDL_Event& event_)
{
    if (this->mp_cur_scene)
        this->mp_cur_scene->HandleEvent(*this, event_);
}

void CWorld::Draw() const
{
    if (this->mp_cur_scene)
        this->mp_cur_scene->Draw();
}

void CWorld::ChangeScene(std::unique_ptr<IScene> new_scene_)
{
    this->mp_cur_scene = std::move(std::unique_ptr<IScene>(
        new CSceneLoadingScreen(this->mp_screen, *this)));
    this->mp_cur_scene->Init();

    /******************************************************************
    * Spawn thread that loads new scene in background
    * DUE TO LIMITATIONS WITH VS12/13, WE CANNOT MOVE a unique_ptr to another
    * thread (it works in g++4.8.1.)
    * We have to resort to storing the next scene into another member variable
    * to be able to access it from the other thread :(

    * This is how we should have done it:
    * std::thread thrd_load_new_scene( &CWorld::LoadNextScene, this, std::move(
    *   new_scene_));
    *******************************************************************/
    this->mp_scene_to_load = std::move(new_scene_);

    std::thread thrd_load_new_scene([this]()
                                    {
                                        this->mp_scene_to_load->Init();
                                        this->mp_cur_scene =
                                            std::move(this->mp_scene_to_load);
                                    });
    thrd_load_new_scene.detach();
}

CWorld::~CWorld()
{
    // manually reset the scene pointer(calling its destruction), before we
    // deallocate SDL resources
    this->mp_cur_scene.reset(NULL);
    this->mp_scene_to_load.reset(NULL);

    // NOW we can free resources
    TTF_Quit();
}
