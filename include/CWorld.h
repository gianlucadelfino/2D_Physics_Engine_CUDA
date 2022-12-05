#ifndef CWORLD_H
#define CWORLD_H

#include <memory>
#include "SDL.h"
#include "SDL_events.h"
#include "IScene.h"

#include "CSceneGalaxy.h"

class IScene;

/**
* CWorld is a Finite-State-Machine that switches between Scenes ( aka "levels"
* or "simulations" )
*/
class CWorld
{
public:
    CWorld(SDL_Renderer* renderer_);

    void Update(float dt) const;
    void HandleEvent(const SDL_Event& event_);
    void Draw() const;

    void ChangeScene(std::unique_ptr<IScene> new_scene_);
    ~CWorld();

private:
    /*forbid copy and assignment*/
    CWorld(const CWorld&);
    CWorld& operator=(const CWorld&);

    std::unique_ptr<IScene> _cur_scene;
    SDL_Renderer* _renderer;
};

#endif
