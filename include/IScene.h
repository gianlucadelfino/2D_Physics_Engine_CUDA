#ifndef ISCENE_H
#define ISCENE_H

#include <vector>
#include <memory>
#include <string>
#include "CEntity.h"
#include "C2DVector.h"
#include "SDL.h"

class CWorld;

/**
* IScene defines the interface for all the scenes that the CWorld contain.
*/
class IScene
{
public:
    IScene(SDL_Renderer* renderer_, CWorld& world_, SDL_Color bgr_color_);
    IScene(const IScene& other_);
    IScene& operator=(const IScene& rhs);

    /**
    * Init loads the scene elements. Should be called before calling update on
    * it.
    */
    virtual void Init();

    /**
    * Update takes care of updating all the IEnities in the scene
    */
    virtual void Update(float dt);
    /**
    * HandleEvent dispatches the events (mouse/keyboars) to the UI and the other
    * IEntities of the scene
    */
    virtual void HandleEvent(CWorld& world_, const SDL_Event& event_);
    /**
    * Draw renders all the IEntities of the scene
    */
    void Draw() const;

    virtual ~IScene();

protected:
    SDL_Renderer* _renderer;
    CWorld* _world;
    SDL_Color _background_color;
    std::shared_ptr<C2DVector> _mouse_coords;
    std::vector<std::unique_ptr<CEntity>> _entities;
    std::vector<std::unique_ptr<CEntity>> _UI_elements;
};

#endif
