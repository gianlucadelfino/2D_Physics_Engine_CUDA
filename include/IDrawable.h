#ifndef IDRAWABLE_H
#define IDRAWABLE_H

#include <memory>
#include "SDL.h"
#include "C2DVector.h"
#include "CSurface.h"

/**
* IDrawable defines the interface for everything that needs to be drawn on
* screen.
*/
class IDrawable
{
public:
    IDrawable(SDL_Renderer* renderer_);

    // copy constructor and assignment operator
    IDrawable(const IDrawable& other_);
    IDrawable& operator=(const IDrawable& other_);

    std::unique_ptr<IDrawable> Clone() const; // non virtual to check that it
                                              // has been overridden (see "c++
                                              // coding standards" elem 54)

    virtual void Draw(const C2DVector& position_,
                      const C2DVector& orientation_) const = 0;

    virtual ~IDrawable();

    void SetSize(const C2DVector& _dimensions);
    void SetScale(float scale_);

protected:
    SDL_Renderer* _renderer;
    C2DVector _dimensions;
    float _scale{};

private:
    virtual std::unique_ptr<IDrawable> DoClone() const = 0;
};

#endif
