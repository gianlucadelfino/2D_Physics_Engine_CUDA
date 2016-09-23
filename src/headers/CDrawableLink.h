#ifndef CDRAWABLELINK_H
#define CDRAWABLELINK_H

#include "SDL.h"
#include "IDrawable.h"
#include "C2Dvector.h"

/**
* CDrawableLink defines how the cloth links are drawn on screen.
*/
class CDrawableLink : public IDrawable
{
public:
    CDrawableLink(SDL_Surface* destination_surf_);

    virtual void Draw(const C2DVector& pos_, const C2DVector& origin_) const;

private:
    virtual std::unique_ptr<IDrawable> DoClone() const;
};

#endif
