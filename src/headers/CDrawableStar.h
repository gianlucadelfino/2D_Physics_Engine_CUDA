#ifndef CDRAWABLESTAR_H
#define CDRAWABLESTAR_H

#include "SDL.h"
#include "C2Dvector.h"
#include "IDrawable.h"
#include "SDL_gfxPrimitives.h"

/**
* CDrawableStar defines how stars are drawn on screen.
*/
class CDrawableStar: public IDrawable
{
public:
	CDrawableStar( SDL_Surface* destination_surf_ );

	virtual std::unique_ptr< IDrawable > Clone() const;

	virtual void Draw( const C2DVector& pos_, const C2DVector& orientation_  ) const;
};

#endif