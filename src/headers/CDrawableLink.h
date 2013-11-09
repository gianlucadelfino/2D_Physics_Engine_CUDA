#ifndef CDRAWABLELINK_H
#define CDRAWABLELINK_H

#include "SDL.h"
#include "IDrawable.h"
#include "C2Dvector.h"

class CDrawableLink: public IDrawable
{
public:
	CDrawableLink( SDL_Surface* destination_surf_ );

	virtual std::unique_ptr< IDrawable > Clone() const;

	virtual void Draw( const C2DVector& pos_, const C2DVector& origin_ ) const;
};

#endif