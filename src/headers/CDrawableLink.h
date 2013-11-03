#ifndef CDRAWABLELINK_H
#define CDRAWABLELINK_H

#include "SDL.h"
#include "SDL_gfxPrimitives.h"
#include "IDrawable.h"
#include "C2Dvector.h"

class CDrawableLink: public IDrawable
{
public:
	CDrawableLink( SDL_Surface* destination_surf_ ):
		IDrawable( destination_surf_)
	{}

	virtual std::unique_ptr< IDrawable > Clone() const
	{
		CDrawableLink* clone = new CDrawableLink( this->mp_destination );
		clone->m_scale = this->m_scale;
		clone->m_dimensions = this->m_dimensions;

		return std::unique_ptr< IDrawable >(clone);
	}

	virtual void Draw( const C2DVector& pos_, const C2DVector& origin_ ) const
	{
		//draw line
		lineRGBA( this->mp_destination,
			static_cast<Sint16>(origin_.x), static_cast<Sint16>(origin_.y),
			static_cast<Sint16>(pos_.x), static_cast<Sint16>(pos_.y),
			0, 0, 0, 255);
	}
};

#endif