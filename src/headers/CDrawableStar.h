#ifndef CDRAWABLESTAR_H
#define CDRAWABLESTAR_H

#include "SDL.h"
#include "C2Dvector.h"
#include "IDrawable.h"
#include "SDL_gfxPrimitives.h"

class CDrawableStar: public IDrawable
{
public:
	CDrawableStar( SDL_Surface* destination_surf_ ):
		IDrawable( destination_surf_)
	{}

	virtual std::unique_ptr< IDrawable > Clone() const
	{
		CDrawableStar* clone = new CDrawableStar( mp_destination );
		clone->m_scale = this->m_scale;
		clone->m_dimensions = this->m_dimensions;

		return std::unique_ptr< IDrawable >(clone);
	}

	virtual void Draw( const C2DVector& pos_, const C2DVector& orientation_  ) const
	{
		circleRGBA( this->mp_destination,
			static_cast<Sint16>(pos_.x),
			static_cast<Sint16>(pos_.y),
			static_cast<Sint16>(m_scale*0.4f),
			255,
			255,
			200,
			255);
	}
};

#endif