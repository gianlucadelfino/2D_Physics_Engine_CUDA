#ifndef CDRAWABLESTAR_H
#define CDRAWABLESTAR_H

#include "SDL.h"
#include "C2Dvector.h"
#include "IDrawable.h"
#include "SDL_gfxPrimitives.h"

class CDrawableStar: public IDrawable
{
public:
	CDrawableStar( SDL_Surface* _destination_surf ):
		IDrawable( _destination_surf)
	{}

	virtual IDrawable* Clone() const
	{
		CDrawableStar* clone = new CDrawableStar( mp_destination );
		clone->m_scale = this->m_scale;
		clone->m_dimensions = this->m_dimensions;

		return clone;
	}

	virtual void Draw( const C2DVector& _pos ) const
	{
		circleRGBA( mp_destination,
			static_cast<Sint16>(_pos.x),
			static_cast<Sint16>(_pos.y),
			static_cast<Sint16>(m_scale*0.4f),
			255,
			255,
			200,
			255);
	}
};

#endif