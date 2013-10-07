#ifndef CDRAWABLELINK_H
#define CDRAWABLELINK_H

#include "SDL.h"
#include "SDL_gfxPrimitives.h"
#include "IDrawable.h"
#include "C2Dvector.h"

class CDrawableLink: public IDrawable
{
public:
	CDrawableLink( SDL_Surface* _destination_surf ):
		IDrawable( _destination_surf)
	{}

	virtual IDrawable* Clone() const
	{
		CDrawableLink* clone = new CDrawableLink( mp_destination );
		clone->m_origin = this->m_origin;
		clone->m_scale = this->m_scale;
		clone->m_dimensions = this->m_dimensions;

		return clone;
	}

	void SetOrigin( const C2DVector& _origin ) { m_origin = _origin; }

	virtual void Draw( const C2DVector& _pos ) const
	{
		//draw line
		lineRGBA( mp_destination,
			static_cast<Sint16>(m_origin.x), static_cast<Sint16>(m_origin.y),
			static_cast<Sint16>(_pos.x), static_cast<Sint16>(_pos.y),
			0, 0, 0, 255);
	}
private:
	C2DVector m_origin;
};

#endif