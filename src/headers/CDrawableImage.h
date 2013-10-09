#ifndef CDRAWABLEIMAGE_H
#define CDRAWABLEIMAGE_H
#include "IDrawable.h"

class CDrawableImage: public IDrawable
{
public:
	CDrawableImage( CSurface* _surf_to_blit, SDL_Surface* _destination_surf ):
		IDrawable( _destination_surf ),
		mp_surface( _surf_to_blit )
	{}

	CDrawableImage( const CDrawableImage& _other ):
		IDrawable(_other.mp_destination),
		mp_surface(_other.mp_surface)
	{}

	virtual IDrawable* Clone() const
	{
		CDrawableImage* clone = new CDrawableImage( mp_surface, mp_destination );
		clone->m_scale = this->m_scale;
		clone->m_dimensions = this->m_dimensions;

		return clone;
	}

	CDrawableImage& operator=( const CDrawableImage& rhs )
	{
		if( this != &rhs)
		{
			IDrawable::operator=( rhs );
			this->mp_surface = rhs.mp_surface;
		}
		return *this;
	}

	virtual void Draw( const C2DVector& pos ) const
	{
		mp_surface->ApplySurface( static_cast<int>(pos.x), static_cast<int>(pos.y), mp_destination );
	}

private:
	CSurface* mp_surface;
};
#endif