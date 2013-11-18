#include "CDrawableLink.h"
#include "SDL.h"
#include "SDL_gfxPrimitives.h"
#include "IDrawable.h"
#include <memory>

CDrawableLink::CDrawableLink( SDL_Surface* destination_surf_ ):
	IDrawable( destination_surf_)
{}

std::unique_ptr< IDrawable > CDrawableLink::DoClone() const
{
	CDrawableLink* clone = new CDrawableLink( this->mp_destination );
	clone->m_scale = this->m_scale;
	clone->m_dimensions = this->m_dimensions;

	return std::unique_ptr< IDrawable >(clone);
}

void CDrawableLink::Draw( const C2DVector& pos_, const C2DVector& origin_ ) const
{
	//draw line
	lineRGBA( this->mp_destination,
		static_cast<Sint16>(origin_.x), static_cast<Sint16>(origin_.y),
		static_cast<Sint16>(pos_.x), static_cast<Sint16>(pos_.y),
		0, 0, 0, 255);
}