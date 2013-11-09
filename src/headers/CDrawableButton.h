#ifndef CDRAWABLEBUTTON_H
#define CDRAWABLEBUTTON_H

#include <string>
#include <memory>
#include "SDL.h"
#include "SDL_ttf.h"
#include "C2DVector.h"
#include "IDrawable.h"
#include "CSurface.h"
#include "CFont.h"

/**
* CDrawableButton defines how buttons are drawn on screen.
*/
class CDrawableButton : public IDrawable
{
public:
	CDrawableButton( std::shared_ptr< CFont > font_, SDL_Surface* destination_surf_, std::string label_, const C2DVector& size_, Uint32 background_color_, SDL_Color label_color_ );

	virtual std::unique_ptr< IDrawable > Clone() const;
	virtual void Draw( const C2DVector& pos, const C2DVector& orientation_ ) const;

private:
	std::string m_label;
	std::shared_ptr< CFont > mp_font;
	std::unique_ptr<CSurface> mp_text_surface;
	C2DVector m_size;
	Uint32 m_background_color;
	SDL_Color m_label_color;
};

#endif