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

class CDrawableButton : public IDrawable
{
public:
	CDrawableButton( std::shared_ptr< CFont > font_, SDL_Surface* destination_surf_, std::string label_, const C2DVector& size_ ):
		IDrawable( destination_surf_ ),
		mp_font( font_ ),
		mp_text_surface( new CSurface( NULL ) ),
		m_size( size_ ),
		m_color( SDL_MapRGB( destination_surf_->format, 255, 255, 255 ) )
	{
		//build the label
		SDL_Color button_label_color = { 0, 0, 0, 0 };
		this->mp_text_surface->LoadSurface( TTF_RenderText_Solid( font_->GetFont(), label_.c_str(), button_label_color ) );
	}

	virtual std::unique_ptr< IDrawable > Clone() const
	{
		return std::unique_ptr< IDrawable >(new CDrawableButton( this->mp_font, this->mp_destination, this->m_label, this->m_size ));
	}

	virtual void Draw( const C2DVector& pos, const C2DVector& orientation_ ) const
	{
		//build the rectangle
		SDL_Rect button;
		button.x = static_cast<Sint16>(pos.x);
		button.y = static_cast<Sint16>(pos.y);
		button.w = static_cast<Uint16>(this->m_size.x);
		button.h = static_cast<Uint16>(this->m_size.y);
		//draw the rectangle
		SDL_FillRect( this->mp_destination, &button, this->m_color );

		//draw the label in the pos(ition) + 10% padding
		this->mp_text_surface->ApplySurface( static_cast<int>(pos.x + this->m_size.x/10) , static_cast<int>(pos.y + this->m_size.y/10 ), this->mp_destination );
	}

private:
	std::string m_label;
	std::shared_ptr< CFont > mp_font;
	std::unique_ptr<CSurface> mp_text_surface;
	C2DVector m_size;
	Uint32 m_color;
};

#endif