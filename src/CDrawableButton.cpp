#include <string>
#include <memory>

#include "SDL.h"
#include "SDL_ttf.h"

#include "CDrawableButton.h"
#include "C2DVector.h"
#include "IDrawable.h"
#include "CSurface.h"
#include "CFont.h"
CDrawableButton::CDrawableButton(std::shared_ptr<CFont> font_,
                                 SDL_Surface* destination_surf_,
                                 std::string label_,
                                 const C2DVector& size_,
                                 Uint32 background_color_,
                                 SDL_Color label_color_)
    : IDrawable(destination_surf_),
      m_label(label_),
      mp_font(font_),
      mp_text_surface(new CSurface(NULL)),
      m_size(size_),
      m_background_color(background_color_),
      m_label_color(label_color_)
{
    // build the label
    this->mp_text_surface->LoadSurface(TTF_RenderText_Solid(
        font_->GetFont(), label_.c_str(), this->m_label_color));
}

CDrawableButton::CDrawableButton(const CDrawableButton& other_)
    : IDrawable(other_),
      m_label(other_.m_label),
      mp_font(other_.mp_font),
      mp_text_surface(new CSurface(NULL)),
      m_size(other_.m_size),
      m_background_color(other_.m_background_color),
      m_label_color(other_.m_label_color)
{
    // build the label
    this->mp_text_surface->LoadSurface(TTF_RenderText_Solid(
        this->mp_font->GetFont(), this->m_label.c_str(), this->m_label_color));
}

CDrawableButton& CDrawableButton::operator=(const CDrawableButton& rhs)
{
    if (this != &rhs)
    {
        IDrawable::operator=(rhs);
        this->m_label = rhs.m_label;
        this->mp_font = rhs.mp_font;
        this->m_size = rhs.m_size;
        this->m_background_color = rhs.m_background_color;
        this->m_label_color = rhs.m_label_color;
    }
    return *this;
}

std::unique_ptr<IDrawable> CDrawableButton::DoClone() const
{
    return std::unique_ptr<IDrawable>(
        new CDrawableButton(this->mp_font,
                            this->mp_destination,
                            this->m_label,
                            this->m_size,
                            this->m_background_color,
                            this->m_label_color));
}

void CDrawableButton::Draw(const C2DVector& pos,
                           const C2DVector& /*orientation_*/) const
{
    // build the rectangle
    SDL_Rect button;
    button.x = static_cast<Sint16>(pos.x);
    button.y = static_cast<Sint16>(pos.y);
    button.w = static_cast<Uint16>(this->m_size.x);
    button.h = static_cast<Uint16>(this->m_size.y);
    // draw the rectangle
    SDL_FillRect(this->mp_destination, &button, this->m_background_color);

    // draw the label in the position + 10% padding
    this->mp_text_surface->ApplySurface(
        static_cast<int>(pos.x + this->m_size.x / 10),
        static_cast<int>(pos.y + this->m_size.y / 10),
        this->mp_destination);
}
