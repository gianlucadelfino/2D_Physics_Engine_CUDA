#include <memory>
#include <string>

#include "SDL.h"
#include "SDL_ttf.h"

#include "C2DVector.h"
#include "CDrawableButton.h"
#include "CFont.h"
#include "CSurface.h"
#include "IDrawable.h"
CDrawableButton::CDrawableButton(std::shared_ptr<CFont> font_,
                                 SDL_Renderer* renderer_,
                                 const std::string& label_,
                                 const C2DVector& size_,
                                 SDL_Color background_color_,
                                 SDL_Color label_color_)
    : IDrawable(renderer_),
      _label(label_),
      _font(font_),
      _size(size_),
      _background_color(background_color_),
      _label_color(label_color_)
{
  // build the label
  _text_surface.LoadSurface(TTF_RenderText_Solid(font_->GetFont(), label_.c_str(), _label_color));
}

CDrawableButton::CDrawableButton(const CDrawableButton& other_)
    : IDrawable(other_),
      _label(other_._label),
      _font(other_._font),
      _size(other_._size),
      _background_color(other_._background_color),
      _label_color(other_._label_color)
{
  // build the label
  _text_surface.LoadSurface(TTF_RenderText_Solid(_font->GetFont(), _label.c_str(), _label_color));
}

CDrawableButton& CDrawableButton::operator=(const CDrawableButton& rhs)
{
  if (this != &rhs)
  {
    IDrawable::operator=(rhs);
    _label = rhs._label;
    _font = rhs._font;
    _size = rhs._size;
    _background_color = rhs._background_color;
    _label_color = rhs._label_color;
  }
  return *this;
}

std::unique_ptr<IDrawable> CDrawableButton::DoClone() const
{
  return std::make_unique<CDrawableButton>(_font, _renderer, _label, _size, _background_color, _label_color);
}

void CDrawableButton::Draw(const C2DVector& pos, const C2DVector& /*orientation_*/) const
{
  // build the rectangle
  SDL_Rect button;
  button.x = static_cast<Sint16>(pos.x);
  button.y = static_cast<Sint16>(pos.y);
  button.w = static_cast<Uint16>(_size.x);
  button.h = static_cast<Uint16>(_size.y);
  // draw the rectangle
  SDL_SetRenderDrawColor(
      _renderer, _background_color.r, _background_color.g, _background_color.b, SDL_ALPHA_OPAQUE);
  SDL_RenderFillRect(_renderer, &button);

  // draw the label in the position + 10% padding
  SDL_Rect label_rect;
  label_rect.x = static_cast<int>(pos.x + _size.x / 10);
  label_rect.y = static_cast<int>(pos.y + _size.y / 10);
  label_rect.w = static_cast<Uint16>(_size.x * .8f);
  label_rect.h = static_cast<Uint16>(_size.y * .8f);

  std::unique_ptr<SDL_Texture, decltype(&SDL_DestroyTexture)> texture = {
      SDL_CreateTextureFromSurface(_renderer, _text_surface.GetSurface()), SDL_DestroyTexture};
  SDL_RenderCopy(_renderer, texture.get(), nullptr, &label_rect);
}
