#include <memory>
#include <string>

#include "SDL.h"
#include "SDL_ttf.h"

#include "drawable_base.h"
#include "drawable_button.h"
#include "font_handler.h"
#include "surface_handler.h"
#include "vec2.h"
drawable_button::drawable_button(std::shared_ptr<font_handler> font_,
                                 const std::string& label_,
                                 const vec2& size_,
                                 SDL_Color background_color_,
                                 SDL_Color label_color_)
    : _label(label_),
      _font(font_),
      _size(size_),
      _background_color(background_color_),
      _label_color(label_color_)
{
  // build the label
  _text_surface.load_surface(TTF_RenderText_Solid(font_->get(), label_.c_str(), _label_color));
}

drawable_button::drawable_button(const drawable_button& other_)
    : drawable_base(other_),
      _label(other_._label),
      _font(other_._font),
      _size(other_._size),
      _background_color(other_._background_color),
      _label_color(other_._label_color)
{
  // build the label
  _text_surface.load_surface(TTF_RenderText_Solid(_font->get(), _label.c_str(), _label_color));
}

drawable_button& drawable_button::operator=(const drawable_button& rhs)
{
  if (this != &rhs)
  {
    drawable_base::operator=(rhs);
    _label = rhs._label;
    _font = rhs._font;
    _size = rhs._size;
    _background_color = rhs._background_color;
    _label_color = rhs._label_color;
  }
  return *this;
}

std::unique_ptr<drawable_base> drawable_button::do_clone() const
{
  return std::make_unique<drawable_button>(_font, _label, _size, _background_color, _label_color);
}

void drawable_button::draw(SDL_Renderer* renderer_,
                           const vec2& pos,
                           const vec2& /*orientation_*/) const
{
  // build the rectangle
  SDL_Rect button;
  button.x = static_cast<Sint16>(pos.x);
  button.y = static_cast<Sint16>(pos.y);
  button.w = static_cast<Uint16>(_size.x);
  button.h = static_cast<Uint16>(_size.y);
  // draw the rectangle
  SDL_SetRenderDrawColor(
      renderer_, _background_color.r, _background_color.g, _background_color.b, SDL_ALPHA_OPAQUE);
  SDL_RenderFillRect(renderer_, &button);

  // draw the label in the position + 10% padding
  SDL_Rect label_rect;
  label_rect.x = static_cast<int>(pos.x + _size.x / 10);
  label_rect.y = static_cast<int>(pos.y + _size.y / 10);
  label_rect.w = static_cast<Uint16>(_size.x * .8f);
  label_rect.h = static_cast<Uint16>(_size.y * .8f);

  std::unique_ptr<SDL_Texture, decltype(&SDL_DestroyTexture)> texture = {
      SDL_CreateTextureFromSurface(renderer_, _text_surface.get()), SDL_DestroyTexture};
  SDL_RenderCopy(renderer_, texture.get(), nullptr, &label_rect);
}
