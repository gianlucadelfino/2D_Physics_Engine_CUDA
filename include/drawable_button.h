#ifndef CDRAWABLEBUTTON_H
#define CDRAWABLEBUTTON_H

#include "SDL.h"
#include "SDL_ttf.h"
#include "drawable_base.h"
#include "font_handler.h"
#include "surface_handler.h"
#include "vec2.h"
#include <memory>
#include <string>

/**
 * drawable_button defines how buttons are drawn on screen.
 */
class drawable_button : public drawable_base
{
public:
  drawable_button(std::shared_ptr<font_handler> font_,
                  const std::string& label_,
                  const vec2& size_,
                  SDL_Color background_color_,
                  SDL_Color label_color_);
  drawable_button(const drawable_button& other_);
  drawable_button& operator=(const drawable_button& other_);

  virtual void draw(SDL_Renderer* renderer_,
                    const vec2& pos,
                    const vec2& orientation_) const override;

private:
  virtual std::unique_ptr<drawable_base> do_clone() const override;

  std::string _label;
  std::shared_ptr<font_handler> _font;
  surface_handler _text_surface;
  vec2 _size;
  SDL_Color _background_color;
  SDL_Color _label_color;
};

#endif
