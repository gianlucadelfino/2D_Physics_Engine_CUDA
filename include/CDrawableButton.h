#ifndef CDRAWABLEBUTTON_H
#define CDRAWABLEBUTTON_H

#include "C2DVector.h"
#include "CFont.h"
#include "CSurface.h"
#include "IDrawable.h"
#include "SDL.h"
#include "SDL_ttf.h"
#include <memory>
#include <string>

/**
 * CDrawableButton defines how buttons are drawn on screen.
 */
class CDrawableButton : public IDrawable
{
public:
  CDrawableButton(std::shared_ptr<CFont> font_,
                  SDL_Renderer* renderer_,
                  const std::string& label_,
                  const C2DVector& size_,
                  SDL_Color background_color_,
                  SDL_Color label_color_);
  CDrawableButton(const CDrawableButton& other_);
  CDrawableButton& operator=(const CDrawableButton& other_);

  virtual void Draw(const C2DVector& pos, const C2DVector& orientation_) const;

private:
  virtual std::unique_ptr<IDrawable> DoClone() const;

  std::string _label;
  std::shared_ptr<CFont> _font;
  CSurface _text_surface;
  C2DVector _size;
  SDL_Color _background_color;
  SDL_Color _label_color;
};

#endif
