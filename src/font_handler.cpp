#include <filesystem>
#include <stdexcept>
#include <string>

#include "SDL.h"
#include "SDL_ttf.h"

#include "font_handler.h"

font_handler::font_handler(std::filesystem::path filename_, unsigned int size_) : _font(nullptr)
{
  // to open the file we first need to know the path of the current
  // executable,
  // because TTF_OpenFont only likes absolute paths :(
  const std::string font_path = std::filesystem::absolute(filename_).string();

  // Now we can finally load the font
  _font = TTF_OpenFont(font_path.c_str(), size_);
  if (!_font)
    throw std::runtime_error("Could not open the font file!");
}

TTF_Font* font_handler::GetFont() const { return _font; }

font_handler::~font_handler()
{
  TTF_CloseFont(_font);
  _font = nullptr;
}
