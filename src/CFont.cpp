#include <filesystem>
#include <stdexcept>
#include <string>

#include "SDL.h"
#include "SDL_ttf.h"

#include "CFont.h"

CFont::CFont(std::filesystem::path filename_, unsigned int size_) : _font(nullptr)
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

TTF_Font* CFont::GetFont() const { return _font; }

CFont::~CFont()
{
  TTF_CloseFont(_font);
  _font = nullptr;
}
