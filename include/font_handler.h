#ifndef CFONT_H
#define CFONT_H

#include <filesystem>
#include <string>

#include "SDL.h"
#include "SDL_ttf.h"

/**
 * font_handler is a RAII class to hold the font resource.
 */
class font_handler
{
public:
  /**
   * font_handler load the resources for the font.
   * @param filename_ the name of the file, with extension, relative to the exe
   * folder
   * @param size_ font size in "points"
   * @throw runtime_error if font file failed to open
   */
  font_handler(std::filesystem::path filename_, unsigned int size_);

  font_handler(const font_handler&) = delete;
  font_handler& operator=(const font_handler&) = delete;

  const TTF_Font* get() const;
  TTF_Font* get();
  ~font_handler();

private:
  TTF_Font* _font{};
};

#endif
