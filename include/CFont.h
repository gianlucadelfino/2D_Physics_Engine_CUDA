#ifndef CFONT_H
#define CFONT_H

#include <string>
#include <filesystem>

#include "SDL.h"
#include "SDL_ttf.h"

/**
* CFont is a RAII class to hold the font resource.
*/
class CFont
{
public:
    /**
    * CFont load the resources for the font.
    * @param filename_ the name of the file, with extension, relative to the exe
    * folder
    * @param size_ font size in "points"
    * @throw runtime_error if font file failed to open
    */
    CFont(std::filesystem::path filename_, unsigned int size_);

    TTF_Font* GetFont() const;
    ~CFont();

private:
    // forbid copy and assignment.
    CFont(const CFont&);
    CFont& operator=(const CFont&);

    TTF_Font* _font;
};

#endif
