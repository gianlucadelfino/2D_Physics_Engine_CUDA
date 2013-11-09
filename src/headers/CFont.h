#ifndef CFONT_H
#define CFONT_H

#include <string>

#include "SDL.h"
#include "SDL_TTF.h"

class CFont
{
public:
	CFont( std::string filename_, unsigned int size_ );

	TTF_Font* GetFont() const;
	~CFont();
private:
	//forbid copy and assignment.
	CFont( const CFont& );
	CFont& operator=( const CFont& );

	TTF_Font* mp_font;
};

#endif