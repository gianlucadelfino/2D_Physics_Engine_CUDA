#ifndef CFONT_H
#define CFONT_H

#include <stdexcept>
#include <string>
#include <windows.h>
#include "SDL.h"
#include "SDL_TTF.h"

class CFont
{
public:
	CFont( std::string filename_, unsigned int size_ ):mp_font(NULL)
	{
		//to open the file we first need to know the path of the current executable,
		//because TTF_OpenFont only likes absolute paths :(
		char ownPath[MAX_PATH];
		GetModuleFileName( NULL, ownPath, sizeof(ownPath) );

		//remove the file name from the path
		std::string::size_type pos = std::string( ownPath ).find_last_of("\\");
		std::string font_path = std::string( ownPath ).substr(0, pos + 1 ) + filename_;

		//Now we can finally load the font
		this->mp_font = TTF_OpenFont( font_path.c_str(), size_ );
		if ( !this->mp_font )
			throw std::runtime_error("Could not open the font file!");
	}

	TTF_Font* GetFont() const { return mp_font; }

	~CFont()
	{
		TTF_CloseFont( this->mp_font );
		this->mp_font = NULL;
	}

private:
	//forbid copy and assignment.
	CFont( const CFont& );
	CFont& operator=( const CFont& );

	TTF_Font* mp_font;
};

#endif