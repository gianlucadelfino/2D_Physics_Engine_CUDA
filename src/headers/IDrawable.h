#ifndef IDRAWABLE_H
#define IDRAWABLE_H

#include <memory>
#include "SDL.h"
#include "C2DVector.h"
#include "CSurface.h"

/**
* IDrawable defines the interface for everything that needs to be drawn on screen.
*/
class IDrawable
{
public:
	IDrawable( SDL_Surface* destination_surf_ );

	//copy constructor and assignment operator
	IDrawable( const IDrawable& other_ );
	IDrawable& operator=( const IDrawable& other_ );

	virtual std::unique_ptr< IDrawable > Clone() const = 0;

	virtual void Draw( const C2DVector& position_, const C2DVector& orientation_ ) const = 0;

	virtual ~IDrawable();

	void SetSize( const C2DVector& _dimensions );
	void SetScale( float scale_ );

protected:
	SDL_Surface* mp_destination;
	C2DVector m_dimensions;
	float m_scale;
};

#endif