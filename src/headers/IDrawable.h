#ifndef IDRAWABLE_H
#define IDRAWABLE_H

#include "SDL.h"
#include "C2DVector.h"
#include "CSurface.h"

class IDrawable
{
public:
	IDrawable( SDL_Surface* _destination_surf ):
		mp_destination( _destination_surf ),
		m_dimensions(),
		m_scale(1)
	{}

	IDrawable( const IDrawable& _other ):
		mp_destination( _other.mp_destination ),
		m_dimensions( _other.m_dimensions ),
		m_scale( _other.m_scale )
	{}

	IDrawable& operator=( const IDrawable& _other )
	{
		if ( this != &_other )
		{
			this->mp_destination = _other.mp_destination;
			this->m_dimensions = _other.m_dimensions;
			this->m_scale = _other.m_scale;
		}
		return *this;
	}

	virtual IDrawable* Clone() const = 0;

	virtual void Draw( const C2DVector& pos ) const = 0;

	virtual ~IDrawable() = 0;

	void SetSize( const C2DVector& _dimensions ) { m_dimensions = _dimensions; }
	void SetScale( float _scale ) { m_scale = _scale; }

protected:
	SDL_Surface* mp_destination;
	C2DVector m_dimensions;
	float m_scale;
};

IDrawable::~IDrawable() {}

#endif