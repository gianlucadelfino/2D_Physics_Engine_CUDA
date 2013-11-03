#include "IDrawable.h"

IDrawable::IDrawable( SDL_Surface* destination_surf_ ):
	mp_destination( destination_surf_ ),
	m_dimensions(),
	m_scale(1)
{}

IDrawable::IDrawable( const IDrawable& other_ ):
	mp_destination( other_.mp_destination ),
	m_dimensions( other_.m_dimensions ),
	m_scale( other_.m_scale )
{}

IDrawable& IDrawable::operator=( const IDrawable& other_ )
{
	if ( this != &other_ )
	{
		this->mp_destination = other_.mp_destination;
		this->m_dimensions = other_.m_dimensions;
		this->m_scale = other_.m_scale;
	}
	return *this;
}

void IDrawable::SetSize( const C2DVector& _dimensions ) { m_dimensions = _dimensions; }
void IDrawable::SetScale( float scale_ ) { m_scale = scale_; }

IDrawable::~IDrawable() {}