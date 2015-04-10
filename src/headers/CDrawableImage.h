#ifndef CDRAWABLEIMAGE_H
#define CDRAWABLEIMAGE_H
#include "IDrawable.h"

class CDrawableImage: public IDrawable
{
public:
    CDrawableImage( CSurface* surf_to_blit_, SDL_Surface* destination_surf_ ):
        IDrawable( destination_surf_ ),
        mp_surface( surf_to_blit_ )
    {}

    CDrawableImage( const CDrawableImage& other_ ):
        IDrawable( other_.mp_destination ),
        mp_surface( other_.mp_surface )
    {}

    virtual std::unique_ptr< IDrawable > Clone() const
    {
        CDrawableImage* clone = new CDrawableImage( mp_surface, mp_destination );
        clone->m_scale = this->m_scale;
        clone->m_dimensions = this->m_dimensions;

        return std::unique_ptr<IDrawable>(clone);
    }

    CDrawableImage& operator=( const CDrawableImage& rhs )
    {
        if( this != &rhs)
        {
            IDrawable::operator=( rhs );
            this->mp_surface = rhs.mp_surface;
        }
        return *this;
    }

    virtual void Draw( const C2DVector& pos, const C2DVector& orientation_ ) const
    {
        this->mp_surface->ApplySurface( static_cast<int>(pos.x), static_cast<int>(pos.y), this->mp_destination );
    }

    virtual ~CDrawableImage()
    {
        delete this->mp_surface;
        this->mp_surface = NULL;
    }

private:
    CSurface* mp_surface;
};

#endif
