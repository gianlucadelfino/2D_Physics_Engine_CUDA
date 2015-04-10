#include <memory>

#include "CEntity.h"
#include "C2DVector.h"
#include "IMoveable.h"
#include "IDrawable.h"

CEntity::CEntity( unsigned int id_, std::unique_ptr< IMoveable > moveable_, std::unique_ptr< IDrawable > drawable_ ):
    m_id( id_ ),
    mp_moveable( std::move( moveable_ ) ), //transfer ownership with move semantics
    mp_drawable( std::move( drawable_ ) )  //transfer ownership with move semantics
{}

CEntity::CEntity( const CEntity& other_ ):
    m_id( other_.m_id )
{
    //I need to deep copy the pointee, otherwise i am using the same IMovable!
    if ( other_.mp_moveable )
        this->mp_moveable = std::move( other_.mp_moveable->Clone() );
    else
        this->mp_moveable = NULL;

    //I need to deep copy the pointee, otherwise i am using the same IDrawable!
    if ( other_.mp_drawable )
        this->mp_drawable = std::move( other_.mp_drawable->Clone() );
    else
        this->mp_drawable = NULL;
}

CEntity& CEntity::operator=( const CEntity& rhs )
{
    if ( this != &rhs )
    {
        this->m_id = rhs.m_id;
        if ( rhs.mp_moveable )
            this->mp_moveable = std::move( rhs.mp_moveable->Clone() ); //I need to deep copy the pointee, otherwise i am using the same IMovable!
        else
            this->mp_moveable = NULL;

        if ( rhs.mp_drawable )
            this->mp_drawable = std::move( rhs.mp_drawable->Clone() ); //I need to deep copy the pointee, otherwise i am using the same IDrawable
        else
            this->mp_drawable = NULL;
    }
    return *this;
}

CEntity::~CEntity() {}

bool CEntity::IsHit( const C2DVector& coords_ ) const
{
    if ( this->mp_moveable )
        return this->mp_moveable->IsHit( coords_ );
    else
        return false;
}

void CEntity::Draw() const
{
    if( this->mp_drawable && this->mp_moveable  )
        mp_drawable->Draw( this->mp_moveable->pos, this->mp_moveable->orientation );
}
