#ifndef CMOVEABLEPARTICLE_H
#define CMOVEABLEPARTICLE_H

#include <memory>
#include "IMoveable.h"
#include "C2DVector.h"

class CMoveableParticle : public IMoveable
{
public:
	CMoveableParticle():IMoveable( 0.0f, 0.0f ){}
	CMoveableParticle( float x_, float y_ ):IMoveable( x_, y_ ){}
	CMoveableParticle( const C2DVector& initial_pos_ ):IMoveable( initial_pos_ ){}

	CMoveableParticle( const CMoveableParticle& other_ ):IMoveable( other_ ){}

	CMoveableParticle& operator=( const CMoveableParticle& other_ )
	{
		if ( &other_ != this )
		{
			IMoveable::operator=( other_ );
		}
		return *this;
	}
	virtual std::unique_ptr< IMoveable > Clone() const
	{
		return std::unique_ptr< IMoveable >( new CMoveableParticle( *this ) );
	}

	virtual bool IsHit( const C2DVector& coords_ ) const
	{
		//check if coords_ is in within the bounding box
		bool check_x = coords_.x < (pos.x + boundingbox_half_side) && coords_.x > ( pos.x - boundingbox_half_side );
		bool check_y = coords_.y < (pos.y + boundingbox_half_side) && coords_.y > ( pos.y - boundingbox_half_side );

		return check_x && check_y;
	}

private:
	static const int boundingbox_half_side = 20;
};
#endif
