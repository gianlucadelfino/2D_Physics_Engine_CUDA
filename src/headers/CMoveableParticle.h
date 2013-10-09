#ifndef CMOVEABLEPARTICLE_H
#define CMOVEABLEPARTICLE_H

#include <memory>
#include "IMoveable.h"
#include "C2DVector.h"

class CMoveableParticle : public IMoveable
{
public:
	CMoveableParticle():IMoveable( 0.0f, 0.0f ){}
	CMoveableParticle( float _x, float _y ):IMoveable( _x, _y){}
	CMoveableParticle( const C2DVector& _initial_pos):IMoveable( _initial_pos ){}

	CMoveableParticle( const CMoveableParticle& _other):IMoveable( _other ){}

	CMoveableParticle& operator=( const CMoveableParticle& _other )
	{
		if ( &_other != this )
		{
			IMoveable::operator=( _other );
		}
		return *this;
	}

	virtual IMoveable* Clone() const
	{
		return new CMoveableParticle( *this );
	}

	virtual bool IsHit( const C2DVector& _coords ) const
	{
		//check if _coords is in within the bounding box
		bool check_x = _coords.x < (pos.x + boundingbox_half_side) && _coords.x > ( pos.x - boundingbox_half_side );
		bool check_y = _coords.y < (pos.y + boundingbox_half_side) && _coords.y > ( pos.y - boundingbox_half_side );

		return check_x && check_y;
	}

private:
	static const int boundingbox_half_side = 20;
};
#endif