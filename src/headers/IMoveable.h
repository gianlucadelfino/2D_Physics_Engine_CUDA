/*
*  IMoveable.h
*  MagicMine
*
*  Created by gianluca on 1/26/13.
*  Copyright 2013 __MyCompanyName__. All rights reserved.
*
*/
#ifndef IMOVEABLE_H
#define IMOVEABLE_H

#include <memory>
#include "C2DVector.h"

/**
* Anything on screen that can move should use an IMoveable, i.e. have a composite object that inherits from IMoveable.
*/
class IMoveable
{
public:
	IMoveable( float _x, float _y ):
		pos( _x, _y ),
		old_pos( _x, _y )
	{}

	IMoveable( const C2DVector& _pos ):
		pos( _pos ),
		old_pos( _pos )
	{}
	IMoveable( const C2DVector& _pos, const C2DVector& _old_pos):
		pos( _pos ),
		old_pos( _old_pos )
	{}

	//copy constructor and assignment operator
	IMoveable( const IMoveable& _other):
		pos( _other.pos),
		old_pos( _other.old_pos)
	{}
	IMoveable& operator=( const IMoveable& _other )
	{
		if ( &_other != this )
		{
			this->pos = _other.pos;
			this->old_pos = _other.old_pos;
		}
		return *this;
	}

	/*
	* Boost offsets the particle which makes it gain speed
	*/
	void Boost( const C2DVector& _new_position )
	{
		this->pos = _new_position;
	}

	/*
	* Reposition moves the particle and sets its speed to zero
	*/
	void Reposition( const C2DVector& _new_position )
	{
		this->pos = _new_position;
		this->old_pos = _new_position;
	}

	/*
	* Translate moves particle and its velocity vector by a shift vector
	*/
	void Translate( const C2DVector& _shift )
	{
		this->pos += _shift;
		this->old_pos += _shift;
	}

	//need this to use in copy constructors!
	virtual IMoveable* Clone() const = 0;

	virtual ~IMoveable()
	{}

	virtual void ImposeConstraints()
	{
		if ( m_constraint_origin ){
			//remind that constraint origin is a pointer because the constraint can move ( e.g. mouse )
			pos = *m_constraint_origin + m_constraint_disp;
		}
	}

	virtual void SetConstraint( std::shared_ptr<C2DVector> _origin, const C2DVector& _displacement )
	{
		m_constraint_origin = _origin;
		m_constraint_disp = _displacement;
	}

	virtual void UnsetConstraint()
	{
		m_constraint_origin.reset();
		m_constraint_disp = C2DVector( 0.0f, 0.0f );
	}

	virtual bool IsHit( const C2DVector& _coords ) const = 0;

	C2DVector pos;
	C2DVector old_pos;

protected:
	//constraints. The origin needs to be a pointer to be dynamic
	std::shared_ptr<C2DVector> m_constraint_origin;
	C2DVector m_constraint_disp;
};
#endif