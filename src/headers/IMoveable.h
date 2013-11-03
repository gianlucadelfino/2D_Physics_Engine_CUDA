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
	IMoveable( float x_, float y_ ):
		pos( x_, y_ ),
		old_pos( x_, y_ ),
		orientation( 0, 0 )
	{}

	IMoveable( const C2DVector& pos_ ):
		pos( pos_ ),
		old_pos( pos_ ),
		orientation( 0, 0 )
	{}
	IMoveable( const C2DVector& pos_, const C2DVector& old_pos_, const C2DVector& orientation_):
		pos( pos_ ),
		old_pos( old_pos_ ),
		orientation( orientation_ )
	{}

	//copy constructor and assignment operator
	IMoveable( const IMoveable& other_):
		pos( other_.pos),
		old_pos( other_.old_pos),
		orientation( other_.orientation )
	{}

	IMoveable& operator=( const IMoveable& other_ )
	{
		if ( &other_ != this )
		{
			this->pos = other_.pos;
			this->old_pos = other_.old_pos;
			this->orientation = other_.orientation;
		}
		return *this;
	}

	/*
	* Boost offsets the particle which makes it gain speed
	*/
	void Boost( const C2DVector& new_position_ )
	{
		this->pos = new_position_;
	}

	/*
	* Reposition moves the particle and sets its speed to zero
	*/
	void Reposition( const C2DVector& new_position_ )
	{
		this->pos = new_position_;
		this->old_pos = new_position_;
	}

	/*
	* Translate moves particle and its velocity vector by a shift vector
	*/
	void Translate( const C2DVector& shift_ )
	{
		this->pos += shift_;
		this->old_pos += shift_;
	}

	//need this to use in copy constructors!
	virtual std::unique_ptr< IMoveable > Clone() const = 0;

	virtual ~IMoveable()
	{}

	virtual void ImposeConstraints()
	{
		if ( m_constraint_origin ){
			//remind that constraint origin is a pointer because the constraint can move ( e.g. mouse )
			pos = *m_constraint_origin + m_constraint_disp;
		}
	}

	virtual void SetConstraint( std::shared_ptr<C2DVector> origin_, const C2DVector& displacement_ )
	{
		m_constraint_origin = origin_;
		m_constraint_disp = displacement_;
	}

	virtual void UnsetConstraint()
	{
		m_constraint_origin.reset();
		m_constraint_disp = C2DVector( 0.0f, 0.0f );
	}

	virtual bool IsHit( const C2DVector& coords_ ) const
	{
		return false;
	}

	C2DVector pos;
	C2DVector old_pos;
	C2DVector orientation;

protected:
	//constraints. The origin needs to be a pointer to be dynamic
	std::shared_ptr<C2DVector> m_constraint_origin;
	C2DVector m_constraint_disp;
};
#endif