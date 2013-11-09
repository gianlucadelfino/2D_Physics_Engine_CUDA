#ifndef IMOVEABLE_H
#define IMOVEABLE_H

#include <memory>
#include "C2DVector.h"

/**
* IMoveable. Anything on screen that can move should use an IMoveable, i.e. have a composite object that inherits from IMoveable.
*/
class IMoveable
{
public:
	IMoveable( float x_, float y_ );
	IMoveable( const C2DVector& pos_ );
	IMoveable( const C2DVector& pos_, const C2DVector& old_pos_, const C2DVector& orientation_);

	//copy constructor and assignment operator
	IMoveable( const IMoveable& other_);
	IMoveable& operator=( const IMoveable& other_ );

	/**
	* Boost offsets the particle which makes it gain speed
	*/
	void Boost( const C2DVector& new_position_ );

	/**
	* Reposition moves the particle and sets its speed to zero
	*/
	void Reposition( const C2DVector& new_position_ );

	/**
	* Translate moves particle and its velocity vector by a shift vector
	*/
	void Translate( const C2DVector& shift_ );

	/**
	* Clone is needed to create a deep copy when IMoveable is used as pimpl
	*/
	virtual std::unique_ptr< IMoveable > Clone() const = 0;

	virtual ~IMoveable();

	/**
	* ImposeConstraints solves the constraints imposed on the position/velocity of the objects and applies them
	*/
	virtual void ImposeConstraints();

	/**
	* SetConstraint add a constraint for the position
	*/
	virtual void SetConstraint( std::shared_ptr<C2DVector> origin_, const C2DVector& displacement_ );
	virtual void UnsetConstraint();

	virtual bool IsHit( const C2DVector& coords_ ) const { return false; }

	C2DVector pos;
	C2DVector old_pos;
	C2DVector orientation;

protected:
	//constraints. The origin needs to be a pointer to be dynamic
	std::shared_ptr<C2DVector> m_constraint_origin;
	C2DVector m_constraint_disp;
};
#endif