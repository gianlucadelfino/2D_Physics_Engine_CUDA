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
	std::unique_ptr< IMoveable > Clone() const; //non virtual to check that it has been overridden (see "c++ coding standards" elem 54)

	virtual ~IMoveable();

	/**
	* ImposeConstraints solves the constraints imposed on the position/velocity of the objects and applies them
	*/
	virtual void ImposeConstraints();

	/**
	* SetConstraint adds the pointer to the position the current instance is supposed to stick to
	*/
	virtual void SetConstraint( std::shared_ptr<C2DVector> constrainted_pos_ );

	/**
	* UnsetConstraint frees the IMoveable from a set constraint
	*/
	virtual void UnsetConstraint();

	/**
	* IsHit receives a set of coordinates and returns true if hit (default to false)
	* @param coords_ the position to test if is hitting the IMoveable
	*/
	virtual bool IsHit( const C2DVector& /*coords_*/ ) const { return false; }

	C2DVector pos;
	C2DVector old_pos;
	C2DVector orientation;

protected:
	//constraints. The origin needs to be a pointer to be dynamic
	std::shared_ptr<C2DVector> m_constraint;
private:
	virtual std::unique_ptr< IMoveable > DoClone() const = 0;
};
#endif