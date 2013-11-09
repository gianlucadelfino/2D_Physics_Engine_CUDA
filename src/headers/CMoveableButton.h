#ifndef CMOVEABLEBUTTON_H
#define CMOVEABLEBUTTON_H

#include <memory>
#include "IMoveable.h"
#include "C2DVector.h"

/**
* CMoveableButton defines the IMoveable functions for a UI button.
*/
class CMoveableButton : public IMoveable
{
public:
	CMoveableButton( const C2DVector& initial_pos_, const C2DVector& size_ );

	CMoveableButton( const CMoveableButton& other_ );
	CMoveableButton& operator=( const CMoveableButton& other_ );
	virtual std::unique_ptr< IMoveable > Clone() const;

	virtual bool IsHit( const C2DVector& coords_ ) const;

private:
	C2DVector m_size;
};
#endif