#ifndef CMOVEABLEBUTTON_H
#define CMOVEABLEBUTTON_H

#include "C2DVector.h"
#include "IMoveable.h"
#include <memory>

/**
 * CMoveableButton defines the IMoveable functions for a UI button.
 */
class CMoveableButton : public IMoveable
{
public:
  CMoveableButton(const C2DVector& initial_pos_, const C2DVector& size_);

  CMoveableButton(const CMoveableButton& other_);
  CMoveableButton& operator=(const CMoveableButton& other_);

  virtual bool IsHit(const C2DVector& coords_) const;

private:
  virtual std::unique_ptr<IMoveable> DoClone() const;

  C2DVector _size;
};
#endif
