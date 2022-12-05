#ifndef CMOVEABLEPARTICLE_H
#define CMOVEABLEPARTICLE_H

#include "moveable_base.h"
#include "vec2.h"
#include <memory>

/**
 * moveable_particle defines the moveable_base functions for a physical particle
 */
class moveable_particle : public moveable_base
{
public:
  moveable_particle();
  moveable_particle(float x_, float y_);
  moveable_particle(const vec2& initial_pos_);

  moveable_particle(const moveable_particle& other_);

  moveable_particle& operator=(const moveable_particle& other_);

  virtual bool is_hit(const vec2& coords_) const;

private:
  virtual std::unique_ptr<moveable_base> do_clone() const;

  static const int boundingbox_half_side = 20;
};
#endif
