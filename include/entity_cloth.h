#ifndef CENTITYCLOTH_H
#define CENTITYCLOTH_H

#include "drawable_link.h"
#include "entity_particle.h"
#include "vec2.h"
#include <cmath>
#include <memory>
#include <vector>

/**
 * entity_cloth is a child of entity_particle to contain CEntities, following the
 * Pattern of the "Composite".
 * Calling update on the collection will update all the Entities in it and impose
 * all the constraints.
 */
class entity_cloth : public entity_base
{
public:
  entity_cloth(unsigned int id_,
               const vec2& initial_pos_,
               std::unique_ptr<drawable_link> drawable_,
               const entity_particle& seam_,
               unsigned int side_length_);

  entity_cloth(const entity_cloth&);
  entity_cloth& operator=(const entity_cloth&);

  virtual void handle_mouse_buttondown(std::shared_ptr<vec2> coords_);
  virtual void handle_mouse_buttonup(std::shared_ptr<vec2> coords_);

  virtual void update(const vec2& external_force_, float dt);
  virtual void draw() const;

private:
  std::vector<std::unique_ptr<entity_particle>> _collection;
  unsigned int _side_length;
  unsigned int _total_seams;
  vec2 _cloth_pos;

  // the more iteration, the more accurate the simulation
  static const int NUMBER_OF_ITERATIONS = 2;
  // define distance between cloth seams
  const float _max_dist;

  virtual void ApplyCollectiveConstraints(const unsigned int id);
  void EnforceMaxDist(int id_a, int id_b);
};

#endif
