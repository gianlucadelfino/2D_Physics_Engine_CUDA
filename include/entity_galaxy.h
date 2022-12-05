#ifndef CENTITYGALAXY_H
#define CENTITYGALAXY_H

#include <memory>
#include <vector>

#include "entity_base.h"
#include "entity_particle.h"
#include "moveable_base.h"
#include "moveable_particle.h"
#include "random_generator.h"
#include "vec2.h"

#include "cuda_runtime.h"
#include "device_compute_grav_forces.h"

/**
 * entity_galaxy is the Entity that manages the galaxy simulation. Given a Star
 * Entity as a prototype, the constructor will populate the galaxy
 * with star_number_ elements.
 * entity_galaxy follows the "Composite" patter: it IS an IEnity and manages many
 * IEnities underneath it.
 */
class entity_galaxy : public entity_base
{
public:
  entity_galaxy(unsigned int id_,
                const vec2& initial_pos_,
                const entity_particle& star_,
                unsigned int star_number_,
                float max_dist_,
                bool use_CUDA_);

  virtual void update(const vec2& external_force_, float dt);
  virtual void draw() const;

  virtual void handle_mouse_buttondown(std::shared_ptr<vec2> coords_);
  virtual void handle_mouse_buttonup(std::shared_ptr<vec2> coords_);

  /**
   * SetsUseCUDA sets whether to use the CUDA and the GPU or the CPU
   */
  void set_use_cuda(bool use_CUDA_);

private:
  // forbid copy and assignment
  entity_galaxy(const entity_galaxy&);
  entity_galaxy& operator=(const entity_galaxy&);

  void update_cuda(const vec2& external_force_, float dt);
  void update_cpu(const vec2& external_force_, float dt);

  random_generator _rand_pos;
  random_generator _rand_mass;

  bool _using_CUDA;

  typedef std::vector<std::unique_ptr<entity_particle>> StarList;
  StarList _collection;
  unsigned int _original_star_num;
};

#endif
