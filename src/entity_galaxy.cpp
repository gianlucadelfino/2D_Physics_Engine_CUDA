#include <barrier>
#include <cassert>
#include <cstddef>
#include <list>
#include <stdexcept>
#include <thread>

#include "entity_galaxy.h"

entity_galaxy::entity_galaxy(unsigned int id_,
                             const vec2& initial_pos_,
                             const entity_particle& star_,
                             unsigned int star_number_,
                             float _bounding_box_side,
                             bool use_CUDA_)
    : entity_base(id_, std::make_unique<moveable_particle>(initial_pos_), nullptr),
      _rand_pos(),
      _rand_mass(),
      _using_CUDA(use_CUDA_),
      _original_star_num(star_number_)
{
  // first check proper number of stars is given
  if (_original_star_num <= 1)
  {
    _original_star_num = 1;
  }

  _rand_pos.set_real_bounds(-_bounding_box_side / 2.0f, _bounding_box_side / 2.0f);
  _rand_mass.set_real_bounds(1, 4);

  for (unsigned int i = 0; i < _original_star_num - 1;
       ++i) // Mind the (-1), we want to add another massive star in the centre!
  {
    std::unique_ptr<entity_particle> new_star = std::make_unique<entity_particle>(star_);

    // randomize position.
    vec2 pos = initial_pos_ + vec2(_rand_pos.rand_real(), _rand_pos.rand_real());
    vec2 initial_vel = vec2(_rand_pos.rand_real() / 200.0f, _rand_pos.rand_real() / 200.0f);

    // randomize mass
    const float mass = _rand_mass.rand_real();
    // adjust size accordingly (assuming constant density). TODO: randomize
    // density and have britness change by it. Make them turn into black
    // holes too
    new_star->set_scale(mass);

    new_star->reposition(pos);
    new_star->boost(pos + initial_vel);
    new_star->set_mass(mass);

    // set id
    new_star->SetId(i);

    // add a copy of star_ to the collection
    _collection.push_back(std::move(new_star));
  }

  // now add a supemassive black hole to be fixed in the center
  std::unique_ptr<entity_particle> super_massive_black_hole =
      std::make_unique<entity_particle>(star_);

  super_massive_black_hole->set_scale(10.0f);
  super_massive_black_hole->reposition(initial_pos_);
  float mass = 100.0f;

  super_massive_black_hole->set_mass(mass);

  // set id
  super_massive_black_hole->SetId(star_number_);

  // make super_massibe_black_hole static
  super_massive_black_hole->block();

  // add a copy of star_ to the collection
  _collection.push_back(std::move(super_massive_black_hole));
}

void entity_galaxy::set_use_cuda(bool use_CUDA_) { _using_CUDA = use_CUDA_; }

void entity_galaxy::update(const vec2& external_force_, float dt)
{
  if (_using_CUDA)
    update_cuda(external_force_, dt);
  else
    update_cpu(external_force_, dt);
}

void entity_galaxy::update_cuda(const vec2& external_force_, float dt)
{
  // instantiate some arrays to pass to the cuda kernels
  std::vector<float> star_positions_x(_collection.size());
  std::vector<float> star_positions_y(_collection.size());
  std::vector<float> masses(_collection.size());
  std::vector<float2> grav_forces(_collection.size());

  // load the vectors
  for (size_t i = 0; i < _collection.size(); ++i)
  {
    // convert vec2s to lighter float2s
    vec2 cur_pos = _collection[i]->get_position();

    star_positions_x[i] = cur_pos.x;
    star_positions_y[i] = cur_pos.y;
    masses[i] = _collection[i]->get_mass();
  }

  // call cuda kernel
  compute_grav(star_positions_x.data(),
               star_positions_y.data(),
               masses.data(),
               grav_forces.data(),
               _collection.size());

  for (size_t i = 0; i < _collection.size(); ++i)
  {
    float mass_i = _collection[i]->get_mass();
    _collection[i]->update(
        external_force_ + vec2(mass_i * grav_forces[i].x, mass_i * grav_forces[i].y), dt);
  }
}

/**
 * update computes the total gravity acting on each single star and calls update
 * on each. Scales as O(N^2)
 */
void entity_galaxy::update_cpu(const vec2& external_force_, float dt)
{
  std::vector<vec2> pairwise_forces(_collection.size() *
                                    _collection.size()); // We actully only need half of the matrix
                                                         // (force_ij=-force_ji), with no trace (no
                                                         // self interaction), however this would
                                                         // lead to complicated look up

  std::list<std::thread> pool;
  const size_t num_cores = std::thread::hardware_concurrency();
  const size_t batch_size =
      std::max((_collection.size() + num_cores - 1) / num_cores, static_cast<size_t>(8));

  const size_t pool_size = (_collection.size() + batch_size - 1) / batch_size;

  std::barrier sync_point(pool_size);

  auto work = [&](size_t start, size_t end)
  {
    // load the forces for each pair of star in the forces vector (N^2
    // operation)
    for (size_t i = start; i < end; ++i)
    {
      // keep j < i to compute only the upper half of the matrix. The matrix is
      // antisimmetric anyway, no need to compute it all!
      for (size_t j = 0; j < i; ++j)
      {
        float mass_i = _collection[i]->get_mass();
        vec2 pos_i = _collection[i]->get_position();

        float mass_j = _collection[j]->get_mass();
        vec2 pos_j = _collection[j]->get_position();

        // compute gravity
        vec2 r = pos_j - pos_i; // vector from i to j

        float dist = r.get_length();
        const float min_dist = 70.0f;    // to avoid infinities
        const float NEWTON_CONST = 2.0f; // higher than in CUDA case since we surely have less
                                         // stars, so let's make it more interesting!

        // force = G*m*M/ r^2
        vec2 force_ij = NEWTON_CONST * mass_i * mass_j / (dist * dist * dist + min_dist) *
                        r; // r is not normalized, therefore we divide by dist^3

        size_t index_ij = i + _collection.size() * j; // col + rows_num*row
        size_t index_ji = j + _collection.size() * i; // col + rows_num*row

        pairwise_forces[index_ij] = force_ij;
        pairwise_forces[index_ji] = (-1) * force_ij; // save redundant
                                                     // information for easy
                                                     // and fast look-ups
      }
    }

    sync_point.arrive_and_wait();

    // now add forces for each particle and apply it ( order N^2 )
    for (size_t i = start; i < end; ++i)
    {
      vec2 force_on_i = vec2(0.0f, 0.0f);
      for (size_t j = 0; j < _collection.size(); ++j) // sum all the column of forces
      {
        if (i != j)
        {
          size_t index_ij = i + _collection.size() * j; // col + rows_num*row
          force_on_i += pairwise_forces[index_ij];
        }
      }
      _collection[i]->update(external_force_ + force_on_i, dt);
    }
  };

  for (int s = 0;; s += batch_size)
  {
    if ((s + batch_size) >= _collection.size())
    {
      pool.emplace_back(std::thread(work, s, _collection.size()));
      break;
    }
    else
    {
      pool.emplace_back(std::thread(work, s, s + batch_size));
    }
  }

  assert(pool.size() == pool_size);

  for (auto&& t : pool)
  {
    if (t.joinable())
      t.join();
  }
}

void entity_galaxy::draw(SDL_Renderer* renderer_) const
{
  for (const auto& star : _collection)
  {
    star->draw(renderer_);
  }
}

void entity_galaxy::handle_mouse_buttondown(std::shared_ptr<vec2> coords_)
{
  // onclick add another massive star like the last one, in the place we
  // clicked
  if (_collection.size() <= (_original_star_num)) // _original_star_num is the current total
                                                  // of stars counting the black hole
  {
    // use the first star as a prototype
    std::unique_ptr<entity_particle> new_star =
        std::make_unique<entity_particle>(*(_collection[0]));

    new_star->reposition(*coords_);

    // make it super massive!
    const float mass = 10000.0f;

    new_star->set_mass(mass);
    new_star->set_scale(10.0f);
    // set id
    new_star->SetId(_collection.size());
    new_star->handle_mouse_buttondown(coords_);
    _collection.push_back(std::move(new_star));
  }
}
void entity_galaxy::handle_mouse_buttonup(std::shared_ptr<vec2> /*coords_*/)
{
  // remove the massive star just added
  if (_collection.size() > (_original_star_num)) // _original_star_num is the current total
                                                 // of stars counting the black hole
    _collection.pop_back();
}
