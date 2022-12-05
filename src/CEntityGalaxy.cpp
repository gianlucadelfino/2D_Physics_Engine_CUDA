#include <stdexcept>

#include "CEntityGalaxy.h"

CEntityGalaxy::CEntityGalaxy(unsigned int id_,
                             const C2DVector& initial_pos_,
                             const CEntityParticle& star_,
                             unsigned int star_number_,
                             float _bounding_box_side,
                             bool use_CUDA_)
    : CEntity(id_, std::make_unique<CMoveableParticle>(initial_pos_), nullptr),
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

  _rand_pos.SetRealBounds(-_bounding_box_side / 2.0f, _bounding_box_side / 2.0f);
  _rand_mass.SetRealBounds(1, 4);

  for (unsigned int i = 0; i < _original_star_num - 1;
       ++i) // Mind the (-1), we want to add another massive star in the centre!
  {
    std::unique_ptr<CEntityParticle> new_star = std::make_unique<CEntityParticle>(star_);

    // randomize position.
    C2DVector pos = initial_pos_ + C2DVector(_rand_pos.RandReal(), _rand_pos.RandReal());
    C2DVector initial_vel = C2DVector(_rand_pos.RandReal() / 200.0f, _rand_pos.RandReal() / 200.0f);

    // randomize mass
    const float mass = _rand_mass.RandReal();
    // adjust size accordingly (assuming constant density). TODO: randomize
    // density and have britness change by it. Make them turn into black
    // holes too
    new_star->SetScale(mass);

    new_star->Reposition(pos);
    new_star->Boost(pos + initial_vel);
    new_star->SetMass(mass);

    // set id
    new_star->SetId(i);

    // add a copy of star_ to the collection
    _collection.push_back(std::move(new_star));
  }

  // now add a supemassive black hole to be fixed in the center
  std::unique_ptr<CEntityParticle> super_massive_black_hole =
      std::make_unique<CEntityParticle>(star_);

  super_massive_black_hole->SetScale(10.0f);
  super_massive_black_hole->Reposition(initial_pos_);
  float mass = 100.0f;

  super_massive_black_hole->SetMass(mass);

  // set id
  super_massive_black_hole->SetId(star_number_);

  // make super_massibe_black_hole static
  super_massive_black_hole->Block();

  // add a copy of star_ to the collection
  _collection.push_back(std::move(super_massive_black_hole));
}

void CEntityGalaxy::SetUseCUDA(bool use_CUDA_) { _using_CUDA = use_CUDA_; }

void CEntityGalaxy::Update(const C2DVector& external_force_, float dt)
{
  if (_using_CUDA)
    UpdateCUDA(external_force_, dt);
  else
    UpdateCPU(external_force_, dt);
}

void CEntityGalaxy::UpdateCUDA(const C2DVector& external_force_, float dt)
{
  // instantiate some arrays to pass to the cuda kernels
  std::vector<float> star_positions_x(_collection.size());
  std::vector<float> star_positions_y(_collection.size());
  std::vector<float> masses(_collection.size());
  std::vector<float2> grav_forces(_collection.size());

  // load the vectors
  for (size_t i = 0; i < _collection.size(); ++i)
  {
    // convert C2DVectors to lighter float2s
    C2DVector cur_pos = _collection[i]->GetPosition();

    star_positions_x[i] = cur_pos.x;
    star_positions_y[i] = cur_pos.y;
    masses[i] = _collection[i]->GetMass();
  }

  // call cuda kernel
  compute_grav(star_positions_x.data(),
               star_positions_y.data(),
               masses.data(),
               grav_forces.data(),
               _collection.size());

  for (size_t i = 0; i < _collection.size(); ++i)
  {
    float mass_i = _collection[i]->GetMass();
    _collection[i]->Update(
        external_force_ + C2DVector(mass_i * grav_forces[i].x, mass_i * grav_forces[i].y), dt);
  }
}

/**
 * Update computes the total gravity acting on each single star and calls update
 * on each. Scales as O(N^2)
 */
void CEntityGalaxy::UpdateCPU(const C2DVector& external_force_, float dt)
{
  std::vector<C2DVector> pairwise_forces(
      _collection.size() * _collection.size()); // We actully only need half of the matrix
                                                // (force_ij=-force_ji), with no trace (no
                                                // self interaction), however this would
                                                // lead to complicated look up

  // load the forces for each pair of star in the forces vector (N^2
  // operation)
  for (unsigned int i = 0; i < _collection.size(); ++i)
  {
    for (unsigned int j = 0; j < i;
         ++j) // keep j < i to compute only the upper half of the matrix. The matrix is
              // antisimmetric anyway, no need to compute it all!
    {
      float mass_i = _collection[i]->GetMass();
      C2DVector pos_i = _collection[i]->GetPosition();

      float mass_j = _collection[j]->GetMass();
      C2DVector pos_j = _collection[j]->GetPosition();

      // compute gravity
      C2DVector r = pos_j - pos_i; // vector from i to j

      float dist = r.GetLenght();
      const float min_dist = 70.0f;    // to avoid infinities
      const float NEWTON_CONST = 2.0f; // higher than in CUDA case since we surely have less stars,
                                       // so let's make it more interesting!

      // force = G*m*M/ r^2
      C2DVector force_ij = NEWTON_CONST * mass_i * mass_j / (dist * dist * dist + min_dist) *
                           r; // r is not normalized, therefore we divide by dist^3

      unsigned int index_ij = i + _collection.size() * j; // col + rows_num*row
      unsigned int index_ji = j + _collection.size() * i; // col + rows_num*row

      pairwise_forces[index_ij] = force_ij;
      pairwise_forces[index_ji] = (-1) * force_ij; // save redundant
                                                   // information for easy
                                                   // and fast look-ups
    }
  }

  // now add forces for each particle and apply it ( order N^2 )
  for (unsigned int i = 0; i < _collection.size(); ++i)
  {
    C2DVector force_on_i = C2DVector(0.0f, 0.0f);
    for (unsigned int j = 0; j < _collection.size(); ++j) // sum all the column of forces
    {
      if (i != j)
      {
        unsigned int index_ij = i + _collection.size() * j; // col + rows_num*row
        force_on_i += pairwise_forces[index_ij];
      }
    }
    _collection[i]->Update(external_force_ + force_on_i, dt);
  }
}

void CEntityGalaxy::Draw() const
{
  for (const auto& star : _collection)
  {
    star->Draw();
  }
}

void CEntityGalaxy::HandleMouseButtonDown(std::shared_ptr<C2DVector> coords_)
{
  // onclick add another massive star like the last one, in the place we
  // clicked
  if (_collection.size() <= (_original_star_num)) // _original_star_num is the current total
                                                  // of stars counting the black hole
  {
    // use the first star as a prototype
    std::unique_ptr<CEntityParticle> new_star =
        std::make_unique<CEntityParticle>(*(_collection[0]));

    new_star->Reposition(*coords_);

    // make it super massive!
    const float mass = 10000.0f;

    new_star->SetMass(mass);
    new_star->SetScale(10.0f);
    // set id
    new_star->SetId(_collection.size());
    new_star->HandleMouseButtonDown(coords_);
    _collection.push_back(std::move(new_star));
  }
}
void CEntityGalaxy::HandleMouseButtonUp(std::shared_ptr<C2DVector> /*coords_*/)
{
  // remove the massive star just added
  if (_collection.size() > (_original_star_num)) // _original_star_num is the current total
                                                 // of stars counting the black hole
    _collection.pop_back();
}
