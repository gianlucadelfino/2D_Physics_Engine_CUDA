#include "entity_cloth.h"

entity_cloth::entity_cloth(unsigned int id_,
                           const vec2& initial_pos_,
                           std::unique_ptr<drawable_link> drawable_,
                           const entity_particle& seam_,
                           unsigned int side_length_)
    : entity_base(id_, std::make_unique<moveable_particle>(initial_pos_), std::move(drawable_)),
      _side_length(side_length_),
      _total_seams(side_length_ * side_length_),
      _max_dist(10.0f)
{
  // build the cloth
  for (unsigned int i = 0; i < _total_seams; ++i)
  {
    // column number
    unsigned int col = i % _side_length;
    // row number
    unsigned int row = i / _side_length;
    // compute the position
    vec2 pos(initial_pos_ + _max_dist * vec2(col * 1.0f, row * 1.0f));

    // add to the collection
    _collection.push_back(std::make_unique<entity_particle>(seam_));

    // reposition it and set the id..
    _collection[i]->reposition(pos);

    _collection[i]->SetId(i);
  }

  // add constraint to top row so they dont move (this assumes linear internal
  // structure like vector)
  for (unsigned int i = 0; i < _side_length; ++i)
  {
    _collection[i]->block();
  }
}

entity_cloth::entity_cloth(const entity_cloth& other_)
    : entity_base(other_), // assign the position of the first particle
      _side_length(other_._side_length),
      _total_seams(other_._side_length * other_._side_length),
      _max_dist(other_._max_dist)
{
  // build the cloth
  for (unsigned int i = 0; i < _total_seams; ++i)
  {
    // compute the position
    vec2 pos = other_._collection[i]->get_position();

    // add to the collection
    _collection.push_back(
        std::make_unique<entity_particle>((*other_._collection[i]))); // pass a copy of the particle

    // reposition it and set the id..
    _collection[i]->reposition(pos);

    _collection[i]->SetId(i);
  }

  // add constraint to top row so they dont move (this assumes linear internal
  // structure like vector)
  for (unsigned int i = 0; i < _side_length; ++i)
  {
    _collection[i]->block();
  }
}

entity_cloth& entity_cloth::operator=(const entity_cloth& rhs)
{
  if (this != &rhs)
  {
    entity_base::operator=(rhs);
    _side_length = rhs._side_length;
    _total_seams = rhs._side_length * rhs._side_length;
  }
  return *this;
}

void entity_cloth::handle_mouse_buttondown(std::shared_ptr<vec2> coords_)
{
  for (auto& entity : _collection)
  {
    entity->handle_mouse_buttondown(coords_);
  }
}

void entity_cloth::handle_mouse_buttonup(std::shared_ptr<vec2> coords_)
{
  for (auto& entity : _collection)
  {
    entity->handle_mouse_buttonup(coords_);
  }
}

void entity_cloth::update(const vec2& external_force_, float dt)
{
  // call update on all elements of _collection, passing the collection total
  // force this indirectly applies local constraints via the entity update
  for (auto& entity : _collection)
  {
    entity->update(external_force_, dt);
  }

  // applying the collective constraints, the more times we apply them, the
  // more precise the simulation will be
  for (unsigned int iter = 0; iter < NUMBER_OF_ITERATIONS; ++iter)
  {
    for (unsigned int id = 0; id < _collection.size(); ++id)
    {
      ApplyCollectiveConstraints(id);
    }
  }
}

void entity_cloth::draw() const
{
  // draw each link, for each id, we draw the left and top link
  for (unsigned int id = 0; id < _collection.size(); ++id)
  {
    // find the links to draw
    int top_id = id - _side_length;
    int left_id = id - 1;

    // find the coords of the 3 points to use in drawing the 2 lines
    vec2 id_pos = _collection[id]->get_position();
    if (top_id >= 0)
    {
      vec2 top_id_position = _collection[top_id]->get_position();
      _drawable->draw(top_id_position, id_pos);
    }

    if ((left_id % _side_length) < (id % _side_length))
    {
      vec2 left_id_position = _collection[left_id]->get_position();
      _drawable->draw(left_id_position, id_pos);
    }
  }
}

void entity_cloth::ApplyCollectiveConstraints(const unsigned int id)
{
  if (id != 0) // the first one doesnt move
  {
    // look at the distance with the seam above and on the left. The id
    // tells me the position in the matrix
    int top_id = id - _side_length;
    int left_id = id - 1;

    // ensure particle is not on the top edge
    if (top_id >= 0)
      EnforceMaxDist(id, top_id);

    // ensure the particle is not on the left edge.
    if ((left_id % _side_length) < (id % _side_length))
      EnforceMaxDist(id, left_id);
  }
}

void entity_cloth::EnforceMaxDist(int id_a, int id_b)
{
  vec2 a_pos = _collection[id_a]->get_position();
  vec2 b_pos = _collection[id_b]->get_position();

  // compute the difference vector and the distance
  vec2 abvect = a_pos - b_pos;
  float ab_length = sqrt(abvect.get_squared_length());
  float delta = (ab_length - _max_dist) / ab_length;

  if (delta > 0.01f) // impose only if be greater than a small treshold
  {
    a_pos -= abvect * delta * 0.5f;
    b_pos += abvect * delta * 0.5f;

    // apply the changes
    _collection[id_a]->boost(a_pos);
    _collection[id_b]->boost(b_pos);
  }
}
