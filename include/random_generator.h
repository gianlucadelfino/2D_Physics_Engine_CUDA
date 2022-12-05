#ifndef CRANDOM_H
#define CRANDOM_H

#include <random>

/**
 * random_generator is a utility class to generate random ints and floats.
 */
class random_generator
{
public:
  random_generator() : _real_distribution(0.0, 1.0f), _int_distribution(0, 1)
  {
    _generator.seed(1337);
  }

  void set_real_bounds(float from_, float to_)
  {
    _real_distribution = std::uniform_real_distribution<float>(from_, to_);
  }

  void set_int_bounds(int from_, int to_)
  {
    _int_distribution = std::uniform_int_distribution<int32_t>(from_, to_);
  }

  float rand_real() { return _real_distribution(_generator); }

  int rand_int() { return _int_distribution(_generator); }

private:
  // disable copy and assignment
  random_generator(const random_generator&);
  random_generator& operator=(const random_generator&);

  std::mt19937 _generator; // Mersenne Twister generator
  std::uniform_real_distribution<float> _real_distribution;
  std::uniform_int_distribution<int32_t> _int_distribution;
};

#endif
