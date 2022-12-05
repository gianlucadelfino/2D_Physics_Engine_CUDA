#ifndef CRANDOM_H
#define CRANDOM_H

#include <random>

/**
* CRandom is a utility class to generate random ints and floats.
*/
class CRandom
{
public:
    CRandom() : _real_distribution(0.0, 1.0f), _int_distribution(0, 1)
    {
        _generator.seed(1337);
    }

    void SetRealBounds(float from_, float to_)
    {
        _real_distribution = std::uniform_real_distribution<float>(from_, to_);
    }

    void SetIntBounds(int from_, int to_)
    {
        _int_distribution = std::uniform_int_distribution<int32_t>(from_, to_);
    }

    float RandReal()
    {
        return _real_distribution(_generator);
    }

    int RandInt()
    {
        return _int_distribution(_generator);
    }

private:
    // disable copy and assignment
    CRandom(const CRandom&);
    CRandom& operator=(const CRandom&);

    std::mt19937 _generator; // Mersenne Twister generator
    std::uniform_real_distribution<float> _real_distribution;
    std::uniform_int_distribution<int32_t> _int_distribution;
};

#endif
