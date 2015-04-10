#ifndef CRANDOM_H
#define CRANDOM_H

#include <random>

/**
* CRandom is a utility class to generate random ints and floats.
*/
class CRandom
{
public:
    CRandom():
        m_real_distribution( 0.0, 1.0f ),
        m_int_distribution( 0, 1)
    {
        m_generator.seed(1337);
    }

    void SetRealBounds( float from_, float to_ )
    {
        m_real_distribution = std::uniform_real_distribution<float>( from_, to_ );
    }

    void SetIntBounds( int from_, int to_ )
    {
        m_int_distribution = std::uniform_int_distribution<int32_t>( from_, to_ );
    }

    float RandReal(){ return m_real_distribution( m_generator ); }

    int RandInt(){ return m_int_distribution( m_generator ); }

private:
    //disable copy and assignment
    CRandom( const CRandom& );
    CRandom& operator=( const CRandom&);

    std::mt19937 m_generator; // Mersenne Twister generator
    std::uniform_real_distribution<float> m_real_distribution;
    std::uniform_int_distribution<int32_t> m_int_distribution;
};

#endif
