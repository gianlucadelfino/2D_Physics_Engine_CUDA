#ifndef vec2_H
#define vec2_H

#include <iostream>

/**
 * vec2 manages two-dimensional vectors of floats and defines all the common
 *   Vector operations.
 */
class vec2
{
public:
  vec2() : x(0.0f), y(0.0f) {}
  vec2(float x_, float y_) : x(x_), y(y_) {}
  vec2(int x_, int y_) : x(static_cast<float>(x_)), y(static_cast<float>(y_)) {}
  vec2(const vec2& vec);

  vec2& operator=(const vec2& rhs);
  vec2& operator*=(const float k);
  vec2& operator+=(const vec2& rhs);
  vec2& operator-=(const vec2& rhs);

  float get_squared_length() const;
  float get_length() const;
  float scalar_product(const vec2&) const;

  friend vec2 operator*(const float k, const vec2& vec);
  friend vec2 operator*(const vec2& vec, const float k);

  friend vec2 operator+(const vec2& first, const vec2& second);
  friend vec2 operator-(const vec2& first, const vec2& second);

  friend std::ostream& operator<<(std::ostream& out, const vec2& vec);

  float x;
  float y;
};
#endif
