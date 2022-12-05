#include "vec2.h"
#include <cmath>

vec2::vec2(const vec2& vec)
{
  x = vec.x;
  y = vec.y;
}

vec2& vec2::operator=(const vec2& rhs)
{
  if (this != &rhs)
  {
    x = rhs.x;
    y = rhs.y;
  }
  return *this;
}

vec2& vec2::operator*=(const float k)
{
  x *= k;
  y *= k;

  return *this;
}

vec2& vec2::operator+=(const vec2& rhs)
{
  x += rhs.x;
  y += rhs.y;

  return *this;
}

vec2& vec2::operator-=(const vec2& rhs)
{
  x -= rhs.x;
  y -= rhs.y;

  return *this;
}

float vec2::get_squared_length() const { return x * x + y * y; }

float vec2::get_length() const { return sqrt(get_squared_length()); }

float vec2::scalar_product(const vec2& other_) const { return x * other_.x + y * other_.y; }

/******************
 *friend functions*
 *****************/

vec2 operator*(const float k, const vec2& vec) { return vec2(k * vec.x, k * vec.y); }

vec2 operator*(const vec2& vec, const float k) { return vec2(k * vec.x, k * vec.y); }

vec2 operator+(const vec2& first, const vec2& second)
{
  return vec2(first.x + second.x, first.y + second.y);
}

vec2 operator-(const vec2& first, const vec2& second)
{
  return vec2(first.x - second.x, first.y - second.y);
}

std::ostream& operator<<(std::ostream& out, const vec2& vec)
{
  out << "x: " << vec.x << ", y: " << vec.y;
  return out;
}
