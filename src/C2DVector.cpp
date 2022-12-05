#include "C2DVector.h"
#include <cmath>

C2DVector::C2DVector(const C2DVector& vec)
{
  x = vec.x;
  y = vec.y;
}

C2DVector& C2DVector::operator=(const C2DVector& rhs)
{
  if (this != &rhs)
  {
    x = rhs.x;
    y = rhs.y;
  }
  return *this;
}

C2DVector& C2DVector::operator*=(const float k)
{
  x *= k;
  y *= k;

  return *this;
}

C2DVector& C2DVector::operator+=(const C2DVector& rhs)
{
  x += rhs.x;
  y += rhs.y;

  return *this;
}

C2DVector& C2DVector::operator-=(const C2DVector& rhs)
{
  x -= rhs.x;
  y -= rhs.y;

  return *this;
}

float C2DVector::GetSquaredLength() const { return x * x + y * y; }

float C2DVector::GetLenght() const { return sqrt(GetSquaredLength()); }

float C2DVector::ScalarProduct(const C2DVector& other_) const
{
  return x * other_.x + y * other_.y;
}

/******************
 *friend functions*
 *****************/

C2DVector operator*(const float k, const C2DVector& vec) { return C2DVector(k * vec.x, k * vec.y); }

C2DVector operator*(const C2DVector& vec, const float k) { return C2DVector(k * vec.x, k * vec.y); }

C2DVector operator+(const C2DVector& first, const C2DVector& second)
{
  return C2DVector(first.x + second.x, first.y + second.y);
}

C2DVector operator-(const C2DVector& first, const C2DVector& second)
{
  return C2DVector(first.x - second.x, first.y - second.y);
}

std::ostream& operator<<(std::ostream& out, const C2DVector& vec)
{
  out << "x: " << vec.x << ", y: " << vec.y;
  return out;
}
