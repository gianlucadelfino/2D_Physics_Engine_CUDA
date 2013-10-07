#include "C2DVector.h"
#include <cmath>

C2DVector::C2DVector( const C2DVector& _vec )
{
	x = _vec.x;
	y = _vec.y;
}

C2DVector& C2DVector::operator=( const C2DVector& rhs)
{
	if( this != &rhs)
	{
		this->x = rhs.x;
		this->y = rhs.y;
	}
	return *this;
}

C2DVector& C2DVector::operator*=( const float k )
{
	this->x *= k;
	this->y *= k;

	return *this;
}

C2DVector& C2DVector::operator+=( const C2DVector& rhs )
{
	this->x += rhs.x;
	this->y += rhs.y;

	return *this;
}

C2DVector& C2DVector::operator-=( const C2DVector& rhs )
{
	this->x -= rhs.x;
	this->y -= rhs.y;

	return *this;
}

float C2DVector::GetSquaredLength() const
{
	return this->x * this->x + this->y * this->y;
}

float C2DVector::GetLenght() const
{
	return sqrt( this->GetSquaredLength() );
}

float C2DVector::GetApproxInverseSqrt() const
{
	return Q_rsqrt( this->GetSquaredLength() );
}

float C2DVector::Q_rsqrt( float number ) const
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
	//      y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

float C2DVector::ScalarProduct( const C2DVector& _other) const
{
	return this->x * _other.x + this->y * _other.y;
}

/******************
*friend functions*
*****************/

C2DVector operator*( const float k, const C2DVector& _vec )
{
	return C2DVector( k * _vec.x, k * _vec.y );
}

C2DVector operator*( const C2DVector& _vec, const float k )
{
	return C2DVector( k * _vec.x, k * _vec.y );
}

C2DVector operator+( const C2DVector& _first, const C2DVector& _second )
{
	return C2DVector( _first.x + _second.x, _first.y + _second.y );
}

C2DVector operator-( const C2DVector& _first, const C2DVector& _second )
{
	return C2DVector( _first.x - _second.x, _first.y - _second.y );
}

std::ostream& operator<< ( std::ostream& out, const C2DVector& vec) {
	out << "x: " << vec.x << ", y: " << vec.y;
	return out;
}