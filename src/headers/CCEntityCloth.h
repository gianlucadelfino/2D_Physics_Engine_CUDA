#ifndef CENTITYCLOTH_H
#define CENTITYCLOTH_H

#include "IEntity.h"
#include "CPhysics.h"
#include <vector>
using namespace std;

class CEntityCloth: public IEntity
{
public:
	CEntityCloth(IDrawable* _d, CPhysics* _p, IMoveable* _m, int _x_seams, int _y_seams):
		IEntity( _d, _p, _m),
		m_x_seams( _x_seams),
		m_y_seams( _y_seams),
		m_seams( _x_seams*_y_seams)
	{}

	virtual void Update( const double dt)
	{}

	virtual void Draw();

private:
	int m_x_seams;
	int m_y_seams;
	vector<IEntity> m_seams;
};

#endif