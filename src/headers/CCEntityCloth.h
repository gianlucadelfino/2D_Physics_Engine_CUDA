#ifndef CENTITYCLOTH_H
#define CENTITYCLOTH_H

#include "CEntity.h"
#include <vector>
using namespace std;

class CEntityCloth: public CEntity 
{
public:
	CEntityCloth(CDrawable* _d, IPhysics* _p, IMoveable* _m, int _x_seams, int _y_seams):
		CEntity( _d, _p, _m),
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
	vector<CEntity> m_seams;
};

#endif