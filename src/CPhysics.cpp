#include "CPhysics.h"

/**
* Integrate increments the position of the entity by integrating the force acting on it
*/
void CPhysics::Integrate( C2DVector& pos, C2DVector& old_pos, const C2DVector& _external_force, float dt )
{
	//verlet
	C2DVector temp = pos;
	pos += (pos - old_pos) + this->m_inverseMass *( this->GetForce( pos ) + _external_force )* dt * dt;
	old_pos = temp;
}