#include "CPhysics.h"

/**
* Integrate increments the position of the entity by integrating the force acting on it
*/
void CPhysics::Integrate( C2DVector& pos, C2DVector& old_pos, const C2DVector& external_force_, float dt )
{
	//verlet
	C2DVector temp = pos;
	pos += (pos - old_pos) + this->m_inverseMass *( this->GetForce( pos ) + external_force_ )* dt * dt;
	old_pos = temp;
}