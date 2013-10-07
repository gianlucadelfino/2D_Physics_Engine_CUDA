/*
*  CPhysics.cpp
*  MagicMine
*
*  Created by gianluca on 2/2/13.
*  Copyright 2013 __MyCompanyName__. All rights reserved.
*
*/

#include "CPhysics.h"

void CPhysics::Integrate( C2DVector& pos, C2DVector& old_pos, const C2DVector& _external_force, float dt )
{
	//verlet
	C2DVector temp = pos;
	pos += (pos - old_pos) + this->m_inverseMass *( this->GetForce( pos ) + _external_force )* dt * dt;
	old_pos = temp;
}