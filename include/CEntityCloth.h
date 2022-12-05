#ifndef CENTITYCLOTH_H
#define CENTITYCLOTH_H

#include <memory>
#include <vector>
#include <cmath>
#include "CEntityParticle.h"
#include "C2DVector.h"
#include "CDrawableLink.h"

/**
* CEntityCloth is a child of CEntityParticle to contain CEntities, following the
* Pattern of the "Composite".
* Calling Update on the collection will update all the Entities in it and impose
* all the constraints.
*/
class CEntityCloth : public CEntity
{
public:
    CEntityCloth(unsigned int id_,
                 const C2DVector& initial_pos_,
                 std::unique_ptr<CDrawableLink> drawable_,
                 const CEntityParticle& seam_,
                 unsigned int side_length_);

    CEntityCloth(const CEntityCloth&);
    CEntityCloth& operator=(const CEntityCloth&);

    virtual void HandleMouseButtonDown(std::shared_ptr<C2DVector> coords_);
    virtual void HandleMouseButtonUp(std::shared_ptr<C2DVector> coords_);

    virtual void Update(const C2DVector& external_force_, float dt);
    virtual void Draw() const;

private:
    std::vector<std::unique_ptr<CEntityParticle>> _collection;
    unsigned int _side_length;
    unsigned int _total_seams;
    C2DVector _cloth_pos;

    // the more iteration, the more accurate the simulation
    static const int NUMBER_OF_ITERATIONS = 2;
    // define distance between cloth seams
    const float _max_dist;

    virtual void ApplyCollectiveConstraints(const unsigned int id);
    void EnforceMaxDist(int id_a, int id_b);
};

#endif
