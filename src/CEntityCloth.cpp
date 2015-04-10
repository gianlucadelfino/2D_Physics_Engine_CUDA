#include "CEntityCloth.h"

CEntityCloth::CEntityCloth( unsigned int id_, const C2DVector& initial_pos_, std::unique_ptr< CDrawableLink > drawable_, const CEntityParticle& seam_, unsigned int side_length_ ):
    CEntity( id_, std::unique_ptr< IMoveable > ( new CMoveableParticle( initial_pos_ ) ), std::move(drawable_) ),
    m_side_length( side_length_ ),
    m_total_seams( side_length_*side_length_ ),
    m_max_dist( 10.0f )
{
    // build the cloth
    for ( unsigned int i = 0; i < m_total_seams ; ++i )
    {
        // column number
        unsigned int col = i % m_side_length;
        // row number
        unsigned int row = i / m_side_length;
        // compute the position
        C2DVector pos( initial_pos_ + m_max_dist * C2DVector( col * 1.0f, row * 1.0f ) );

        //add to the collection
        this->m_collection.push_back( std::shared_ptr<CEntityParticle>( new CEntityParticle( seam_ ) ) );

        //reposition it and set the id..
        this->m_collection[i]->Reposition( pos );

        this->m_collection[i]->SetId(i);
    }

    //add constraint to top row so they dont move (this assumes linear internal structure like vector)
    for ( unsigned int i = 0; i < m_side_length; ++i )
    {
        this->m_collection[i]->Block();
    }
}

CEntityCloth::CEntityCloth( const CEntityCloth& other_ ):
    CEntity( other_ ), //assign the position of the first particle
    m_side_length( other_.m_side_length ),
    m_total_seams( other_.m_side_length*other_.m_side_length ),
    m_max_dist( other_.m_max_dist )
{
    // build the cloth
    for ( unsigned int i = 0; i < m_total_seams ; ++i )
    {
        // compute the position
        C2DVector pos = other_.m_collection[i]->GetPosition();

        //add to the collection
        this->m_collection.push_back( std::shared_ptr<CEntityParticle>( new CEntityParticle( *other_.m_collection[i] ) ) );//pass a copy of the particle

        //reposition it and set the id..
        this->m_collection[i]->Reposition( pos );

        this->m_collection[i]->SetId(i);
    }

    //add constraint to top row so they dont move (this assumes linear internal structure like vector)
    for ( unsigned int i = 0; i < m_side_length; ++i )
    {
        this->m_collection[i]->Block();
    }
}

CEntityCloth& CEntityCloth::operator=( const CEntityCloth& rhs )
{
    if ( this != &rhs )
    {
        CEntity::operator=( rhs );
        this->m_side_length = rhs.m_side_length;
        this->m_total_seams = rhs.m_side_length*rhs.m_side_length;
    }
    return *this;
}

void CEntityCloth::HandleMouseButtonDown( std::shared_ptr<C2DVector> coords_ )
{
    for ( std::vector< shared_ptr< CEntityParticle > >::iterator it = m_collection.begin(); it != m_collection.end(); ++it )
        (*it)->HandleMouseButtonDown( coords_ );
}

void CEntityCloth::HandleMouseButtonUp( std::shared_ptr<C2DVector> coords_ )
{
    for ( std::vector< shared_ptr< CEntityParticle > >::iterator it = m_collection.begin(); it != m_collection.end(); ++it )
        (*it)->HandleMouseButtonUp( coords_ );
}

void CEntityCloth::Update( const C2DVector& external_force_, float dt )
{
    //call update on all elements of c_collection, passing the collection total force
    //this indirectly applies local constraints via the entity update
    for ( std::vector< std::shared_ptr<CEntityParticle> >::iterator it = m_collection.begin(); it != m_collection.end(); ++it )
    {
        (*it)->Update( external_force_, dt );
    }

    //applying the collective constraints, the more times we apply them, the more precise the simulation will be
    for ( unsigned int iter = 0; iter < NUMBER_OF_ITERATIONS; ++iter)
    {
        for ( unsigned int id = 0; id < m_collection.size(); ++id )
        {
            this->ApplyCollectiveConstraints( id );
        }
    }
}

void CEntityCloth::Draw() const
{
    //draw each link, for each id, we draw the left and top link
    for ( unsigned int id = 0; id < this->m_collection.size(); ++id )
    {
        //find the links to draw
        int top_id = id - m_side_length;
        int left_id = id - 1;

        //find the coords of the 3 points to use in drawing the 2 lines
        C2DVector id_pos = this->m_collection[id]->GetPosition();
        if ( top_id >= 0 )
        {
            C2DVector top_id_position = this->m_collection[top_id]->GetPosition();
            this->mp_drawable->Draw( top_id_position, id_pos );
        }

        if ( (left_id % m_side_length) < (id % m_side_length) )
        {
            C2DVector left_id_position = this->m_collection[left_id]->GetPosition();
            this->mp_drawable->Draw( left_id_position, id_pos );
        }
    }
}

void CEntityCloth::ApplyCollectiveConstraints( const unsigned int id )
{
    if ( id != 0 )//the first one doesnt move
    {
        // look at the distance with the seam above and on the left. The id tells me the position in the matrix
        int top_id = id - m_side_length;
        int left_id = id - 1;

        // ensure particle is not on the top edge
        if ( top_id >= 0 )
            EnforceMaxDist( id, top_id );

        // ensure the particle is not on the left edge.
        if ( (left_id % m_side_length) < (id % m_side_length)  )
            EnforceMaxDist( id, left_id);
    }
}

void CEntityCloth::EnforceMaxDist( int id_a, int id_b )
{
    C2DVector a_pos = m_collection[id_a]->GetPosition();
    C2DVector b_pos = m_collection[id_b]->GetPosition();

    //compute the difference vector and the distance
    C2DVector abvect = a_pos - b_pos;
    float ab_length = sqrt( abvect.GetSquaredLength() );
    float delta = ( ab_length - m_max_dist ) / ab_length;

    if ( delta > 0.01f ) //impose only if be greater than a small treshold
    {
        a_pos -= abvect * delta * 0.5f;
        b_pos += abvect * delta * 0.5f;

        //apply the changes
        m_collection[id_a]->Boost( a_pos );
        m_collection[id_b]->Boost( b_pos );
    }
}
