#include "CSceneGalaxy.h"

CSceneGalaxy::CSceneGalaxy( SDL_Surface* screen_, CWorld& world_, bool use_CUDA_ ):
	IScene( screen_, world_ ),
	m_use_CUDA( use_CUDA_ )
{
	//build prototype star for the galaxy
	std::unique_ptr< CDrawableStar > star_drawable( new CDrawableStar( screen_ ) );
	std::unique_ptr< CPhysics > star_physics( new CPhysics( 1.0f, C2DVector( 0.0f, 0.0f ) ) );
	star_physics->SetGravity( C2DVector( 0.0f, 0.0f ) );
	std::unique_ptr< CMoveableParticle > star_moveable( new CMoveableParticle() );
	CEntityParticle star( 1, std::move( star_moveable ), std::move( star_drawable ), std::move( star_physics ) );

	//create galaxy
	unsigned int stars_num = this->m_use_CUDA? 8*1024 : 1000; //if I am using CUDA I can do 8 times the stars!
	std::unique_ptr< IEntity > galaxy( new CEntityGalaxy( 2, C2DVector( 600.0f, 300.0f), star, stars_num, 500, this->m_use_CUDA ) ); // id, pos, star prototype, stars number (max 1024*9), size

	//add simulation Elements:
	this->m_entities.push_back( std::move( galaxy ) );
}

void CSceneGalaxy::Init()
{
	//add UI elements to Scene
	std::shared_ptr< CFont > button_font( new CFont( "pcseniorSmall.ttf", 20 ) );
	std::string CUDA_CPU_switch_label = this->m_use_CUDA? "SWITCH TO CPU" : "SWITCH TO CUDA";
	std::unique_ptr< CDrawableButton > cloth_button_drawable( new CDrawableButton( button_font, this->mp_screen, CUDA_CPU_switch_label, C2DVector( 320.0f, 22.0f ) ) );
	std::unique_ptr< CMoveableButton > moveable_button( new CMoveableButton( C2DVector( 50.0f, 600.0f ), C2DVector( 320.0f, 22.0f ) ) );

	std::unique_ptr< IEntity > cloth_sim_switch_button( new CEntityButton(
		1, std::move( moveable_button ),
		std::move( cloth_button_drawable ),
		this->mr_world,
		std::move( std::unique_ptr< IScene >( new CSceneGalaxy( this->mp_screen, this->mr_world, !this->m_use_CUDA ) ) )
		));
	this->m_UI_elements.push_back( std::move( cloth_sim_switch_button ) );
}