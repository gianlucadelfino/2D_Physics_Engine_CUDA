#include "CSceneCloth.h"
#include "CSceneGalaxy.h"
#include "CDrawableLink.h"
#include "CEntityCloth.h"

CSceneCloth::CSceneCloth( SDL_Surface* screen_, CWorld& world_ ):
	IScene( screen_, world_, SDL_MapRGB( screen_->format, 255, 255, 255 ) )
{
	//build cloth scene
	std::unique_ptr<CPhysics> phys_seam( new CPhysics( 1.0f ) );
	std::unique_ptr<IMoveable> seam_moveable( new CMoveableParticle(50.0f, 50.0f) );
	std::unique_ptr<CDrawableLink> cloth_drawable( new CDrawableLink( screen_ ) );

	CEntityParticle seam( 1, std::move(seam_moveable), NULL, std::move(phys_seam) );
	std::unique_ptr< IEntity > cloth( new CEntityCloth( 1, C2DVector( 300.0f,20.0f), std::move( cloth_drawable ), seam, 40 ) ); // id, pos, IDrawable, seam prototype, size
	//add simulation Elements:
	this->m_entities.push_back( std::move( cloth ) );
}

void CSceneCloth::Init()
{
	//add UI elements to Scene
	std::shared_ptr< CFont > button_font( new CFont( "pcseniorSmall.ttf", 20 ) );
	Uint32 black_color = SDL_MapRGB( this->mp_screen->format, 0, 0, 0 );
	SDL_Color button_label_color = { 255, 255, 255, 0 };

	std::unique_ptr< CDrawableButton > cloth_button_drawable( new CDrawableButton( button_font, this->mp_screen, "GALAXY", C2DVector( 220.0f, 22.0f ), black_color, button_label_color ) );
	std::unique_ptr< CMoveableButton > moveable_button( new CMoveableButton( C2DVector( 50.0f, 600.0f ), C2DVector( 220.0f, 22.0f ) ) );

	std::unique_ptr< IEntity > cloth_sim_switch_button( new CEntityButton(
		1, std::move( moveable_button ),
		std::move( cloth_button_drawable ),
		this->mr_world,
		std::move( std::unique_ptr< IScene >( new CSceneGalaxy( this->mp_screen, this->mr_world, true ) ) )
		));
	this->m_UI_elements.push_back( std::move( cloth_sim_switch_button ) );
}