#include "CSceneCloth.h"
#include "CSceneMainMenu.h"
#include "CEntityCloth.h"

CSceneCloth::CSceneCloth( SDL_Surface* screen_, CWorld& world_ ):
	IScene( screen_, world_, SDL_MapRGB( screen_->format, 255, 255, 255 ) ),
	m_font(  new CFont( "pcseniorSmall.ttf", 20 ) )
{
	//build cloth scene
	std::unique_ptr<CPhysics> phys_seam( new CPhysics( 1.0f ) );
	std::unique_ptr<IMoveable> seam_moveable( new CMoveableParticle(50.0f, 50.0f) );
	std::unique_ptr<CDrawableLink> cloth_drawable( new CDrawableLink( screen_ ) );

	CEntityParticle seam( 1, std::move(seam_moveable), NULL, std::move(phys_seam) );
	std::unique_ptr< CEntity > cloth( new CEntityCloth( 1, C2DVector( 400.0f, 10.0f), std::move( cloth_drawable ), seam, 40 ) ); // id, pos, IDrawable, seam prototype, size
	//add simulation Elements:
	this->m_entities.push_back( std::move( cloth ) );
}

void CSceneCloth::Init()
{
	//add UI elements to Scene
	Uint32 black_color = SDL_MapRGB( this->mp_screen->format, 0, 0, 0 );
	SDL_Color button_label_color = { 255, 255, 255, 0 };

	std::unique_ptr< CMoveableButton > main_menu_moveable_button( new CMoveableButton( C2DVector( 50.0f, 600.0f ), C2DVector( 200.0f, 22.0f ) ) );
	std::unique_ptr< CDrawableButton > main_menu_button_drawable( new CDrawableButton( this->m_font, this->mp_screen, "MAIN MENU", C2DVector( 200.0f, 22.0f ), black_color, button_label_color ) );

	std::unique_ptr< CEntity > galaxy_sim_switch_button( new CEntityButton(
		1, std::move( main_menu_moveable_button ),
		std::move( main_menu_button_drawable ),
		this->mr_world,
		std::move( std::unique_ptr< IScene >( new CSceneMainMenu( this->mp_screen, this->mr_world ) ) )
		));
	this->m_UI_elements.push_back( std::move( galaxy_sim_switch_button ) );

	//instructions label (first half)
	Uint32 white_color = SDL_MapRGB( this->mp_screen->format, 255, 255, 255 );
	SDL_Color label_color = { 0, 0, 0, 0 };

	std::unique_ptr< CMoveableButton > instructions_first_label_moveable( new CMoveableButton( C2DVector( 900.0f, 300.0f ), C2DVector( 220.0f, 22.0f ) ) );
	std::unique_ptr< CDrawableButton > instructions_first_label_drawable( new CDrawableButton( this->m_font, this->mp_screen, "click and hold to", C2DVector( 220.0f, 22.0f ), white_color, label_color ) );

	std::unique_ptr< CEntity > instructions_first_label( new CEntity(
		1, std::move( instructions_first_label_moveable ),
		std::move( instructions_first_label_drawable )
		));
	this->m_UI_elements.push_back( std::move( instructions_first_label ) );
	//instructions label (second half)
	std::unique_ptr< CMoveableButton > instructions_second_label_moveable( new CMoveableButton( C2DVector( 900.0f, 330.0f ), C2DVector( 220.0f, 22.0f ) ) );
	std::unique_ptr< CDrawableButton > instructions_second_label_drawable( new CDrawableButton( this->m_font, this->mp_screen, "drag the cloth", C2DVector( 220.0f, 22.0f ), white_color, label_color ) );

	std::unique_ptr< CEntity > instructions_second_label( new CEntity(
		1, std::move( instructions_second_label_moveable ),
		std::move( instructions_second_label_drawable )
		));
	this->m_UI_elements.push_back( std::move( instructions_second_label ) );
}