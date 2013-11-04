#include <string>
#include "CSceneMainMenu.h"
#include "CSceneGalaxy.h"
#include "CSceneCloth.h"

CSceneMainMenu::CSceneMainMenu( SDL_Surface* screen_, CWorld& world_ ):
	IScene( screen_, world_, SDL_MapRGB( screen_->format, 0, 0, 0 ) )
{}

void CSceneMainMenu::Init()
{
	//add UI elements to Scene
	Uint32 white_color = SDL_MapRGB( this->mp_screen->format, 255, 255, 255 );
	Uint32 black_color = SDL_MapRGB( this->mp_screen->format, 0, 0, 0 );
	SDL_Color button_label_color = { 0, 0, 0, 0 };

	//TITLE label
	std::shared_ptr< CFont > title_font( new CFont( "pcseniorSmall.ttf", 30 ) );
	SDL_Color title_color = { 255, 255, 255, 0 };
	std::unique_ptr< CMoveableButton > title_moveable( new CMoveableButton( C2DVector( 400.0f, 50.0f ), C2DVector( 300.0f, 31.0f ) ) );
	std::unique_ptr< CDrawableButton > title_drawable( new CDrawableButton( title_font, this->mp_screen, "SIMULINO 2000", C2DVector( 200.0f, 31.0f ), black_color, title_color ) );

	std::unique_ptr< CEntity > title( new CEntity(
		1, std::move( title_moveable ),
		std::move( title_drawable )
		));
	this->m_UI_elements.push_back( std::move( title ) );

	//switch to CLOTH simulation button
	std::shared_ptr< CFont > button_font( new CFont( "pcseniorSmall.ttf", 20 ) );

	std::unique_ptr< CMoveableButton > cloth_moveable_button( new CMoveableButton( C2DVector( 500.0f, 300.0f ), C2DVector( 200.0f, 22.0f ) ) );
	std::unique_ptr< CDrawableButton > cloth_button_drawable( new CDrawableButton( button_font, this->mp_screen, "CLOTH", C2DVector( 200.0f, 22.0f ), white_color, button_label_color ) );

	std::unique_ptr< CEntity > cloth_sim_switch_button( new CEntityButton(
		1, std::move( cloth_moveable_button ),
		std::move( cloth_button_drawable ),
		this->mr_world,
		std::move( std::unique_ptr< IScene >( new CSceneCloth( this->mp_screen, this->mr_world) ) )
		));
	this->m_UI_elements.push_back( std::move( cloth_sim_switch_button ) );

	//switch to galaxy simulation button
	std::unique_ptr< CMoveableButton > moveable_button( new CMoveableButton( C2DVector( 500.0f, 400.0f ), C2DVector(  200.0f, 22.0f ) ) );
	std::unique_ptr< CDrawableButton > galaxy_button_drawable( new CDrawableButton( button_font, this->mp_screen, "GALAXY", C2DVector(  200.0f, 22.0f ), white_color, button_label_color ) );

	std::unique_ptr< CEntity > galaxy_sim_switch_button( new CEntityButton(
		1, std::move( moveable_button ),
		std::move( galaxy_button_drawable ),
		this->mr_world,
		std::move( std::unique_ptr< IScene >( new CSceneGalaxy( this->mp_screen, this->mr_world, true, 8*1024 ) ) )//start in CUDA mode!
		));
	this->m_UI_elements.push_back( std::move( galaxy_sim_switch_button ) );

	//credits label
	std::unique_ptr< CMoveableButton > credits_moveable( new CMoveableButton( C2DVector( 800.0f, 660.0f ), C2DVector( 300.0f, 22.0f ) ) );
	std::unique_ptr< CDrawableButton > credits_drawable( new CDrawableButton( button_font, this->mp_screen, "by Gianluca Delfino", C2DVector( 300.0f, 22.0f ), black_color, title_color ) );

	std::unique_ptr< CEntity > credits( new CEntity(
		1, std::move( credits_moveable ),
		std::move( credits_drawable )
		));
	this->m_UI_elements.push_back( std::move( credits ) );
}