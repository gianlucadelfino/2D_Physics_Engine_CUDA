#include <string>

#include "SDL.h"

#include "CSceneLoadingScreen.h"
#include "CDrawableButton.h"
#include "CMoveableButton.h"
#include "CEntityButton.h"
#include "CFont.h"

CSceneLoadingScreen::CSceneLoadingScreen( SDL_Surface* screen_, CWorld& world_ ):
    IScene( screen_, world_, SDL_MapRGB( screen_->format, 0, 0, 0 ) )
{}

CSceneLoadingScreen::CSceneLoadingScreen( const CSceneLoadingScreen& other_ ):
    IScene( other_ )
{}

CSceneLoadingScreen& CSceneLoadingScreen::operator=( const CSceneLoadingScreen& rhs_ )
{
    if ( this != &rhs_ ) {
        IScene::operator=( rhs_ );
    }
    return *this;
}

void CSceneLoadingScreen::Init()
{
    //LOADING label
    Uint32 black_color = SDL_MapRGB( this->mp_screen->format, 0, 0, 0 );
    std::shared_ptr< CFont > title_font( new CFont( "pcseniorSmall.ttf", 30 ) );
    SDL_Color title_color = { 255, 255, 255, 0 };
    std::unique_ptr< CMoveableButton > title_moveable( new CMoveableButton( C2DVector( 700.0f, 600.0f ), C2DVector( 300.0f, 31.0f ) ) );
    std::unique_ptr< CDrawableButton > title_drawable( new CDrawableButton( title_font, this->mp_screen, "LOADING...", C2DVector( 200.0f, 31.0f ), black_color, title_color ) );

    std::unique_ptr< CEntity > title( new CEntity(
        1, std::move( title_moveable ),
        std::move( title_drawable )
        ));
    this->m_UI_elements.push_back( std::move( title ) );
}
