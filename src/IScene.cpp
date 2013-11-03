#include "IScene.h"
#include "CWorld.h"

IScene::IScene( SDL_Surface* screen_, CWorld& world_ ):mp_screen( screen_ ), mr_world( world_ )
{}

void IScene::Init()
{}

void IScene::Update( float dt )
{
	//update the "Entities"
	for( unique_ptr< IEntity >& it : this->m_entities )
	{
		it->Update( C2DVector( 0.0f, 0.0f ), dt );
	}
}

void IScene::HandleEvent( CWorld& world_, const SDL_Event& event_ )
{
	//mouse coords
	int x, y;
	bool UI_element_hit = false;

	switch( event_.type )
	{
	case SDL_KEYDOWN:
		break;
	case SDL_MOUSEBUTTONDOWN:
		SDL_GetMouseState(&x, &y);
		//see if I hit a UI element first, if so, dont look at the entities!
		for( unique_ptr< IEntity >& it : this->m_UI_elements )
		{
			if ( it->IsHit( C2DVector( x, y ) ) )
			{
				it->HandleMouseButtonDown( std::shared_ptr<C2DVector>( new C2DVector( x, y ) ) );
				UI_element_hit = true;
				break;
			}
		}
		//now propagate the click if to the entities if I didnt hit a UI element
		if ( !UI_element_hit )
		{
			for( unique_ptr< IEntity >& it : this->m_entities )
			{
				it->HandleMouseButtonDown( std::shared_ptr<C2DVector>( new C2DVector( x, y ) ) );
			}
		}
		break;
	case SDL_MOUSEBUTTONUP:
		SDL_GetMouseState(&x, &y);
		//see if I hit a UI element first, if so, dont look at the entities!
		for( unique_ptr< IEntity >& it : this->m_UI_elements )
		{
			if ( it->IsHit( C2DVector( x, y ) ) )
			{
				it->HandleMouseButtonUp( std::shared_ptr<C2DVector>( new C2DVector( x, y ) ) );
				UI_element_hit = true;
				break;
			}
		}
		//now propagate the click if to the entities if I didnt hit a UI element
		if ( !UI_element_hit )
		{
			for( unique_ptr< IEntity >& it : this->m_entities )
			{
				it->HandleMouseButtonUp( std::shared_ptr<C2DVector>( new C2DVector( x, y ) ) );
			}
		}
		break;
	}
}

void IScene::Draw() const
{
	//Draw Background
	SDL_Rect screen_rect;
	screen_rect.x = 0;
	screen_rect.y = 0;
	screen_rect.w = static_cast<Uint16>(this->mp_screen->w);
	screen_rect.h = static_cast<Uint16>(this->mp_screen->h);
	Uint32 color_black = SDL_MapRGB( this->mp_screen->format, 0	, 0, 0);
	SDL_FillRect( this->mp_screen, &screen_rect, color_black );

	//Draw the "Entities"
	for( vector< unique_ptr< IEntity > >::const_iterator cit = this->m_entities.begin(); cit != this->m_entities.end(); ++cit )
	{
		(*cit)->Draw();
	}
	//Draw the HUD/UI
	for( const unique_ptr< IEntity >& cit : this->m_UI_elements )
	{
		cit->Draw();
	}
}

void IScene::AddUIelement( std::unique_ptr< IEntity > UI_element_ )
{
	this->m_UI_elements.push_back( std::move( UI_element_ ) );
}

IScene::~IScene(){}