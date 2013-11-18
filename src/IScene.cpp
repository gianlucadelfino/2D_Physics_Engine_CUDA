#include "IScene.h"
#include "CWorld.h"

IScene::IScene( SDL_Surface* screen_, CWorld& world_, Uint32 background_color_ ):
	mp_screen( screen_ ),
	mr_world( world_ ),
	m_background_color( background_color_ ),
	m_mouse_coords( new C2DVector() )
{}

void IScene::Init()
{}

void IScene::Update( float dt )
{
	//update the "Entities"
	for( unique_ptr< CEntity >& it : this->m_entities )
	{
		it->Update( C2DVector( 0.0f, 0.0f ), dt );
	}
}

void IScene::HandleEvent( CWorld& /*world_*/, const SDL_Event& event_ )
{
	bool UI_element_hit = false;

	switch( event_.type )
	{
	case SDL_KEYDOWN:
		break;
	case SDL_MOUSEMOTION:
		this->m_mouse_coords->x = static_cast<float>(event_.motion.x);
		this->m_mouse_coords->y = static_cast<float>(event_.motion.y);
		break;
	case SDL_MOUSEBUTTONDOWN:
		//see if I hit a UI element first, if so, dont look at the entities!
		for( unique_ptr< CEntity >& it : this->m_UI_elements )
		{
			if ( it->IsHit( *this->m_mouse_coords ) )
			{
				it->HandleMouseButtonDown( this->m_mouse_coords );
				UI_element_hit = true;
				break;
			}
		}
		//now propagate the click if to the entities if I didnt hit a UI element
		if ( !UI_element_hit )
		{
			for( unique_ptr< CEntity >& it : this->m_entities )
			{
				it->HandleMouseButtonDown( this->m_mouse_coords );
			}
		}
		break;
	case SDL_MOUSEBUTTONUP:
		//see if I hit a UI element first, if so, dont look at the entities!
		for( unique_ptr< CEntity >& it : this->m_UI_elements )
		{
			if ( it->IsHit( *this->m_mouse_coords ) )
			{
				it->HandleMouseButtonUp( this->m_mouse_coords );
				UI_element_hit = true;
				break;
			}
		}
		//now propagate the click if to the entities if I didnt hit a UI element
		if ( !UI_element_hit )
		{
			for( unique_ptr< CEntity >& it : this->m_entities )
			{
				it->HandleMouseButtonUp( this->m_mouse_coords );
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
	SDL_FillRect( this->mp_screen, &screen_rect, this->m_background_color );

	//Draw the "Entities"
	for( const unique_ptr< CEntity >& cit : this->m_entities )
	{
		cit->Draw();
	}
	//Draw the HUD/UI last (on top)
	for( const unique_ptr< CEntity >& cit : this->m_UI_elements )
	{
		cit->Draw();
	}
}

IScene::~IScene(){}