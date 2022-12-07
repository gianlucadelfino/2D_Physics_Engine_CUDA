#include "SDL.h"
#include "scene_main_menu.h"
#include "world_manager.h"
#include <SDL_video.h>
#include <iostream>
#include <memory>

// SDL Requires the arguments argc and **argv to be esplicitly referenced
int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
  const int SCREEN_WIDTH = 1280;
  const int SCREEN_HEIGHT = 720;

  const int MS_PER_UPDATE = 32; // 30Hz

  // init SDL
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) // returns 0 if everything is ok!
  {
    std::cerr << "Could NOT initialize SDL. Error: " << SDL_GetError() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
      SDL_CreateWindow("2D Physics Engine. Gianluca.Delfino@gmail.com",
                       SDL_WINDOWPOS_CENTERED,
                       SDL_WINDOWPOS_CENTERED,
                       SCREEN_WIDTH,
                       SCREEN_HEIGHT,
                       0),
      SDL_DestroyWindow);

  if (!window)
  {
    std::cerr << "Failed to create window\n";
    std::cerr << "SDL2 Error: " << SDL_GetError() << "\n";
    exit(EXIT_FAILURE);
  }

  std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)> renderer(
      SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC),
      SDL_DestroyRenderer);

  if (!renderer)
  {
    std::cerr << "Failed to get window's surface\n";
    std::cerr << "SDL2 Error: " << SDL_GetError() << "\n";
    exit(EXIT_FAILURE);
  }
  // instantiate World
  world_manager world;

  bool quit = false;

  // keep track of the lag if the game is slowing down (update only, without
  // rendering, to keep up)
  int lag = 0;
  int prev_ticks = SDL_GetTicks();

  // Main Loop
  while (!quit)
  {
    // update dt
    const int cur_ticks = SDL_GetTicks();
    const int dt = cur_ticks - prev_ticks;
    prev_ticks = cur_ticks;

    // add the time it took last frame to the lag, then we know how many
    // times we have to cycle over MS_PER_UPDATE to catch up with real time
    lag += dt;

    while (lag >= MS_PER_UPDATE)
    {
      // event handling
      SDL_Event event{};
      while (SDL_PollEvent(&event))
      {
        world.handle_event(event);
        if (event.type == SDL_QUIT)
          quit = true;
      }
      // update World (constant in front of time is arbitrary to set
      // "distances")
      world.update(static_cast<float>(MS_PER_UPDATE) * 0.01f);

      lag -= MS_PER_UPDATE;
    }

    // draw world
    world.draw(renderer.get());

    // update Screen
    SDL_RenderPresent(renderer.get());

    // sleep if it took less than MS_PER_UPDATE
    const int delay = cur_ticks + MS_PER_UPDATE - SDL_GetTicks();
    if (delay > 0)
    {
      SDL_Delay(delay);
    }
  }

  SDL_Quit();

  return 0;
}
