file(GLOB LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

add_library(engine_utils STATIC ${LIB_SOURCES})

set_target_properties(engine_utils PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS NO
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 52)
target_link_libraries(engine_utils SDL2::SDL2 SDL2_image::SDL2_image SDL2_ttf::SDL2_ttf)
