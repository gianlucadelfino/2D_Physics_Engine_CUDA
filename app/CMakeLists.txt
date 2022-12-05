add_executable(engine engine.cpp)

include_directories(${CMAKE_CUR_SOURCE_DIR})

set_target_properties(engine PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS NO
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 52)

target_link_libraries(engine
        ${CMAKE_DL_LIBS}
        engine_utils
    )
