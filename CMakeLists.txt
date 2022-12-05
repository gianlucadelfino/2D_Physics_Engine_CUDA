cmake_minimum_required(VERSION 3.10.0)
project(2d_Phys_engine LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# set(MY_CUDA_ARCHITECTURE 52;70) // TODO

# Conan configuration below
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(
    STATUS
      "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(
    DOWNLOAD
    "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
    "${CMAKE_BINARY_DIR}/conan.cmake" TLS_VERIFY ON)
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(REQUIRES gtest/cci.20210126
                      REQUIRES sdl/2.26.0
                      REQUIRES sdl_ttf/2.0.18
                      REQUIRES sdl_image/2.0.5
                      REQUIRES libpng/1.6.39
                      GENERATORS cmake_find_package)

set(CONAN_PROFILE "default")
conan_cmake_autodetect(settings)

conan_cmake_install(
  PATH_OR_REFERENCE
  .
  BUILD
  missing
  PROFILE
  ${CONAN_PROFILE}
  REMOTE
  conancenter
  SETTINGS
  ${settings})

# Dependencies
find_package(GTest REQUIRED)

find_package(SDL2)
find_package(SDL2_ttf)
find_package(SDL2_image)

# pthread
find_package(Threads)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "No CMAKE_BUILD_TYPE specified, using Release")
endif()

if(NOT WIN32)
  add_compile_options("$<$<CONFIG:RELEASE>:-Wall;-Wextra>")
  add_compile_options("$<$<CONFIG:DEBUG>:-O0;-g;-Wall;-Wextra>")
endif()

file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

include_directories(include)

add_subdirectory(app)
add_subdirectory(src)
