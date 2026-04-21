cmake_minimum_required(VERSION 3.15)
project(panda3d_rl_sim_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_raycaster src/raycaster.cpp)

target_compile_options(_raycaster PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W4 /permissive->
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -Wextra -Wpedantic>
)

install(TARGETS _raycaster LIBRARY DESTINATION panda3d_rl_sim_cpp)
