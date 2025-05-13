cmake_minimum_required(VERSION 3.14)
project(SimpleTriangle C)
include(FetchContent)
FetchContent_Declare(
    libretro-common
    GIT_REPOSITORY https://github.com/libretro/libretro-common.git
    GIT_TAG master
)
FetchContent_MakeAvailable(libretro-common)
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v0.1.36
)
FetchContent_MakeAvailable(glad)
set(GLAD_PROFILE "core" CACHE STRING "OpenGL profile")
set(GLAD_API "gl=3.3" CACHE STRING "API type/version")
set(GLAD_GENERATOR "c" CACHE STRING "Language to generate")
# testing
set(USE_OPENGL ON)
# hello_world_core library
add_library(hello_world_core SHARED src/lib.c)
# glad
target_link_libraries(hello_world_core PRIVATE glad)
# opengl
if(WIN32 AND USE_OPENGL)
    target_link_libraries(hello_world_core PRIVATE opengl32)
endif()
# include folder for headers
target_include_directories(hello_world_core PRIVATE
    ${libretro-common_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${glad_SOURCE_DIR}/include
)
# compile definitions
target_compile_definitions(hello_world_core PRIVATE _CRT_SECURE_NO_WARNINGS)
if(USE_OPENGL)
    target_compile_definitions(hello_world_core PRIVATE USE_OPENGL)
endif()
# hello_world_core.dll
set_target_properties(hello_world_core PROPERTIES
    PREFIX ""
    OUTPUT_NAME "hello_world_core"
    SUFFIX ".dll"
)
set_property(TARGET hello_world_core PROPERTY C_STANDARD 99)