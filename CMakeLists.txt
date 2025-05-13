cmake_minimum_required(VERSION 3.14)
project(SimpleTriangle C)

# Enable FetchContent
include(FetchContent)

# Fetch libretro-common for headers
FetchContent_Declare(
    libretro-common
    GIT_REPOSITORY https://github.com/libretro/libretro-common.git
    GIT_TAG master
)
FetchContent_MakeAvailable(libretro-common)

# # Declare GLFW
# FetchContent_Declare(
#     glfw
#     GIT_REPOSITORY https://github.com/glfw/glfw.git
#     GIT_TAG 3.4
# )
# FetchContent_MakeAvailable(glfw)

# # Configure GLFW options
# set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
# set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Declare GLAD
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v0.1.36
)
FetchContent_MakeAvailable(glad)

# Configure GLAD options
set(GLAD_PROFILE "core" CACHE STRING "OpenGL profile")
set(GLAD_API "gl=3.3" CACHE STRING "API type/version")
set(GLAD_GENERATOR "c" CACHE STRING "Language to generate")

# Add GLAD library
# add_subdirectory(${glad_SOURCE_DIR} ${glad_BINARY_DIR})

# Create executable
# add_executable(simple_triangle src/main.c)

# Link libraries
# target_link_libraries(simple_triangle PRIVATE glfw glad)

# Include directories for GLAD
# target_include_directories(simple_triangle PRIVATE 
#   ${glad_SOURCE_DIR}/include
# )

# add_executable(simple_trianglepp src/main.cpp)
# target_link_libraries(simple_trianglepp PRIVATE glfw glad)
# target_include_directories(simple_trianglepp PRIVATE ${glad_SOURCE_DIR}/include)

# Define the shared library
add_library(hello_world_core SHARED 
  src/lib.c 
  # src/font.h
)

target_link_libraries(hello_world_core PRIVATE 
  glad
)


# Set include directories
target_include_directories(hello_world_core PRIVATE
    ${libretro-common_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${glad_SOURCE_DIR}/include
)

# Set compile definitions
target_compile_definitions(hello_world_core PRIVATE
    _CRT_SECURE_NO_WARNINGS
)

# Ensure the output is a .dll with no prefix
set_target_properties(hello_world_core PROPERTIES
    PREFIX ""
    OUTPUT_NAME "hello_world_core"
    SUFFIX ".dll"
)

# Set C standard
set_property(TARGET hello_world_core PROPERTY C_STANDARD 99)
