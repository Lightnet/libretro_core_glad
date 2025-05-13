# libretro_core_glad

# License: MIT
This project is licensed under the MIT License. See LICENSE for details.

# Project Overview
The libretro core glad is a minimal, content-less Libretro core that demonstrates hardware-accelerated rendering using OpenGL 3.3 and the GLAD library within the RetroArch frontend. The core renders a pulsing green quad (optionally changing to blue or red based on joypad input) in a 512x512 framebuffer, scaled to a 960x720 window by RetroArch. It serves as an educational example for building Libretro cores with modern OpenGL, showcasing:

- Hardware Rendering: Uses OpenGL 3.3 core profile with GLAD for function loading.
- Framebuffer Management: Renders to a frontend-provided framebuffer (FBO) or falls back to the default FBO.
- Input Handling: Changes quad color based on joypad inputs (A for blue, B for red).
- Animation: Implements a simple pulsing animation for the quad size.
- Content-less Operation: Runs without ROMs or game content, ideal for prototyping.

This project is designed for developers learning the Libretro API, OpenGL integration, and RetroArch’s rendering pipeline. The codebase is lightweight, well-documented, and extensible for adding features like textures, complex animations, or save states.

# Repository Structure

```text
libretro_core_glad/
├── src/
│   └── lib.c              # Main core implementation (Libretro API, OpenGL rendering)
├── build/
└── README.md              # Brief project overview and setup instructions
```

# Setup Instructions

## Prerequisites

- Operating System: Windows 10/11 (other platforms possible with modifications).
- RetroArch: Version 1.21.0 or later, installed at "path"\RetroArch-Win64.
- Compiler: MSVC (Visual Studio 2019/2022) with C++ support.
- Git: For cloning the repository and submodules.
- Dependencies:
    - GLAD (from cmake repo).
    - Libretro headers (from cmake repo).

## Installation
1. Clone the Repository:
```
git clone https://github.com/Lightnet/libretro_core_glad.git
```
```
cd libretro_core_glad
```

2. Set Up RetroArch:
- Download and install RetroArch from [retroarch.com](https://www.retroarch.com).
  - Recommended path: "path"\RetroArch-Win64.
  - Configure video driver to glcore (Settings > Driver > Video > glcore).
  - Enable verbose logging (Settings > Logging > Show Log Messages).

3. Build the Core:

```
./build.bat
```

- Output: build\Debug\hello_world_core.dll.

4. Copy the Core to RetroArch:
- Copy build\Debug\hello_world_core.dll to "path"\RetroArch-Win64\cores

5. Run in RetroArch:

- Launch RetroArch:
```
cd "path"/RetroArch-Win64/retroarch.exe --verbose -L cores\hello_world_core.dll
```
 - The core should display a pulsing green quad in a 960x720 window.
 - Press Joypad A (e.g., keyboard Z) to turn the quad blue, or B (X) for red.

## Troubleshooting:
 - Black Screen:
    - Ensure glcore driver is selected.
    - Check logs (core.log or RetroArch console) for OpenGL errors.
    - Revert to default FBO by setting use_default_fbo = true in retro_run.
 - No Input Response:
    - Verify input mappings (Settings > Input > Port 1 Controls).
    - Map A to Z and B to X for keyboard testing.
    - Check [DEBUG] Input state: A=<0/1>, B=<0/1> in logs.
 - Build Errors:
    - Ensure GLAD submodule is initialized (git submodule update --init).
    - Verify MSVC and C++ toolsets are installed.

# How It Works

## Core Functionality
The core implements a minimal Libretro core that:
Initializes an OpenGL 3.3 context using GLAD.
    
- Renders a single quad in a 512x512 framebuffer, scaled to 960x720 by RetroArch.
- Supports content-less operation (no ROMs required).
- Changes quad color based on input (green default, blue for A, red for B).
- Animates the quad size (pulsing between 80% and 100% of viewport).

## Key Components

- Libretro API:  
    - Implements required functions (retro_init, retro_run, retro_load_game, etc.).
    - Sets up hardware rendering via RETRO_ENVIRONMENT_SET_HW_RENDER.
    - Handles input via retro_set_input_poll and retro_set_input_state.
- OpenGL Rendering:
    - Uses GLAD to load OpenGL 3.3 core profile functions.
    - Creates a shader program for solid-color rendering.
    - Draws a quad using vertex buffer objects (VBOs) and vertex array objects (VAOs).
    - Renders to a frontend-provided FBO or the default FBO (0).
- GLAD Integration:
    - GLAD generates OpenGL function pointers at runtime.
    - Loaded via gladLoadGLLoader((GLADloadproc)get_proc_address) in init_opengl.
- Input and Animation:
    - Polls joypad input to change quad color.
    - Updates quad size using a sine-based animation (sinf(animation_time * 2.0f)).

## Logic Design
The core’s logic is structured around the Libretro lifecycle, interacting with RetroArch and GLAD. Below is a detailed explanation with a visual diagram.

Logic Flow:

1. Initialization (retro_init):
    - Sets up logging and core state.
        
2. Game Loading (retro_load_game):
    - Requests OpenGL 3.3 context.
    - Stores get_current_framebuffer and get_proc_address callbacks.
        
3. OpenGL Setup (init_opengl):
    - Loads GLAD, creates shaders, and sets up VBO/VAO.
        
4. Frame Rendering (retro_run):
    - Polls input.
    - Binds frontend FBO (or default).
    - Sets viewport (512x512).
    - Clears framebuffer.
    - Draws animated quad with input-based color.
    - Presents frame via video_cb(RETRO_HW_FRAME_BUFFER_VALID, 960, 720, 0).
        
5. Cleanup (retro_deinit):
    - Frees OpenGL resources and closes log file.

# Visual Diagram

```plaintext
+-------------------+       +-------------------+       +-------------------+
|   RetroArch       |       |   Libretro Core   |       |   GLAD/OpenGL     |
| (Frontend)        |       | (hello_world_core)|       |                   |
+-------------------+       +-------------------+       +-------------------+
| Initialize        | ----> | retro_init()      |       |                   |
|                   |       | - Set logging     |       |                   |
|                   |       +-------------------+       |                   |
| Load Core         | ----> | retro_load_game() |       |                   |
|                   |       | - Request GL 3.3  |       |                   |
|                   |       | - Store callbacks |       |                   |
|                   |       +-------------------+       |                   |
|                   |       | init_opengl()     | ----> | - Load GLAD       |
|                   |       | - Create shaders  |       | - Create VBO/VAO  |
|                   |       | - Set up GL state |       | - Compile shaders |
|                   |       +-------------------+       +-------------------+
| Run Frame         | ----> | retro_run()       |       |                   |
|                   |       | - Poll input      |       |                   |
|                   |       | - Bind FBO        | ----> | - Render quad     |
|                   |       | - Set viewport    |       | - Update framebuffer|
|                   |       | - Draw quad       |       |                   |
|                   |       | - Present frame   |       |                   |
|                   |       +-------------------+       +-------------------+
| Unload Core       | ----> | retro_deinit()    |       |                   |
|                   |       | - Free GL resources |     |                   |
|                   |       | - Close log file  |       |                   |
+-------------------+       +-------------------+       +-------------------+
```

## Description:
- RetroArch: Manages the core lifecycle, provides callbacks (get_current_framebuffer, get_proc_address), and handles windowing/input.  
- Libretro Core: Implements the Libretro API, orchestrates rendering, and manages OpenGL state.
- GLAD/OpenGL: Loads OpenGL functions and executes rendering commands.

## GLAD Integration

GLAD is used to load OpenGL function pointers at runtime, ensuring compatibility across platforms. Here’s how it integrates with the core:

## Steps:

1. Callback Acquisition:
    - In retro_load_game, the core stores hw_render.get_proc_address from RetroArch.
        
2. GLAD Initialization:
    - In init_opengl, GLAD is loaded using gladLoadGLLoader((GLADloadproc)get_proc_address).
        
3. Function Usage:
    - GLAD provides OpenGL 3.3 core profile functions (e.g., glCreateShader, glBindFramebuffer).  
    - The core uses these to set up shaders, VBOs, and render the quad.
        
4. Error Checking:
    - check_gl_error uses GLAD-loaded glGetError to detect issues.

# Visual Diagram

```plaintext
+-------------------+       +-------------------+       +-------------------+
|   RetroArch       |       |   Libretro Core   |       |   GLAD            |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
| Provide           | ----> | retro_load_game() |       |                   |
| get_proc_address  |       | - Store callback  |       |                   |
| callback          |       +-------------------+       |                   |
|                   |       | init_opengl()     | ----> | gladLoadGLLoader()|
|                   |       | - Call GLAD       |       | - Load OpenGL 3.3 |
|                   |       +-------------------+       |   functions       |
|                   |       | retro_run()       | ----> | - Use gl*() funcs |
|                   |       | - Render quad     |       |   (e.g., glDrawArrays)|
|                   |       | - Check errors    |       | - Check glGetError|
+-------------------+       +-------------------+       +-------------------+
```

## Description:

- RetroArch: Supplies get_proc_address to load OpenGL function pointers.    
- Libretro Core: Calls GLAD and uses loaded functions for rendering.
- GLAD: Dynamically loads OpenGL 3.3 functions via get_proc_address.

## RetroArch and Libretro Lifecycle

The Libretro API defines a lifecycle for cores, managed by RetroArch. Below is an overview of the lifecycle and how this core implements it.

## Lifecycle Stages:

1. Core Loading:
    - RetroArch loads the core DLL (hello_world_core.dll).
    - Calls retro_set_environment, retro_set_video_refresh, retro_set_input_poll, retro_set_input_state, etc., to set callbacks.
    - Calls retro_init to initialize core state.
        
2. Game Loading:
    - Calls retro_load_game to set up OpenGL context and store callbacks.
    - Core requests hardware rendering (RETRO_ENVIRONMENT_SET_HW_RENDER).
        
3. Frame Execution:
    - Calls retro_run repeatedly (~60 FPS).
    - Core polls input, renders the quad, and presents the frame.
        
4. Core Unloading:
    - Calls retro_unload_game to clean up game-specific resources.
    - Calls retro_deinit to free OpenGL resources and close logs.
    - Unloads the DLL.

## Visual Diagram

```plaintext
+-------------------+       +-------------------+
|   RetroArch       |       |   Libretro Core   |
| (Frontend)        |       | (hello_world_core)|
+-------------------+       +-------------------+
| Load DLL          | ----> |                   |
|                   |       |                   |
| retro_set_*()     | ----> | Set callbacks     |
| (environment,     |       | (video, input, etc.)|
| video, input, etc.)|      |                   |
|                   |       +-------------------+
| retro_init()      | ----> | Initialize core   |
|                   |       | - Set logging     |
|                   |       +-------------------+
| retro_load_game() | ----> | Load game         |
|                   |       | - Request GL 3.3  |
|                   |       | - Init OpenGL     |
|                   |       +-------------------+
| retro_run()       | ----> | Run frame         |
| (repeated ~60 FPS)|       | - Poll input      |
|                   |       | - Render quad     |
|                   |       | - Present frame   |
|                   |       +-------------------+
| retro_unload_game()| ----> | Unload game      |
|                   |       | - Clean up game   |
|                   |       +-------------------+
| retro_deinit()    | ----> | Deinitialize core|
|                   |       | - Free GL resources|
|                   |       | - Close log file  |
|                   |       +-------------------+
| Unload DLL        | ----> |                   |
+-------------------+       +-------------------+
```

## Description:
- RetroArch: Orchestrates the lifecycle, calling Libretro API functions and providing callbacks.
- Libretro Core: Responds to lifecycle events, managing initialization, rendering, and cleanup.

## Libretro Format

The Libretro API is a cross-platform interface for emulators, game engines, and multimedia applications. Key aspects used in this core:

- Content-less Support: Set via RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME, allowing the core to run without ROMs.
- Hardware Rendering: Uses RETRO_ENVIRONMENT_SET_HW_RENDER to request an OpenGL 3.3 context.
- Callbacks:
    - retro_environment_t: Configures core settings (e.g., content-less mode).
    - retro_video_refresh_t: Presents frames to RetroArch.
    - retro_input_poll_t/retro_input_state_t: Handles input.
    - retro_hw_get_current_framebuffer_t: Provides the frontend’s FBO.
    - retro_hw_get_proc_address_t: Loads OpenGL functions.
- AV Info: Defines geometry (320x240 base, 512x512 max), 60 FPS, and 48kHz audio (stubbed).
- API Version: Uses Libretro API v1 (RETRO_API_VERSION).
    
## Usage

1. Running the Core:
    - Load the core in RetroArch via Load Core > Hello World Core.
    - Start without content (Start Core).
    - Observe the pulsing green quad.
        
2. Interacting:
    - Press Joypad A to turn the quad blue.
    - Press Joypad B to turn the quad red.
    - Release buttons to revert to green.
        
3. Debugging:
    - Check core.log in the project root for detailed logs.
    - Use RetroArch’s verbose output (--verbose).
        

## Extending the Core

1. Add Textures:
    - Use stb_image.h to load PNG images.
    - Update shaders to sample textures (see previous responses).
        
2. Complex Animations:
    - Move or rotate the quad based on animation_time.
        
3. Multiple Quads:
    - Render additional quads with different colors/positions.
        
4. Save States:
    - Implement retro_serialize/retro_unserialize to save animation_time and color.

# Credits:

This project was developed with the help of the following resources and references:

- Libretro Documentation: The official [Libretro API documentation](https://docs.libretro.com/) provided essential guidance on implementing the Libretro core interface, hardware rendering, and content-less operation.
    
- RetroArch: The [RetroArch frontend](https://www.retroarch.com/) and its open-source codebase were critical for testing and understanding the hardware rendering pipeline, particularly the OpenGL core profile integration.
    
- GLAD: The [GLAD OpenGL loader generator](https://glad.dav1d.de/) was used to generate OpenGL 3.3 core profile function pointers, with its documentation aiding proper integration.
    
- Libretro GL Example: The [libretro-gl example](https://github.com/libretro/libretro-samples/tree/master/video/opengl) served as a reference for setting up OpenGL hardware rendering and framebuffer handling within a Libretro core.
    
- OpenGL Documentation: The [OpenGL 3.3 core profile specification](https://www.opengl.org/registry/) and [Khronos Group resources](https://www.khronos.org/opengl/) informed the shader and rendering pipeline implementation.
    
- Community Support: Discussions and code snippets from the Libretro and RetroArch communities on GitHub and forums provided insights into debugging and optimizing the core.
    

Special thanks to the open-source community for maintaining these resources, enabling projects like this one to thrive.
