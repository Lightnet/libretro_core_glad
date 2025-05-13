#include <libretro.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <glad/glad.h>
#include <math.h>

// Framebuffer dimensions
#define WIDTH 320
#define HEIGHT 240
#define HW_WIDTH 512  // Match RetroArch HW render size
#define HW_HEIGHT 512

// Global variables
static retro_environment_t environ_cb;
static retro_log_printf_t log_cb;
static retro_video_refresh_t video_cb;
static retro_input_poll_t input_poll_cb;
static retro_input_state_t input_state_cb;
static retro_hw_get_current_framebuffer_t get_current_framebuffer;
static retro_hw_get_proc_address_t get_proc_address;
static struct retro_hw_render_callback hw_render;
static bool initialized = false;
static FILE *log_file = NULL;
static GLuint solid_shader_program = 0;
static GLuint vbo, vao;
static bool gl_initialized = false;
static bool use_default_fbo = false; // Prefer frontend FBO
static float animation_time = 0.0f; // For pulsing animation

// File-based logging
static void fallback_log(const char *level, const char *msg) {
   if (!log_file) {
      log_file = fopen("core.log", "a");
      if (!log_file) {
         fprintf(stderr, "[ERROR] Failed to open core.log\n");
         return;
      }
   }
   fprintf(log_file, "[%s] %s\n", level, msg);
   fflush(log_file);
   fprintf(stderr, "[%s] %s\n", level, msg);
}

static void fallback_log_format(const char *level, const char *fmt, ...) {
   va_list args;
   va_start(args, fmt);
   if (!log_file) {
      log_file = fopen("core.log", "a");
      if (!log_file) {
         fprintf(stderr, "[ERROR] Failed to open core.log\n");
         va_end(args);
         return;
      }
   }
   fprintf(log_file, "[%s] ", level);
   vfprintf(log_file, fmt, args);
   fprintf(log_file, "\n");
   fflush(log_file);
   fprintf(stderr, "[%s] ", level);
   vfprintf(stderr, fmt, args);
   fprintf(stderr, "\n");
   va_end(args);
}

// Check OpenGL errors
static void check_gl_error(const char *context) {
   GLenum err;
   bool has_error = false;
   while ((err = glGetError()) != GL_NO_ERROR) {
      has_error = true;
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] OpenGL error in %s: %d", context, err);
      else
         fallback_log_format("ERROR", "OpenGL error in %s: %d", context, err);
   }
   if (!has_error && log_cb)
      log_cb(RETRO_LOG_DEBUG, "[DEBUG] No OpenGL errors in %s", context);
}

// Shaders (GLSL 330 core)
static const char *solid_vertex_shader_src =
   "#version 330 core\n"
   "layout(location = 0) in vec2 position;\n"
   "void main() {\n"
   "   gl_Position = vec4(position, 0.0, 1.0);\n"
   "}\n";

static const char *solid_fragment_shader_src =
   "#version 330 core\n"
   "out vec4 frag_color;\n"
   "uniform vec4 color;\n"
   "void main() {\n"
   "   frag_color = color;\n"
   "}\n";

// Create shader program
static GLuint create_shader_program(const char *vs_src, const char *fs_src, const char *name) {
   GLuint vs = glCreateShader(GL_VERTEX_SHADER);
   glShaderSource(vs, 1, &vs_src, NULL);
   glCompileShader(vs);
   GLint success;
   glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
   if (!success) {
      char info_log[512];
      glGetShaderInfoLog(vs, 512, NULL, info_log);
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] %s vertex shader compilation failed: %s", name, info_log);
      else
         fallback_log_format("ERROR", "%s vertex shader compilation failed: %s", name, info_log);
      return 0;
   }

   GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(fs, 1, &fs_src, NULL);
   glCompileShader(fs);
   glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
   if (!success) {
      char info_log[512];
      glGetShaderInfoLog(fs, 512, NULL, info_log);
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] %s fragment shader compilation failed: %s", name, info_log);
      else
         fallback_log_format("ERROR", "%s fragment shader compilation failed: %s", name, info_log);
      return 0;
   }

   GLuint program = glCreateProgram();
   glAttachShader(program, vs);
   glAttachShader(program, fs);
   glLinkProgram(program);
   glGetProgramiv(program, GL_LINK_STATUS, &success);
   if (!success) {
      char info_log[512];
      glGetProgramInfoLog(program, 512, NULL, info_log);
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] %s shader program linking failed: %s", name, info_log);
      else
         fallback_log_format("ERROR", "%s shader program linking failed: %s", name, info_log);
      return 0;
   }

   glDeleteShader(vs);
   glDeleteShader(fs);
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] %s shader program created successfully", name);
   return program;
}

// Initialize OpenGL
static void init_opengl(void) {
   if (gl_initialized) {
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] OpenGL already initialized, skipping");
      return;
   }

   if (!get_proc_address) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] No get_proc_address callback provided, cannot initialize GLAD");
      else
         fallback_log("ERROR", "No get_proc_address callback provided, cannot initialize GLAD");
      return;
   }

   if (!gladLoadGLLoader((GLADloadproc)get_proc_address)) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to initialize GLAD");
      else
         fallback_log("ERROR", "Failed to initialize GLAD");
      return;
   }

   const char *gl_version = (const char *)glGetString(GL_VERSION);
   if (!gl_version) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to get OpenGL version");
      else
         fallback_log("ERROR", "Failed to get OpenGL version");
      return;
   }
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] OpenGL version: %s", gl_version);
   else
      fallback_log_format("DEBUG", "OpenGL version: %s", gl_version);

   if (!GLAD_GL_VERSION_3_3) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] OpenGL 3.3 core profile not supported");
      else
         fallback_log("ERROR", "OpenGL 3.3 core profile not supported");
      return;
   }

   solid_shader_program = create_shader_program(solid_vertex_shader_src, solid_fragment_shader_src, "Solid");
   if (!solid_shader_program) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to create solid shader program");
      else
         fallback_log("ERROR", "Failed to create solid shader program");
      return;
   }

   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);

   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindVertexArray(0);
   check_gl_error("init_opengl VAO setup");

   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   check_gl_error("init_opengl state setup");

   gl_initialized = true;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] OpenGL initialized successfully");
   else
      fallback_log("DEBUG", "OpenGL initialized successfully");
}

// Clean up OpenGL
static void deinit_opengl(void) {
   if (gl_initialized) {
      glDeleteProgram(solid_shader_program);
      glDeleteBuffers(1, &vbo);
      glDeleteVertexArrays(1, &vao);
      gl_initialized = false;
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] OpenGL deinitialized");
      else
         fallback_log("DEBUG", "OpenGL deinitialized");
   }
}

// Draw a solid quad
static void draw_solid_quad(float x, float y, float w, float h, float r, float g, float b, float a, float vp_width, float vp_height) {
   if (!glIsProgram(solid_shader_program) || !glIsVertexArray(vao) || !glIsBuffer(vbo)) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Invalid GL state in draw_solid_quad");
      else
         fallback_log("ERROR", "Invalid GL state in draw_solid_quad");
      return;
   }

   float x0 = (x / vp_width) * 2.0f - 1.0f;
   float y0 = 1.0f - (y / vp_height) * 2.0f;
   float x1 = ((x + w) / vp_width) * 2.0f - 1.0f;
   float y1 = 1.0f - ((y + h) / vp_height) * 2.0f;

   float vertices[] = { x0, y0, x1, y0, x0, y1, x1, y1 };
   if (log_cb)
      log_cb(RETRO_LOG_DEBUG, "[DEBUG] Quad vertices: (%f,%f), (%f,%f), (%f,%f), (%f,%f)",
             vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6], vertices[7]);

   glUseProgram(solid_shader_program);
   glBindVertexArray(vao);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

   GLint color_loc = glGetUniformLocation(solid_shader_program, "color");
   glUniform4f(color_loc, r, g, b, a);

   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindVertexArray(0);
   glUseProgram(0);
   check_gl_error("draw_solid_quad");

   if (log_cb)
      log_cb(RETRO_LOG_DEBUG, "[DEBUG] Drew solid quad at (%f, %f), size (%f, %f)", x, y, w, h);
}

// Set environment
void retro_set_environment(retro_environment_t cb) {
   environ_cb = cb;
   if (!cb) {
      fallback_log("ERROR", "retro_set_environment: Null environment callback");
      return;
   }

   bool contentless = true;
   if (environ_cb(RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME, &contentless)) {
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] Content-less support enabled");
   } else {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to set content-less support");
   }
}

// Video refresh callback
void retro_set_video_refresh(retro_video_refresh_t cb) {
   video_cb = cb;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Video refresh callback set");
}

// Input callbacks
void retro_set_input_poll(retro_input_poll_t cb) {
   input_poll_cb = cb;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Input poll callback set");
}

void retro_set_input_state(retro_input_state_t cb) {
   input_state_cb = cb;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Input state callback set");
}

// Stubbed audio callbacks
void retro_set_audio_sample(retro_audio_sample_t cb) { (void)cb; }
void retro_set_audio_sample_batch(retro_audio_sample_batch_t cb) { (void)cb; }

// Initialize core
void retro_init(void) {
   initialized = true;
   struct retro_log_callback logging;
   if (environ_cb && environ_cb(RETRO_ENVIRONMENT_GET_LOG_INTERFACE, &logging)) {
      log_cb = logging.log;
   }
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Hello World core initialized");
   else
      fallback_log("DEBUG", "Hello World core initialized");
}

// Deinitialize core
void retro_deinit(void) {
   deinit_opengl();
   if (log_file) {
      fclose(log_file);
      log_file = NULL;
   }
   initialized = false;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Core deinitialized");
   else
      fallback_log("DEBUG", "Core deinitialized");
}

// System info
void retro_get_system_info(struct retro_system_info *info) {
   memset(info, 0, sizeof(*info));
   info->library_name = "Hello World Core";
   info->library_version = "1.0";
   info->need_fullpath = false;
   info->block_extract = false;
   info->valid_extensions = "";
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] System info: %s v%s", info->library_name, info->library_version);
}

// AV info
void retro_get_system_av_info(struct retro_system_av_info *info) {
   memset(info, 0, sizeof(*info));
   info->geometry.base_width = WIDTH;
   info->geometry.base_height = HEIGHT;
   info->geometry.max_width = HW_WIDTH;
   info->geometry.max_height = HW_HEIGHT;
   info->geometry.aspect_ratio = (float)WIDTH / HEIGHT;
   info->timing.fps = 60.0;
   info->timing.sample_rate = 48000.0;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] AV info: %dx%d, max %dx%d, %.2f fps",
             WIDTH, HEIGHT, HW_WIDTH, HW_HEIGHT, info->timing.fps);
}

// Controller port
void retro_set_controller_port_device(unsigned port, unsigned device) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Controller port device set: port=%u, device=%u", port, device);
}

// Reset core
void retro_reset(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Core reset");
}

// Load game
bool retro_load_game(const struct retro_game_info *game) {
   (void)game;
   if (!environ_cb) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Environment callback not set");
      else
         fallback_log("ERROR", "Environment callback not set");
      return false;
   }

   hw_render.context_type = RETRO_HW_CONTEXT_OPENGL_CORE;
   hw_render.version_major = 3;
   hw_render.version_minor = 3;
   hw_render.context_reset = init_opengl;
   hw_render.context_destroy = deinit_opengl;
   hw_render.bottom_left_origin = true;
   hw_render.depth = true;
   hw_render.stencil = false;
   hw_render.cache_context = false;
   hw_render.debug_context = true;
   if (!environ_cb(RETRO_ENVIRONMENT_SET_HW_RENDER, &hw_render)) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to set OpenGL context");
      else
         fallback_log("ERROR", "Failed to set OpenGL context");
      return false;
   }

   get_current_framebuffer = hw_render.get_current_framebuffer;
   get_proc_address = hw_render.get_proc_address;
   if (!get_proc_address) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] No get_proc_address callback provided");
      else
         fallback_log("ERROR", "No get_proc_address callback provided");
      return false;
   }
   if (!get_current_framebuffer) {
      if (log_cb)
         log_cb(RETRO_LOG_WARN, "[WARN] No get_current_framebuffer callback provided, will attempt default framebuffer");
      else
         fallback_log("WARN", "No get_current_framebuffer callback provided, will attempt default framebuffer");
      use_default_fbo = true;
   } else {
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] get_current_framebuffer callback set successfully");
      else
         fallback_log("DEBUG", "get_current_framebuffer callback set successfully");
      use_default_fbo = false;
   }

   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Game loaded (content-less)");
   return true;
}

// Run frame
void retro_run(void) {
   printf("render\n");
   if (!initialized) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Core not initialized");
      else
         fallback_log("ERROR", "Core not initialized");
      return;
   }

   if (!gl_initialized) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] OpenGL not initialized");
      else
         fallback_log("ERROR", "OpenGL not initialized");
      return;
   }
   printf("gl_initialized\n");

   if (!glIsProgram(solid_shader_program) || !glIsVertexArray(vao) || !glIsBuffer(vbo)) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Invalid GL state");
      else
         fallback_log("ERROR", "Invalid GL state");
      return;
   }
   printf("solid_shader_program\n");

   // Poll input for interactivity
   if (input_poll_cb)
      input_poll_cb();

   // Bind framebuffer
   GLuint fbo = 0;
   if (use_default_fbo || !get_current_framebuffer) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] Using default framebuffer (0)");
      else
         fallback_log("DEBUG", "Using default framebuffer (0)");
   } else {
      fbo = (GLuint)(uintptr_t)(get_current_framebuffer());
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] get_current_framebuffer returned FBO: %u", fbo);
      else
         fallback_log_format("DEBUG", "get_current_framebuffer returned FBO: %u", fbo);
      if (fbo == 0) {
         if (log_cb)
            log_cb(RETRO_LOG_WARN, "[WARN] get_current_framebuffer returned 0, falling back to default framebuffer");
         else
            fallback_log("WARN", "get_current_framebuffer returned 0, falling back to default framebuffer");
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
         use_default_fbo = true;
      } else {
         glBindFramebuffer(GL_FRAMEBUFFER, fbo);
         GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
         if (status != GL_FRAMEBUFFER_COMPLETE) {
            if (log_cb)
               log_cb(RETRO_LOG_ERROR, "[ERROR] Framebuffer %u incomplete (status: %d), falling back to default framebuffer", fbo, status);
            else
               fallback_log_format("ERROR", "Framebuffer %u incomplete (status: %d), falling back to default framebuffer", fbo, status);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            use_default_fbo = true;
         } else {
            if (log_cb)
               log_cb(RETRO_LOG_INFO, "[DEBUG] Successfully bound FBO: %u", fbo);
            else
               fallback_log_format("DEBUG", "Successfully bound FBO: %u", fbo);
         }
      }
   }
   printf("get_current_framebuffer\n");
   check_gl_error("framebuffer binding");

   // Set viewport to match framebuffer dimensions
   GLint viewport[4];
   glGetIntegerv(GL_VIEWPORT, viewport);
   if (viewport[2] != HW_WIDTH || viewport[3] != HW_HEIGHT) {
      glViewport(0, 0, HW_WIDTH, HW_HEIGHT);
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] Set viewport to %dx%d", HW_WIDTH, HW_HEIGHT);
      else
         fallback_log_format("DEBUG", "Set viewport to %dx%d", HW_WIDTH, HW_HEIGHT);
   }
   check_gl_error("glViewport");

   // Clear framebuffer
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   check_gl_error("glClear");

   // Change quad color based on input
   float r = 0.0f, g = 0.5f, b = 0.0f; // Default green
   if (input_state_cb) {
      int a_state = input_state_cb(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_A);
      int b_state = input_state_cb(0, RETRO_DEVICE_JOYPAD, 0, RETRO_DEVICE_ID_JOYPAD_B);
      if (log_cb)
         log_cb(RETRO_LOG_DEBUG, "[DEBUG] Input state: A=%d, B=%d", a_state, b_state);
      if (a_state)
         g = 0.0f, b = 1.0f; // Blue when A is pressed
      if (b_state)
         r = 1.0f, g = 0.0f; // Red when B is pressed
   }

   // Pulsing animation
   animation_time += 0.016f; // ~60 FPS
   float scale = 0.8f + 0.2f * sinf(animation_time * 2.0f);
   float quad_width = HW_WIDTH * scale;
   float quad_height = HW_HEIGHT * scale;
   float quad_x = (HW_WIDTH - quad_width) * 0.5f;
   float quad_y = (HW_HEIGHT - quad_height) * 0.5f;

   // Draw quad
   draw_solid_quad(quad_x, quad_y, quad_width, quad_height, r, g, b, 1.0f, HW_WIDTH, HW_HEIGHT);
   check_gl_error("draw_solid_quad");

   // Log current FBO binding
   GLint current_fbo;
   glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
   if (log_cb)
      log_cb(RETRO_LOG_DEBUG, "[DEBUG] Current FBO binding after rendering: %d", current_fbo);
   else
      fallback_log_format("DEBUG", "Current FBO binding after rendering: %d", current_fbo);

   // Unbind framebuffer
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   check_gl_error("unbind framebuffer");

   // Present frame
   if (video_cb) {
      video_cb(RETRO_HW_FRAME_BUFFER_VALID, 960, 720, 0);
      if (log_cb)
         log_cb(RETRO_LOG_DEBUG, "[DEBUG] Frame presented with size 960x720");
      else
         fallback_log("DEBUG", "Frame presented with size 960x720");
   } else {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] No video callback set");
      else
         fallback_log("ERROR", "No video callback set");
   }
}

// Load special game
bool retro_load_game_special(unsigned game_type, const struct retro_game_info *info, size_t num_info) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] retro_load_game_special called (stubbed)");
   return false;
}

// Unload game
void retro_unload_game(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Game unloaded");
}

// Get region
unsigned retro_get_region(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Region: NTSC");
   return RETRO_REGION_NTSC;
}

// Stubbed serialization
bool retro_serialize(void *data, size_t size) { return false; }
bool retro_unserialize(const void *data, size_t size) { return false; }
size_t retro_serialize_size(void) { return 0; }

// Stubbed cheats
void retro_cheat_reset(void) {}
void retro_cheat_set(unsigned index, bool enabled, const char *code) {}

// Stubbed memory
void *retro_get_memory_data(unsigned id) { return NULL; }
size_t retro_get_memory_size(unsigned id) { return 0; }

// API version
unsigned retro_api_version(void) {
   return RETRO_API_VERSION;
}