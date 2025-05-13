#include <libretro.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#ifdef USE_OPENGL
#include <glad/glad.h>
#endif

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
static bool initialized = false;
static bool contentless_set = false;
static bool pixel_format_set = false;
static int env_call_count = 0;
static FILE *log_file = NULL;
#ifdef USE_OPENGL
static GLuint solid_shader_program = 0;
static GLuint vbo, vao;
static bool gl_initialized = false;
static bool use_default_fbo = true; // Force default FBO for HW rendering
#endif
// static bool use_software_rendering = true; // Enable software rendering
static bool use_software_rendering = false; // Enable software rendering
static uint16_t framebuffer[WIDTH * HEIGHT]; // For RGB565 or 0RGB1555

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

#ifdef USE_OPENGL
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
   else if (!has_error)
      fallback_log_format("DEBUG", "No OpenGL errors in %s", context);
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
   else
      fallback_log_format("DEBUG", "%s shader program created successfully", name);
   return program;
}

// Initialize OpenGL
static void init_opengl(void) {
   if (gl_initialized) {
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] OpenGL already initialized, skipping");
      else
         fallback_log("DEBUG", "OpenGL already initialized, skipping");
      return;
   }

   if (!gladLoadGL()) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to initialize GLAD, OpenGL version: %s", glGetString(GL_VERSION));
      else
         fallback_log_format("ERROR", "Failed to initialize GLAD, OpenGL version: %s", glGetString(GL_VERSION));
      return;
   }

   const char *gl_version = (const char *)glGetString(GL_VERSION);
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] OpenGL version: %s", gl_version ? gl_version : "unknown");
   else
      fallback_log_format("DEBUG", "OpenGL version: %s", gl_version ? gl_version : "unknown");

   // Create solid color shader program
   solid_shader_program = create_shader_program(solid_vertex_shader_src, solid_fragment_shader_src, "Solid");
   if (!solid_shader_program) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to create solid shader program");
      else
         fallback_log("ERROR", "Failed to create solid shader program");
      return;
   }

   // Set up vertex buffer and array
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
   if (!glIsProgram(solid_shader_program)) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Invalid shader program in draw_solid_quad");
      else
         fallback_log("ERROR", "Invalid shader program in draw_solid_quad");
      return;
   }
   if (!glIsVertexArray(vao)) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Invalid VAO in draw_solid_quad");
      else
         fallback_log("ERROR", "Invalid VAO in draw_solid_quad");
      return;
   }
   if (!glIsBuffer(vbo)) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Invalid VBO in draw_solid_quad");
      else
         fallback_log("ERROR", "Invalid VBO in draw_solid_quad");
      return;
   }

   float x0 = (x / vp_width) * 2.0f - 1.0f;
   float y0 = 1.0f - (y / vp_height) * 2.0f;
   float x1 = ((x + w) / vp_width) * 2.0f - 1.0f;
   float y1 = 1.0f - ((y + h) / vp_height) * 2.0f;

   float vertices[] = {
      x0, y0,
      x1, y0,
      x0, y1,
      x1, y1
   };

   glUseProgram(solid_shader_program);
   check_gl_error("draw_solid_quad glUseProgram");
   glBindVertexArray(vao);
   check_gl_error("draw_solid_quad glBindVertexArray");
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   check_gl_error("draw_solid_quad glBindBuffer");
   glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
   check_gl_error("draw_solid_quad glBufferSubData");

   GLint color_loc = glGetUniformLocation(solid_shader_program, "color");
   glUniform4f(color_loc, r, g, b, a);
   check_gl_error("draw_solid_quad glUniform4f");

   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   check_gl_error("draw_solid_quad glDrawArrays");

   glBindBuffer(GL_ARRAY_BUFFER, 0);
   glBindVertexArray(0);
   glUseProgram(0);
   check_gl_error("draw_solid_quad cleanup");

   if (log_cb)
      log_cb(RETRO_LOG_DEBUG, "[DEBUG] Drew solid quad at (%f, %f), size (%f, %f), vp: %fx%f",
             x, y, w, h, vp_width, vp_height);
   else
      fallback_log_format("DEBUG", "Drew solid quad at (%f, %f), size (%f, %f), vp: %fx%f",
                         x, y, w, h, vp_width, vp_height);
}
#endif

// Set environment
void retro_set_environment(retro_environment_t cb) {
   environ_cb = cb;
   env_call_count++;
   if (!cb) {
      fallback_log("ERROR", "retro_set_environment: Null environment callback");
      return;
   }

   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] retro_set_environment called (count: %d)", env_call_count);
   else
      fallback_log_format("DEBUG", "retro_set_environment called (count: %d)", env_call_count);

   // Set content-less support
   if (!contentless_set) {
      bool contentless = true;
      if (environ_cb(RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME, &contentless)) {
         contentless_set = true;
         if (log_cb)
            log_cb(RETRO_LOG_INFO, "[DEBUG] Content-less support enabled");
         else
            fallback_log("DEBUG", "Content-less support enabled");
      } else {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to set content-less support");
         else
            fallback_log("ERROR", "Failed to set content-less support");
      }
   }
}

// Video refresh callback
void retro_set_video_refresh(retro_video_refresh_t cb) {
   video_cb = cb;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Video refresh callback set");
   else
      fallback_log("DEBUG", "Video refresh callback set");
}

// Input callbacks
void retro_set_input_poll(retro_input_poll_t cb) {
   input_poll_cb = cb;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Input poll callback set");
   else
      fallback_log("DEBUG", "Input poll callback set");
}

void retro_set_input_state(retro_input_state_t cb) {
   input_state_cb = cb;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Input state callback set");
   else
      fallback_log("DEBUG", "Input state callback set");
}

// Stubbed audio callbacks
void retro_set_audio_sample(retro_audio_sample_t cb) { (void)cb; }
void retro_set_audio_sample_batch(retro_audio_sample_batch_t cb) { (void)cb; }

// Initialize core
void retro_init(void) {
   initialized = true;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Hello World core initialized");
   else
      fallback_log("DEBUG", "Hello World core initialized");

   // Set up logging
   struct retro_log_callback logging;
   if (environ_cb && environ_cb(RETRO_ENVIRONMENT_GET_LOG_INTERFACE, &logging)) {
      log_cb = logging.log;
      if (log_cb)
         log_cb(RETRO_LOG_INFO, "[DEBUG] Logging callback initialized");
   } else {
      if (log_cb)
         log_cb(RETRO_LOG_WARN, "[WARN] Failed to get log interface");
      else
         fallback_log("WARN", "Failed to get log interface");
   }
}

// Deinitialize core
void retro_deinit(void) {
#ifdef USE_OPENGL
   deinit_opengl();
#endif
   if (log_file) {
      fclose(log_file);
      log_file = NULL;
   }
   initialized = false;
   contentless_set = false;
   pixel_format_set = false;
   env_call_count = 0;
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
      log_cb(RETRO_LOG_INFO, "[DEBUG] System info: %s v%s, need_fullpath=%d",
             info->library_name, info->library_version, info->need_fullpath);
   else
      fallback_log_format("DEBUG", "System info: %s v%s, need_fullpath=%d",
                         info->library_name, info->library_version, info->need_fullpath);
}

// AV info
void retro_get_system_av_info(struct retro_system_av_info *info) {
   memset(info, 0, sizeof(*info));
   info->geometry.base_width = WIDTH;
   info->geometry.base_height = HEIGHT;
   info->geometry.max_width = WIDTH;
   info->geometry.max_height = HEIGHT;
   info->geometry.aspect_ratio = (float)WIDTH / HEIGHT;
   info->timing.fps = 60.0;
   info->timing.sample_rate = 48000.0;
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] AV info: %dx%d, %.2f fps", WIDTH, HEIGHT, info->timing.fps);
   else
      fallback_log_format("DEBUG", "AV info: %dx%d, %.2f fps", WIDTH, HEIGHT, info->timing.fps);
}

// Controller port
void retro_set_controller_port_device(unsigned port, unsigned device) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Controller port device set: port=%u, device=%u", port, device);
   else
      fallback_log_format("DEBUG", "Controller port device set: port=%u, device=%u", port, device);
}

// Reset core
void retro_reset(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Core reset");
   else
      fallback_log("DEBUG", "Core reset");
}

// Load game
bool retro_load_game(const struct retro_game_info *game) {
   (void)game;
   if (!environ_cb) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Environment callback not set in retro_load_game");
      else
         fallback_log("ERROR", "Environment callback not set in retro_load_game");
      return false;
   }

   if (use_software_rendering) {
      if (!pixel_format_set) {
         enum retro_pixel_format pixel_format = RETRO_PIXEL_FORMAT_RGB565;
         //enum retro_pixel_format pixel_format = RETRO_PIXEL_FORMAT_0RGB1555; // Uncomment to test 0RGB1555
         int retries = 3;
         while (retries > 0) {
            if (environ_cb(RETRO_ENVIRONMENT_SET_PIXEL_FORMAT, &pixel_format)) {
               pixel_format_set = true;
               if (log_cb)
                  log_cb(RETRO_LOG_INFO, "[DEBUG] Pixel format set to %s in retro_load_game",
                         pixel_format == RETRO_PIXEL_FORMAT_RGB565 ? "RGB565" : "0RGB1555");
               else
                  fallback_log("DEBUG", pixel_format == RETRO_PIXEL_FORMAT_RGB565 ? "Pixel format set to RGB565 in retro_load_game" : "Pixel format set to 0RGB1555 in retro_load_game");
               break;
            }
            retries--;
            if (log_cb)
               log_cb(RETRO_LOG_WARN, "[WARN] Failed to set pixel format to %s, retries left: %d",
                      pixel_format == RETRO_PIXEL_FORMAT_RGB565 ? "RGB565" : "0RGB1555", retries);
            else
               fallback_log_format("WARN", "Failed to set pixel format to %s, retries left: %d",
                                  pixel_format == RETRO_PIXEL_FORMAT_RGB565 ? "RGB565" : "0RGB1555", retries);
         }
         if (!pixel_format_set) {
            if (log_cb)
               log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to set pixel format to %s after retries",
                      pixel_format == RETRO_PIXEL_FORMAT_RGB565 ? "RGB565" : "0RGB1555");
            else
               fallback_log("ERROR", pixel_format == RETRO_PIXEL_FORMAT_RGB565 ? "Failed to set pixel format to RGB565 after retries" : "Failed to set pixel format to 0RGB1555 after retries");
            return false;
         }
      } else {
         if (log_cb)
            log_cb(RETRO_LOG_DEBUG, "[DEBUG] Pixel format already set, skipping");
         else
            fallback_log("DEBUG", "Pixel format already set, skipping");
      }
   } else {
#ifdef USE_OPENGL
      // Set OpenGL context
      struct retro_hw_render_callback hw_render = {
         .context_type = RETRO_HW_CONTEXT_OPENGL_CORE,
         .version_major = 3,
         .version_minor = 3,
         .context_reset = init_opengl,
         .context_destroy = deinit_opengl,
         .get_current_framebuffer = NULL,
         .bottom_left_origin = true,
         .depth = true,
         .stencil = false,
         .cache_context = false,
         .debug_context = true
      };
      if (!environ_cb(RETRO_ENVIRONMENT_SET_HW_RENDER, &hw_render)) {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] Failed to set OpenGL context in retro_load_game");
         else
            fallback_log("ERROR", "Failed to set OpenGL context in retro_load_game");
         return false;
      }

      // Store and validate get_current_framebuffer callback
      get_current_framebuffer = hw_render.get_current_framebuffer;
      if (!get_current_framebuffer) {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] No get_current_framebuffer callback provided, will use default framebuffer");
         else
         fallback_log("ERROR", "No get_current_framebuffer callback provided, will use default framebuffer");
         use_default_fbo = true;
      } else {
         if (log_cb)
            log_cb(RETRO_LOG_INFO, "[DEBUG] get_current_framebuffer callback set successfully");
         else
         fallback_log("DEBUG", "get_current_framebuffer callback set successfully");
      }
#endif
   }

   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Game loaded (content-less): Displaying Hello World");
   else
      fallback_log("DEBUG", "Game loaded (content-less): Displaying Hello World");
   return true;
}

// Run frame
void retro_run(void) {
   if (log_cb) {
      log_cb(RETRO_LOG_DEBUG, "[DEBUG] Entering retro_run");
      log_cb(RETRO_LOG_DEBUG, "[DEBUG] Flushing logs");
   } else {
      fallback_log("DEBUG", "Entering retro_run");
      fallback_log("DEBUG", "Flushing logs");
   }

   if (!initialized) {
      if (log_cb)
         log_cb(RETRO_LOG_ERROR, "[ERROR] Core not initialized, exiting retro_run");
      else
         fallback_log("ERROR", "Core not initialized, exiting retro_run");
      return;
   }

   if (use_software_rendering) {
      // Software rendering: Fill framebuffer with green (RGB565 or 0RGB1555)
      for (int i = 0; i < WIDTH * HEIGHT; i++)
         framebuffer[i] = 0x07E0; // Green in RGB565 (5-6-5)
         //framebuffer[i] = 0x7C00; // Green in 0RGB1555 (5-5-5), uncomment if testing 0RGB1555

      if (video_cb) {
         video_cb(framebuffer, WIDTH, HEIGHT, WIDTH * sizeof(uint16_t));
         if (log_cb)
            log_cb(RETRO_LOG_DEBUG, "[DEBUG] Frame presented (software)");
         else
            fallback_log("DEBUG", "Frame presented (software)");
      } else {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] No video callback set, exiting retro_run");
         else
            fallback_log("ERROR", "No video callback set, exiting retro_run");
         return;
      }
   } else {
#ifdef USE_OPENGL
      if (!gl_initialized) {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] OpenGL not initialized, exiting retro_run");
         else
            fallback_log("ERROR", "OpenGL not initialized, exiting retro_run");
         return;
      }

      // Validate shader program
      if (!glIsProgram(solid_shader_program)) {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] Solid shader program is invalid, exiting retro_run");
         else
            fallback_log("ERROR", "Solid shader program is invalid, exiting retro_run");
         return;
      }

      // Validate VAO and VBO
      if (!glIsVertexArray(vao)) {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] Invalid VAO, exiting retro_run");
         else
            fallback_log("ERROR", "Invalid VAO, exiting retro_run");
         return;
      }
      if (!glIsBuffer(vbo)) {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] Invalid VBO, exiting retro_run");
         else
         fallback_log("ERROR", "Invalid VBO, exiting retro_run");
         return;
      }

      // Log GL state
      const char *gl_version = (const char *)glGetString(GL_VERSION);
      if (log_cb)
         log_cb(RETRO_LOG_DEBUG, "[DEBUG] Current OpenGL version: %s", gl_version ? gl_version : "unknown");
      else
         fallback_log_format("DEBUG", "Current OpenGL version: %s", gl_version ? gl_version : "unknown");

      // Bind framebuffer
      GLuint fbo = 0;
      if (use_default_fbo || !get_current_framebuffer) {
         glBindFramebuffer(GL_FRAMEBUFFER, 0);
         if (log_cb)
            log_cb(RETRO_LOG_WARN, "[WARN] Using default framebuffer %s",
                   get_current_framebuffer ? "due to use_default_fbo" : "because get_current_framebuffer is NULL");
         else
         fallback_log("WARN", get_current_framebuffer ? "Using default framebuffer due to use_default_fbo" : "Using default framebuffer because get_current_framebuffer is NULL");
      } else {
         fbo = (GLuint)(uintptr_t)(get_current_framebuffer());
         if (fbo == 0) {
            if (log_cb)
               log_cb(RETRO_LOG_ERROR, "[ERROR] get_current_framebuffer returned 0, using default framebuffer");
            else
               fallback_log("ERROR", "get_current_framebuffer returned 0, using default framebuffer");
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            use_default_fbo = true;
         } else {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
               if (log_cb)
                  log_cb(RETRO_LOG_ERROR, "[ERROR] Framebuffer incomplete: %d, using default framebuffer", status);
               else
                  fallback_log_format("ERROR", "Framebuffer incomplete: %d, using default framebuffer", status);
               glBindFramebuffer(GL_FRAMEBUFFER, 0);
               use_default_fbo = true;
            } else {
               GLint color_attachment;
               glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &color_attachment);
               if (log_cb)
                  log_cb(RETRO_LOG_DEBUG, "[DEBUG] Binding FBO: %u, color attachment: %d", fbo, color_attachment);
               else
                  fallback_log_format("DEBUG", "Binding FBO: %u, color attachment: %d", fbo, color_attachment);
            }
         }
      }
      check_gl_error("framebuffer binding");

      // Clear framebuffer
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      check_gl_error("glClear");

      // Set viewport
      float vp_width = HW_WIDTH;
      float vp_height = HW_HEIGHT;
      glViewport(0, 0, (GLint)vp_width, (GLint)vp_height);
      check_gl_error("glViewport");

      // Draw full-screen green quad
      draw_solid_quad(0.0f, 0.0f, vp_width, vp_height, 0.0f, 1.0f, 0.0f, 1.0f, vp_width, vp_height);

      // Unbind FBO
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      check_gl_error("unbind framebuffer");

      // Present frame
      if (video_cb) {
         video_cb(NULL, WIDTH, HEIGHT, 0);
         if (log_cb)
            log_cb(RETRO_LOG_DEBUG, "[DEBUG] Frame presented");
         else
            fallback_log("DEBUG", "Frame presented");
      } else {
         if (log_cb)
            log_cb(RETRO_LOG_ERROR, "[ERROR] No video callback set, exiting retro_run");
         else
         fallback_log("ERROR", "No video callback set, exiting retro_run");
         return;
      }
#endif
   }

   if (log_cb)
      log_cb(RETRO_LOG_DEBUG, "[DEBUG] Exiting retro_run");
   else
      fallback_log("DEBUG", "Exiting retro_run");
}

// Load special game
bool retro_load_game_special(unsigned game_type, const struct retro_game_info *info, size_t num_info) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] retro_load_game_special called (stubbed)");
   else
      fallback_log("DEBUG", "retro_load_game_special called (stubbed)");
   return false;
}

// Unload game
void retro_unload_game(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Game unloaded");
   else
      fallback_log("DEBUG", "Game unloaded");
}

// Get region
unsigned retro_get_region(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Region: NTSC");
   else
      fallback_log("DEBUG", "Region: NTSC");
   return RETRO_REGION_NTSC;
}

// Stubbed serialization
bool retro_serialize(void *data, size_t size) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Serialize called (stubbed)");
   else
      fallback_log("DEBUG", "Serialize called (stubbed)");
   return false;
}
bool retro_unserialize(const void *data, size_t size) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Unserialize called (stubbed)");
   else
      fallback_log("DEBUG", "Unserialize called (stubbed)");
   return false;
}
size_t retro_serialize_size(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Serialize size: 0");
   else
      fallback_log("DEBUG", "Serialize size: 0");
   return 0;
}

// Stubbed cheats
void retro_cheat_reset(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Cheat reset (stubbed)");
   else
      fallback_log("DEBUG", "Cheat reset (stubbed)");
}
void retro_cheat_set(unsigned index, bool enabled, const char *code) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Cheat set: index=%u, enabled=%d, code=%s (stubbed)", index, enabled, code);
   else
      fallback_log_format("DEBUG", "Cheat set: index=%u, enabled=%d, code=%s (stubbed)", index, enabled, code);
}

// Stubbed memory
void *retro_get_memory_data(unsigned id) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Memory data: id=%u (stubbed)", id);
   else
      fallback_log_format("DEBUG", "Memory data: id=%u (stubbed)", id);
   return NULL;
}
size_t retro_get_memory_size(unsigned id) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] Memory size: id=%u (stubbed)", id);
   else
      fallback_log_format("DEBUG", "Memory size: id=%u (stubbed)", id);
   return 0;
}

// API version
unsigned retro_api_version(void) {
   if (log_cb)
      log_cb(RETRO_LOG_INFO, "[DEBUG] API version: %u", RETRO_API_VERSION);
   else
      fallback_log_format("DEBUG", "API version: %u", RETRO_API_VERSION);
   return RETRO_API_VERSION;
}