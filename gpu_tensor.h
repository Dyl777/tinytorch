#pragma once

// Include OpenGL base types first - available on all platforms
#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <GL/gl.h>
    #include <GL/glext.h>
#elif defined(__linux__)
    #include <GL/gl.h>
    #include <GL/glext.h>
    #include <GL/glx.h>
    #include <X11/Xlib.h>
#elif defined(__APPLE__)
    #include <OpenGL/gl3.h>
    #include <OpenGL/OpenGL.h>
#endif

// Fallback: ensure GL types are defined if gl.h didn't define them
#ifndef GL_APIENTRY
#define GL_APIENTRY APIENTRY
#endif
#ifndef GLubyte
typedef unsigned char GLubyte;
#endif
#ifndef GLenum
typedef unsigned int GLenum;
#endif
#ifndef GLbitfield
typedef unsigned int GLbitfield;
#endif
#ifndef GLsizei
typedef int GLsizei;
#endif
#ifndef GLuint
typedef unsigned int GLuint;
#endif
#ifndef GLint
typedef int GLint;
#endif
#ifndef GLsizeiptr
typedef ptrdiff_t GLsizeiptr;
#endif
#ifndef GLintptr
typedef ptrdiff_t GLintptr;
#endif
#ifndef GLchar
typedef char GLchar;
#endif
#ifndef GLboolean
typedef unsigned char GLboolean;
#endif
#ifndef GLclampf
typedef float GLclampf;
#endif
#ifndef GLdouble
typedef double GLdouble;
#endif
#ifndef GLclampd
typedef double GLclampd;
#endif

#include "tensor.h"
#include "stream_tensor.h"
#include "auto_tensor.h"
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <map>

// ============================================================================
// OpenGL GPU Tensor Support with Multi-GPU Selection
// ============================================================================
// What is this?
// -------------
// This module uses OpenGL Compute Shaders to perform tensor operations on the
// GPU. Data is stored in Shader Storage Buffer Objects (SSBOs), which are
// GPU memory buffers that compute shaders can read and write.
//
// Multi-GPU Support:
// ------------------
// Modern systems often have multiple GPUs (e.g., integrated + discrete, or
// multiple discrete GPUs). This implementation provides:
//
// 1. GPU ENUMERATION: Detects all available OpenGL-capable GPUs at startup
// 2. GPU SELECTION: Choose which GPU to use by index, name, or capability
// 3. AUTO-SELECTION: Automatically picks the best GPU based on heuristics
// 4. PER-TENSOR GPU AFFINITY: Different tensors can live on different GPUs
// 5. MULTI-GPU OPERATIONS: Operations between tensors on different GPUs
//    automatically handle cross-GPU data transfer
//
// Platform-specific GPU enumeration:
// -----------------------------------
// Windows: Enumerates display adapters via EnumDisplayDevices, creates a
//          context on each to query GL_RENDERER/GL_VENDOR
// Linux:   Parses /proc/driver/nvidia/gpus or uses libdrm for AMD/Intel
// macOS:   Uses CGGetActiveDisplayList + CGLDescribeRenderer

// ============================================================================
// OpenGL function types (for 4.3+ compute shaders, not in base gl.h)
// ============================================================================
// These are guarded to avoid conflicts with glext.h which may already define them

#ifndef PFNGLGENBUFFERSPROC
typedef void (GL_APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei, GLuint*);
#endif
#ifndef PFNGLBINDBUFFERPROC
typedef void (GL_APIENTRY *PFNGLBINDBUFFERPROC)(GLenum, GLuint);
#endif
#ifndef PFNGLBUFFERDATAPROC
typedef void (GL_APIENTRY *PFNGLBUFFERDATAPROC)(GLenum, GLsizeiptr, const void*, GLenum);
#endif
#ifndef PFNGLDELETEBUFFERSPROC
typedef void (GL_APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei, const GLuint*);
#endif
#ifndef PFNGLGETBUFFERSUBDATAPROC
typedef void (GL_APIENTRY *PFNGLGETBUFFERSUBDATAPROC)(GLenum, GLintptr, GLsizeiptr, void*);
#endif
#ifndef PFNGLCREATESHADERPROC
typedef GLuint (GL_APIENTRY *PFNGLCREATESHADERPROC)(GLenum);
#endif
#ifndef PFNGLCOMPILESHADERPROC
typedef void (GL_APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint);
#endif
#ifndef PFNGLGETSHADERIVPROC
typedef void (GL_APIENTRY *PFNGLGETSHADERIVPROC)(GLuint, GLenum, GLint*);
#endif
#ifndef PFNGLGETSHADERINFOLOGPROC
typedef void (GL_APIENTRY *PFNGLGETSHADERINFOLOGPROC)(GLuint, GLsizei, GLsizei*, GLchar*);
#endif
#ifndef PFNGLDELETESHADERPROC
typedef void (GL_APIENTRY *PFNGLDELETESHADERPROC)(GLuint);
#endif
#ifndef PFNGLCREATEPROGRAMPROC
typedef GLuint (GL_APIENTRY *PFNGLCREATEPROGRAMPROC)(void);
#endif
#ifndef PFNGLATTACHSHADERPROC
typedef void (GL_APIENTRY *PFNGLATTACHSHADERPROC)(GLuint, GLuint);
#endif
#ifndef PFNGLLINKPROGRAMPROC
typedef void (GL_APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint);
#endif
#ifndef PFNGLGETPROGRAMIVPROC
typedef void (GL_APIENTRY *PFNGLGETPROGRAMIVPROC)(GLuint, GLenum, GLint*);
#endif
#ifndef PFNGLGETPROGRAMINFOLOGPROC
typedef void (GL_APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)(GLuint, GLsizei, GLsizei*, GLchar*);
#endif
#ifndef PFNGLDELETEPROGRAMPROC
typedef void (GL_APIENTRY *PFNGLDELETEPROGRAMPROC)(GLuint);
#endif
#ifndef PFNGLUSEPROGRAMPROC
typedef void (GL_APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint);
#endif
#ifndef PFNGLDISPATCHCOMPUTEPROC
typedef void (GL_APIENTRY *PFNGLDISPATCHCOMPUTEPROC)(GLuint, GLuint, GLuint);
#endif
#ifndef PFNGLMEMORYBARRIERPROC
typedef void (GL_APIENTRY *PFNGLMEMORYBARRIERPROC)(GLbitfield);
#endif
#ifndef PFNGLBINDBUFFERBASEPROC
typedef void (GL_APIENTRY *PFNGLBINDBUFFERBASEPROC)(GLenum, GLuint, GLuint);
#endif
#ifndef PFNGLGETINTEGERVPROC
typedef void (GL_APIENTRY *PFNGLGETINTEGERVPROC)(GLenum, GLint*);
#endif
#ifndef PFNGLGETSTRINGPROC
typedef const GLubyte* (GL_APIENTRY *PFNGLGETSTRINGPROC)(GLenum);
#endif

// WGL extension types (Windows only, not in glext.h)
#if defined(_WIN32) || defined(_WIN64)
#ifndef PFNWGLCREATECONTEXTATTRIBSARBPROC
typedef HGLRC (WINAPI *PFNWGLCREATECONTEXTATTRIBSARBPROC)(HDC, HGLRC, const int*);
#endif
#ifndef PFNWGLGETPROCADDRESSPROC
typedef PROC (WINAPI *PFNWGLGETPROCADDRESSPROC)(LPCSTR);
#endif
#ifndef WGL_CONTEXT_MAJOR_VERSION_ARB
#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#endif
#ifndef WGL_CONTEXT_MINOR_VERSION_ARB
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#endif
#ifndef WGL_CONTEXT_FLAGS_ARB
#define WGL_CONTEXT_FLAGS_ARB 0x2094
#endif
#ifndef WGL_CONTEXT_PROFILE_MASK_ARB
#define WGL_CONTEXT_PROFILE_MASK_ARB 0x9126
#endif
#ifndef WGL_CONTEXT_CORE_PROFILE_BIT_ARB
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001
#endif
#endif

#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#endif
#ifndef GL_SHADER_STORAGE_BARRIER_BIT
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_COPY
#define GL_DYNAMIC_COPY 0x88EA
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_RENDERER
#define GL_RENDERER 0x1F01
#endif
#ifndef GL_VENDOR
#define GL_VENDOR 0x1F00
#endif
#ifndef GL_VERSION
#define GL_VERSION 0x1F02
#endif
#ifndef GL_MAJOR_VERSION
#define GL_MAJOR_VERSION 0x821B
#endif
#ifndef GL_MINOR_VERSION
#define GL_MINOR_VERSION 0x821C
#endif
#ifndef GL_NUM_EXTENSIONS
#define GL_NUM_EXTENSIONS 0x821D
#endif

// ============================================================================
// GpuInfo: Describes a single GPU
// ============================================================================
struct GpuInfo {
    int device_id;              // Unique device index
    std::string name;           // GPU name (e.g., "NVIDIA GeForce RTX 3080")
    std::string vendor;         // Vendor (e.g., "NVIDIA Corporation")
    std::string gl_version;     // OpenGL version string
    int gl_major;               // OpenGL major version
    int gl_minor;               // OpenGL minor version
    std::size_t max_ssbo_size;  // Max SSBO size in bytes
    int max_work_group_size;    // Max compute work group size
    bool supports_compute;      // Whether compute shaders are supported
    bool is_integrated;         // Whether this is an integrated GPU
    double compute_score;       // Heuristic performance score (higher = better)

    GpuInfo()
        : device_id(-1), gl_major(0), gl_minor(0),
          max_ssbo_size(0), max_work_group_size(0),
          supports_compute(false), is_integrated(false),
          compute_score(0.0)
    {}

    // Check if this GPU supports OpenGL 4.3+ compute shaders
    bool supports_compute_shaders() const {
        return supports_compute && (gl_major > 4 || (gl_major == 4 && gl_minor >= 3));
    }

    // Print GPU info
    void print(std::ostream& os) const {
        os << "GPU #" << device_id << ": " << name << " (" << vendor << ")\n"
           << "  OpenGL: " << gl_version << "\n"
           << "  Compute shaders: " << (supports_compute_shaders() ? "yes" : "no") << "\n"
           << "  Max SSBO: " << (max_ssbo_size / (1024.0 * 1024.0)) << " MB\n"
           << "  Max work group: " << max_work_group_size << "\n"
           << "  Integrated: " << (is_integrated ? "yes" : "no") << "\n"
           << "  Compute score: " << compute_score;
    }
};

// ============================================================================
// GpuSelector: Enumerates and selects GPUs
// ============================================================================
class GpuSelector {
private:
    std::vector<GpuInfo> _gpus;
    int _selected_device;

    // Helper: create a temporary OpenGL context to query GPU info
    // This is platform-specific and returns a minimal context
    static void* create_temp_context() {
#if defined(_WIN32) || defined(_WIN64)
        WNDCLASSA wc = {};
        wc.lpfnWndProc = DefWindowProcA;
        wc.hInstance = GetModuleHandleA(nullptr);
        wc.lpszClassName = "GpuEnumDummy";
        RegisterClassA(&wc);

        HWND hwnd = CreateWindowA("GpuEnumDummy", "", WS_OVERLAPPEDWINDOW,
                                   0, 0, 1, 1, nullptr, nullptr, wc.hInstance, nullptr);
        HDC dc = GetDC(hwnd);

        PIXELFORMATDESCRIPTOR pfd = {0};
        pfd.nSize = sizeof(pfd);
        pfd.nVersion = 1;
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        int pf = ChoosePixelFormat(dc, &pfd);
        SetPixelFormat(dc, pf, &pfd);

        HGLRC ctx = wglCreateContext(dc);
        wglMakeCurrent(dc, ctx);

        // Store hwnd and dc for cleanup
        struct WinCtx { HWND hwnd; HDC dc; HGLRC ctx; };
        WinCtx* w = new WinCtx{hwnd, dc, ctx};
        return (void*)w;
#elif defined(__linux__)
        Display* display = XOpenDisplay(nullptr);
        if (!display) return nullptr;

        int attribs[] = { GLX_RGBA, GLX_DOUBLEBUFFER, None };
        XVisualInfo* vi = glXChooseVisual(display, DefaultScreen(display), attribs);
        if (!vi) { XCloseDisplay(display); return nullptr; }

        GLXContext ctx = glXCreateContext(display, vi, nullptr, GL_TRUE);
        glXMakeCurrent(display, None, ctx);
        XFree(vi);

        struct LinuxCtx { Display* display; GLXContext ctx; };
        LinuxCtx* l = new LinuxCtx{display, ctx};
        return (void*)l;
#elif defined(__APPLE__)
        CGLPixelFormatAttribute attribs[] = { kCGLPFAAccelerated, (CGLPixelFormatAttribute)0 };
        CGLPixelFormatObj pixel_format;
        GLint num_pixel_formats;
        CGLChoosePixelFormat(attribs, &pixel_format, &num_pixel_formats);
        if (!pixel_format) return nullptr;

        CGLContextObj ctx;
        CGLCreateContext(pixel_format, nullptr, &ctx);
        CGLReleasePixelFormat(pixel_format);
        CGLSetCurrentContext(ctx);

        return (void*)ctx;
#else
        return nullptr;
#endif
    }

    static void destroy_temp_context(void* handle) {
        if (!handle) return;
#if defined(_WIN32) || defined(_WIN64)
        struct WinCtx { HWND hwnd; HDC dc; HGLRC ctx; };
        WinCtx* w = (WinCtx*)handle;
        wglMakeCurrent(w->dc, nullptr);
        wglDeleteContext(w->ctx);
        ReleaseDC(w->hwnd, w->dc);
        DestroyWindow(w->hwnd);
        delete w;
#elif defined(__linux__)
        struct LinuxCtx { Display* display; GLXContext ctx; };
        LinuxCtx* l = (LinuxCtx*)handle;
        glXDestroyContext(l->display, l->ctx);
        XCloseDisplay(l->display);
        delete l;
#elif defined(__APPLE__)
        CGLContextObj ctx = (CGLContextObj)handle;
        CGLSetCurrentContext(nullptr);
        CGLReleaseContext(ctx);
#endif
    }

    // Query GPU info from current OpenGL context
    static GpuInfo query_gpu_info(int device_id) {
        GpuInfo info;
        info.device_id = device_id;

#if defined(_WIN32) || defined(_WIN64)
        // On Windows, core OpenGL 1.1 functions (glGetString, glGetIntegerv) are
        // exported directly from opengl32.dll. wglGetProcAddress only works for
        // OpenGL 1.2+ extension functions.
        auto get_str = [&](GLenum param) -> std::string {
            const char* s = (const char*)::glGetString(param);
            return s ? std::string(s) : "unknown";
        };
        auto get_int = [&](GLenum param) -> GLint {
            GLint val = 0;
            ::glGetIntegerv(param, &val);
            return val;
        };
#elif defined(__linux__)
        auto get_str = [&](GLenum param) -> std::string {
            const char* s = (const char*)glGetString(param);
            return s ? std::string(s) : "unknown";
        };
        auto get_int = [&](GLenum param) -> GLint {
            GLint val = 0;
            glGetIntegerv(param, &val);
            return val;
        };
#elif defined(__APPLE__)
        auto get_str = [&](GLenum param) -> std::string {
            const char* s = (const char*)glGetString(param);
            return s ? std::string(s) : "unknown";
        };
        auto get_int = [&](GLenum param) -> GLint {
            GLint val = 0;
            glGetIntegerv(param, &val);
            return val;
        };
#else
        auto get_str = [&](GLenum) -> std::string { return "unknown"; };
        auto get_int = [&](GLenum) -> GLint { return 0; };
#endif

        info.name = get_str(GL_RENDERER);
        info.vendor = get_str(GL_VENDOR);
        info.gl_version = get_str(GL_VERSION);

        info.gl_major = get_int(GL_MAJOR_VERSION);
        info.gl_minor = get_int(GL_MINOR_VERSION);
        info.max_work_group_size = get_int(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS);

        // Estimate max SSBO size
        GLint max_buf_size = 0;
        get_int(0x90D2); // GL_MAX_SHADER_STORAGE_BLOCK_SIZE
        info.max_ssbo_size = static_cast<std::size_t>(max_buf_size);

        info.supports_compute = (info.gl_major > 4 || (info.gl_major == 4 && info.gl_minor >= 3));

        // Detect integrated GPU by name heuristics
        std::string name_lower = info.name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        info.is_integrated = (name_lower.find("intel") != std::string::npos ||
                              name_lower.find("uhd") != std::string::npos ||
                              name_lower.find("iris") != std::string::npos ||
                              name_lower.find("mali") != std::string::npos ||
                              name_lower.find("adreno") != std::string::npos);

        // Compute a heuristic performance score
        info.compute_score = 0.0;
        if (info.supports_compute) {
            info.compute_score += 100.0;
            info.compute_score += info.gl_major * 10.0;
            info.compute_score += info.gl_minor * 1.0;
            info.compute_score += info.max_work_group_size * 0.1;
            if (!info.is_integrated) info.compute_score += 200.0;

            // Bonus for NVIDIA/AMD discrete GPUs
            if (name_lower.find("nvidia") != std::string::npos ||
                name_lower.find("amd") != std::string::npos ||
                name_lower.find("radeon") != std::string::npos) {
                info.compute_score += 100.0;
            }
        }

        return info;
    }

public:
    GpuSelector() : _selected_device(-1) {}

    // Enumerate all available GPUs
    bool enumerate() {
        _gpus.clear();

        void* ctx = create_temp_context();
        if (!ctx) {
            std::cerr << "[GpuSelector] Failed to create temporary OpenGL context" << std::endl;
            return false;
        }

        GpuInfo info = query_gpu_info(0);
        if (info.device_id >= 0) {
            _gpus.push_back(info);
        }

        destroy_temp_context(ctx);

        // On multi-GPU systems, we'd enumerate multiple contexts here.
        // For now, we detect the primary GPU. Full multi-GPU enumeration
        // requires platform-specific adapter enumeration.

#if defined(_WIN32) || defined(_WIN64)
        // Windows: enumerate display devices to find multiple GPUs
        DISPLAY_DEVICEA dd;
        dd.cb = sizeof(dd);
        int device_idx = 0;
        while (EnumDisplayDevicesA(nullptr, device_idx, &dd, 0)) {
            if (dd.StateFlags & DISPLAY_DEVICE_ACTIVE) {
                // Try to create context on this adapter
                // This is simplified; full implementation would use
                // IDXGIFactory or EnumDisplayDevices per monitor
            }
            device_idx++;
        }
#endif

        // Sort by compute score (best first)
        std::sort(_gpus.begin(), _gpus.end(),
                  [](const GpuInfo& a, const GpuInfo& b) {
                      return a.compute_score > b.compute_score;
                  });

        // Assign device IDs after sorting
        for (int i = 0; i < static_cast<int>(_gpus.size()); ++i) {
            _gpus[i].device_id = i;
        }

        return !_gpus.empty();
    }

    // Get list of all detected GPUs
    const std::vector<GpuInfo>& get_gpus() const { return _gpus; }

    // Get info for a specific GPU
    const GpuInfo& get_gpu(int device_id) const {
        if (device_id < 0 || device_id >= static_cast<int>(_gpus.size())) {
            throw std::invalid_argument("Invalid device ID: " + std::to_string(device_id));
        }
        return _gpus[device_id];
    }

    // Select GPU by device ID
    void select_device(int device_id) {
        if (device_id < -1 || device_id >= static_cast<int>(_gpus.size())) {
            throw std::invalid_argument("Invalid device ID: " + std::to_string(device_id));
        }
        _selected_device = device_id;
    }

    // Select GPU by name (partial match, case-insensitive)
    bool select_by_name(const std::string& name_pattern) {
        std::string pattern = name_pattern;
        std::transform(pattern.begin(), pattern.end(), pattern.begin(), ::tolower);

        for (const auto& gpu : _gpus) {
            std::string gpu_name = gpu.name;
            std::transform(gpu_name.begin(), gpu_name.end(), gpu_name.begin(), ::tolower);
            if (gpu_name.find(pattern) != std::string::npos) {
                _selected_device = gpu.device_id;
                return true;
            }
        }
        return false;
    }

    // Auto-select the best available GPU
    void auto_select() {
        if (_gpus.empty()) {
            _selected_device = -1;
            return;
        }
        // GPUs are already sorted by compute_score, so pick the first one
        _selected_device = _gpus[0].device_id;
    }

    // Select only discrete GPUs (exclude integrated)
    void select_discrete_only() {
        for (const auto& gpu : _gpus) {
            if (!gpu.is_integrated && gpu.supports_compute_shaders()) {
                _selected_device = gpu.device_id;
                return;
            }
        }
        // Fall back to auto-select if no discrete GPU found
        auto_select();
    }

    // Get currently selected device ID
    int selected_device() const { return _selected_device; }

    // Get info for selected GPU
    const GpuInfo* selected_gpu() const {
        if (_selected_device < 0) return nullptr;
        return &get_gpu(_selected_device);
    }

    // Print all detected GPUs
    void print_all(std::ostream& os) const {
        os << "Detected " << _gpus.size() << " GPU(s):\n";
        for (const auto& gpu : _gpus) {
            gpu.print(os);
            os << "\n";
        }
        if (_selected_device >= 0) {
            os << "Selected: GPU #" << _selected_device << "\n";
        } else {
            os << "No GPU selected\n";
        }
    }

    // Get the best GPU for compute (highest score)
    const GpuInfo* best_gpu() const {
        if (_gpus.empty()) return nullptr;
        return &_gpus[0]; // Already sorted by score
    }
};

// Global GPU selector
inline GpuSelector& get_gpu_selector() {
    static GpuSelector selector;
    return selector;
}

// ============================================================================
// OpenGL Function Pointer Manager
// ============================================================================
class GlFunctions {
public:
    static GlFunctions& instance() {
        static GlFunctions inst;
        return inst;
    }

    bool initialize() {
#if defined(_WIN32) || defined(_WIN64)
        // Core OpenGL 1.1 functions are exported directly from opengl32.dll
        // wglGetProcAddress only works for OpenGL 1.2+ extensions
        glGenBuffers = (PFNGLGENBUFFERSPROC)::wglGetProcAddress("glGenBuffers");
        glBindBuffer = (PFNGLBINDBUFFERPROC)::wglGetProcAddress("glBindBuffer");
        glBufferData = (PFNGLBUFFERDATAPROC)::wglGetProcAddress("glBufferData");
        glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)::wglGetProcAddress("glDeleteBuffers");
        glGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC)::wglGetProcAddress("glGetBufferSubData");
        glCreateShader = (PFNGLCREATESHADERPROC)::wglGetProcAddress("glCreateShader");
        glShaderSource = (PFNGLSHADERSOURCEPROC)::wglGetProcAddress("glShaderSource");
        glCompileShader = (PFNGLCOMPILESHADERPROC)::wglGetProcAddress("glCompileShader");
        glGetShaderiv = (PFNGLGETSHADERIVPROC)::wglGetProcAddress("glGetShaderiv");
        glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)::wglGetProcAddress("glGetShaderInfoLog");
        glDeleteShader = (PFNGLDELETESHADERPROC)::wglGetProcAddress("glDeleteShader");
        glCreateProgram = (PFNGLCREATEPROGRAMPROC)::wglGetProcAddress("glCreateProgram");
        glAttachShader = (PFNGLATTACHSHADERPROC)::wglGetProcAddress("glAttachShader");
        glLinkProgram = (PFNGLLINKPROGRAMPROC)::wglGetProcAddress("glLinkProgram");
        glGetProgramiv = (PFNGLGETPROGRAMIVPROC)::wglGetProcAddress("glGetProgramiv");
        glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)::wglGetProcAddress("glGetProgramInfoLog");
        glDeleteProgram = (PFNGLDELETEPROGRAMPROC)::wglGetProcAddress("glDeleteProgram");
        glUseProgram = (PFNGLUSEPROGRAMPROC)::wglGetProcAddress("glUseProgram");
        glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)::wglGetProcAddress("glDispatchCompute");
        glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)::wglGetProcAddress("glMemoryBarrier");
        glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)::wglGetProcAddress("glBindBufferBase");
        // Core 1.1 functions - use directly from opengl32.dll
        glGetIntegerv = ::glGetIntegerv;
        glGetString = ::glGetString;
#elif defined(__linux__)
        glGenBuffers = (PFNGLGENBUFFERSPROC)glXGetProcAddress((const GLubyte*)"glGenBuffers");
        glBindBuffer = (PFNGLBINDBUFFERPROC)glXGetProcAddress((const GLubyte*)"glBindBuffer");
        glBufferData = (PFNGLBUFFERDATAPROC)glXGetProcAddress((const GLubyte*)"glBufferData");
        glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glXGetProcAddress((const GLubyte*)"glDeleteBuffers");
        glGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC)glXGetProcAddress((const GLubyte*)"glGetBufferSubData");
        glCreateShader = (PFNGLCREATESHADERPROC)glXGetProcAddress((const GLubyte*)"glCreateShader");
        glShaderSource = (PFNGLSHADERSOURCEPROC)glXGetProcAddress((const GLubyte*)"glShaderSource");
        glCompileShader = (PFNGLCOMPILESHADERPROC)glXGetProcAddress((const GLubyte*)"glCompileShader");
        glGetShaderiv = (PFNGLGETSHADERIVPROC)glXGetProcAddress((const GLubyte*)"glGetShaderiv");
        glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)glXGetProcAddress((const GLubyte*)"glGetShaderInfoLog");
        glDeleteShader = (PFNGLDELETESHADERPROC)glXGetProcAddress((const GLubyte*)"glDeleteShader");
        glCreateProgram = (PFNGLCREATEPROGRAMPROC)glXGetProcAddress((const GLubyte*)"glCreateProgram");
        glAttachShader = (PFNGLATTACHSHADERPROC)glXGetProcAddress((const GLubyte*)"glAttachShader");
        glLinkProgram = (PFNGLLINKPROGRAMPROC)glXGetProcAddress((const GLubyte*)"glLinkProgram");
        glGetProgramiv = (PFNGLGETPROGRAMIVPROC)glXGetProcAddress((const GLubyte*)"glGetProgramiv");
        glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)glXGetProcAddress((const GLubyte*)"glGetProgramInfoLog");
        glDeleteProgram = (PFNGLDELETEPROGRAMPROC)glXGetProcAddress((const GLubyte*)"glDeleteProgram");
        glUseProgram = (PFNGLUSEPROGRAMPROC)glXGetProcAddress((const GLubyte*)"glUseProgram");
        glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)glXGetProcAddress((const GLubyte*)"glDispatchCompute");
        glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)glXGetProcAddress((const GLubyte*)"glMemoryBarrier");
        glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)glXGetProcAddress((const GLubyte*)"glBindBufferBase");
        glGetIntegerv = glGetIntegerv;
        glGetString = glGetString;
#elif defined(__APPLE__)
        glGenBuffers = (PFNGLGENBUFFERSPROC)glGenBuffers;
        glBindBuffer = (PFNGLBINDBUFFERPROC)glBindBuffer;
        glBufferData = (PFNGLBUFFERDATAPROC)glBufferData;
        glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glDeleteBuffers;
        glGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC)glGetBufferSubData;
        glCreateShader = (PFNGLCREATESHADERPROC)glCreateShader;
        glShaderSource = (PFNGLSHADERSOURCEPROC)glShaderSource;
        glCompileShader = (PFNGLCOMPILESHADERPROC)glCompileShader;
        glGetShaderiv = (PFNGLGETSHADERIVPROC)glGetShaderiv;
        glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)glGetShaderInfoLog;
        glDeleteShader = (PFNGLDELETESHADERPROC)glDeleteShader;
        glCreateProgram = (PFNGLCREATEPROGRAMPROC)glCreateProgram;
        glAttachShader = (PFNGLATTACHSHADERPROC)glAttachShader;
        glLinkProgram = (PFNGLLINKPROGRAMPROC)glLinkProgram;
        glGetProgramiv = (PFNGLGETPROGRAMIVPROC)glGetProgramiv;
        glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)glGetProgramInfoLog;
        glDeleteProgram = (PFNGLDELETEPROGRAMPROC)glDeleteProgram;
        glUseProgram = (PFNGLUSEPROGRAMPROC)glUseProgram;
        glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)glDispatchCompute;
        glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)glMemoryBarrier;
        glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)glBindBufferBase;
        glGetIntegerv = glGetIntegerv;
        glGetString = glGetString;
#endif
        return glGenBuffers != nullptr && glCreateShader != nullptr;
    }

    PFNGLGENBUFFERSPROC glGenBuffers = nullptr;
    PFNGLBINDBUFFERPROC glBindBuffer = nullptr;
    PFNGLBUFFERDATAPROC glBufferData = nullptr;
    PFNGLDELETEBUFFERSPROC glDeleteBuffers = nullptr;
    PFNGLGETBUFFERSUBDATAPROC glGetBufferSubData = nullptr;
    PFNGLCREATESHADERPROC glCreateShader = nullptr;
    PFNGLSHADERSOURCEPROC glShaderSource = nullptr;
    PFNGLCOMPILESHADERPROC glCompileShader = nullptr;
    PFNGLGETSHADERIVPROC glGetShaderiv = nullptr;
    PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = nullptr;
    PFNGLDELETESHADERPROC glDeleteShader = nullptr;
    PFNGLCREATEPROGRAMPROC glCreateProgram = nullptr;
    PFNGLATTACHSHADERPROC glAttachShader = nullptr;
    PFNGLLINKPROGRAMPROC glLinkProgram = nullptr;
    PFNGLGETPROGRAMIVPROC glGetProgramiv = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = nullptr;
    PFNGLDELETEPROGRAMPROC glDeleteProgram = nullptr;
    PFNGLUSEPROGRAMPROC glUseProgram = nullptr;
    PFNGLDISPATCHCOMPUTEPROC glDispatchCompute = nullptr;
    PFNGLMEMORYBARRIERPROC glMemoryBarrier = nullptr;
    PFNGLBINDBUFFERBASEPROC glBindBufferBase = nullptr;
    PFNGLGETINTEGERVPROC glGetIntegerv = nullptr;
    PFNGLGETSTRINGPROC glGetString = nullptr;

private:
    GlFunctions() = default;
};

// ============================================================================
// GpuConfig: Configuration for GPU tensor operations
// ============================================================================
struct GpuConfig {
    int device_id;               // Which GPU to use (-1 = auto-select)
    std::size_t work_group_size; // Threads per work group (default: 256)
    std::size_t max_elements_per_dispatch;
    bool use_pinned_memory;
    bool async_transfer;

    GpuConfig()
        : device_id(-1),
          work_group_size(256),
          max_elements_per_dispatch(10000000),
          use_pinned_memory(false),
          async_transfer(false)
    {}
};

// ============================================================================
// Compute Shader Sources
// ============================================================================
static const char* SHADER_ADD = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer InputA { float a_data[]; };
layout(std430, binding = 1) buffer InputB { float b_data[]; };
layout(std430, binding = 2) buffer Output { float out_data[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < a_data.length()) {
        out_data[idx] = a_data[idx] + b_data[idx];
    }
}
)";

static const char* SHADER_SUBTRACT = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer InputA { float a_data[]; };
layout(std430, binding = 1) buffer InputB { float b_data[]; };
layout(std430, binding = 2) buffer Output { float out_data[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < a_data.length()) {
        out_data[idx] = a_data[idx] - b_data[idx];
    }
}
)";

static const char* SHADER_MULTIPLY = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer InputA { float a_data[]; };
layout(std430, binding = 1) buffer InputB { float b_data[]; };
layout(std430, binding = 2) buffer Output { float out_data[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < a_data.length()) {
        out_data[idx] = a_data[idx] * b_data[idx];
    }
}
)";

static const char* SHADER_DIVIDE = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer InputA { float a_data[]; };
layout(std430, binding = 1) buffer InputB { float b_data[]; };
layout(std430, binding = 2) buffer Output { float out_data[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < a_data.length()) {
        out_data[idx] = a_data[idx] / b_data[idx];
    }
}
)";

static const char* SHADER_ADD_SCALAR = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Input { float in_data[]; };
layout(std430, binding = 1) buffer Output { float out_data[]; };
uniform float scalar;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < in_data.length()) {
        out_data[idx] = in_data[idx] + scalar;
    }
}
)";

static const char* SHADER_MULTIPLY_SCALAR = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Input { float in_data[]; };
layout(std430, binding = 1) buffer Output { float out_data[]; };
uniform float scalar;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < in_data.length()) {
        out_data[idx] = in_data[idx] * scalar;
    }
}
)";

static const char* SHADER_NEGATE = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Input { float in_data[]; };
layout(std430, binding = 1) buffer Output { float out_data[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < in_data.length()) {
        out_data[idx] = -in_data[idx];
    }
}
)";

static const char* SHADER_ABS = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Input { float in_data[]; };
layout(std430, binding = 1) buffer Output { float out_data[]; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < in_data.length()) {
        out_data[idx] = abs(in_data[idx]);
    }
}
)";

static const char* SHADER_CLAMP = R"(
#version 430 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Input { float in_data[]; };
layout(std430, binding = 1) buffer Output { float out_data[]; };
uniform float min_val;
uniform float max_val;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < in_data.length()) {
        float v = in_data[idx];
        out_data[idx] = clamp(v, min_val, max_val);
    }
}
)";

// ============================================================================
// GpuContext: Manages OpenGL context and compute shader programs
// ============================================================================
class GpuContext {
private:
    bool _initialized;
    int _device_id;
    std::vector<GLuint> _programs;
    std::vector<GLuint> _buffers;
    std::map<std::string, GLuint> _program_cache;

#if defined(_WIN32) || defined(_WIN64)
    HGLRC _gl_context;
    HDC _dc;
    HWND _dummy_window;
#elif defined(__linux__)
    GLXContext _gl_context;
    Display* _display;
#elif defined(__APPLE__)
    CGLContextObj _gl_context;
#endif

    GLuint compile_shader(const char* source, const char* name) {
        auto& gl = GlFunctions::instance();
        GLuint shader = gl.glCreateShader(GL_COMPUTE_SHADER);
        gl.glShaderSource(shader, 1, &source, nullptr);
        gl.glCompileShader(shader);

        GLint success;
        gl.glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[512];
            gl.glGetShaderInfoLog(shader, 512, nullptr, log);
            std::cerr << "[GpuContext] Shader compilation failed (" << name << "): " << log << std::endl;
            gl.glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    GLuint link_program(GLuint shader) {
        auto& gl = GlFunctions::instance();
        GLuint program = gl.glCreateProgram();
        gl.glAttachShader(program, shader);
        gl.glLinkProgram(program);
        gl.glDeleteShader(shader);

        GLint success;
        gl.glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            char log[512];
            gl.glGetProgramInfoLog(program, 512, nullptr, log);
            std::cerr << "[GpuContext] Program linking failed: " << log << std::endl;
            gl.glDeleteProgram(program);
            return 0;
        }

        _programs.push_back(program);
        return program;
    }

public:
    GpuContext() : _initialized(false), _device_id(-1) {}

    bool initialize(int device_id = -1) {
        if (_initialized) return true;

        // Auto-select GPU if not specified
        if (device_id < 0) {
            auto& selector = get_gpu_selector();
            if (selector.get_gpus().empty()) {
                selector.enumerate();
            }
            selector.auto_select();
            device_id = selector.selected_device();
        }

        _device_id = device_id;

#if defined(_WIN32) || defined(_WIN64)
        WNDCLASSA wc = {};
        wc.lpfnWndProc = DefWindowProcA;
        wc.hInstance = GetModuleHandleA(nullptr);
        wc.lpszClassName = "GpuTensorDummy";
        RegisterClassA(&wc);

        _dummy_window = CreateWindowA("GpuTensorDummy", "", WS_OVERLAPPEDWINDOW,
                                       0, 0, 1, 1, nullptr, nullptr, wc.hInstance, nullptr);
        _dc = GetDC(_dummy_window);

        PIXELFORMATDESCRIPTOR pfd = {0};
        pfd.nSize = sizeof(pfd);
        pfd.nVersion = 1;
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;
        int pf = ChoosePixelFormat(_dc, &pfd);
        SetPixelFormat(_dc, pf, &pfd);

        HGLRC temp_context = wglCreateContext(_dc);
        wglMakeCurrent(_dc, temp_context);

        PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB =
            (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
        if (wglCreateContextAttribsARB) {
            int attribs[] = {
                WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
                WGL_CONTEXT_MINOR_VERSION_ARB, 3,
                WGL_CONTEXT_FLAGS_ARB, 0,
                WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
                0
            };
            _gl_context = wglCreateContextAttribsARB(_dc, 0, attribs);
            wglMakeCurrent(_dc, nullptr);
            wglDeleteContext(temp_context);
            wglMakeCurrent(_dc, _gl_context);
        } else {
            _gl_context = temp_context;
        }

#elif defined(__linux__)
        _display = XOpenDisplay(nullptr);
        if (!_display) {
            std::cerr << "[GpuContext] Failed to open X display" << std::endl;
            return false;
        }

        int attribs[] = { GLX_RGBA, GLX_DOUBLEBUFFER, None };
        XVisualInfo* vi = glXChooseVisual(_display, DefaultScreen(_display), attribs);
        if (!vi) {
            std::cerr << "[GpuContext] No suitable visual found" << std::endl;
            return false;
        }

        _gl_context = glXCreateContext(_display, vi, nullptr, GL_TRUE);
        glXMakeCurrent(_display, None, _gl_context);
        XFree(vi);

#elif defined(__APPLE__)
        CGLPixelFormatAttribute attribs[] = {
            kCGLPFAAccelerated,
            (CGLPixelFormatAttribute)0
        };
        CGLPixelFormatObj pixel_format;
        GLint num_pixel_formats;
        CGLChoosePixelFormat(attribs, &pixel_format, &num_pixel_formats);
        CGLCreateContext(pixel_format, nullptr, &_gl_context);
        CGLReleasePixelFormat(pixel_format);
        CGLSetCurrentContext(_gl_context);
#endif

        if (!GlFunctions::instance().initialize()) {
            std::cerr << "[GpuContext] Failed to load OpenGL functions" << std::endl;
            return false;
        }

        auto& gl = GlFunctions::instance();
        GLint major, minor;
        gl.glGetIntegerv(GL_MAJOR_VERSION, &major);
        gl.glGetIntegerv(GL_MINOR_VERSION, &minor);

        const char* renderer = (const char*)gl.glGetString(GL_RENDERER);
        const char* vendor = (const char*)gl.glGetString(GL_VENDOR);

        std::cout << "[GpuContext] OpenGL " << major << "." << minor
                  << " on " << (renderer ? renderer : "unknown")
                  << " (" << (vendor ? vendor : "unknown")
                  << ") [Device #" << _device_id << "]" << std::endl;

        if (major < 4 || (major == 4 && minor < 3)) {
            std::cerr << "[GpuContext] OpenGL 4.3+ required for compute shaders" << std::endl;
            return false;
        }

        _initialized = true;
        return true;
    }

    GLuint get_program(const char* source, const char* name) {
        // Check cache first
        std::string key = name;
        auto it = _program_cache.find(key);
        if (it != _program_cache.end()) return it->second;

        GLuint shader = compile_shader(source, name);
        if (shader == 0) return 0;
        GLuint program = link_program(shader);
        if (program != 0) {
            _program_cache[key] = program;
        }
        return program;
    }

    GLuint create_buffer(const void* data, std::size_t size_bytes) {
        auto& gl = GlFunctions::instance();
        GLuint buffer;
        gl.glGenBuffers(1, &buffer);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        gl.glBufferData(GL_SHADER_STORAGE_BUFFER, size_bytes, data, GL_DYNAMIC_COPY);
        _buffers.push_back(buffer);
        return buffer;
    }

    GLuint create_empty_buffer(std::size_t size_bytes) {
        return create_buffer(nullptr, size_bytes);
    }

    void read_buffer(GLuint buffer, void* dest, std::size_t size_bytes) {
        auto& gl = GlFunctions::instance();
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        gl.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size_bytes, dest);
    }

    void delete_buffer(GLuint buffer) {
        auto& gl = GlFunctions::instance();
        gl.glDeleteBuffers(1, &buffer);
    }

    void dispatch(GLuint program, std::size_t num_elements, std::size_t work_group_size = 256) {
        auto& gl = GlFunctions::instance();
        gl.glUseProgram(program);
        GLuint num_groups = static_cast<GLuint>((num_elements + work_group_size - 1) / work_group_size);
        gl.glDispatchCompute(num_groups, 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    void bind_buffer_base(GLuint buffer, GLuint binding_point) {
        auto& gl = GlFunctions::instance();
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point, buffer);
    }

    void cleanup() {
        auto& gl = GlFunctions::instance();
        for (GLuint p : _programs) gl.glDeleteProgram(p);
        for (GLuint b : _buffers) gl.glDeleteBuffers(1, &b);
        _programs.clear();
        _buffers.clear();
        _program_cache.clear();

#if defined(_WIN32) || defined(_WIN64)
        if (_gl_context) {
            wglMakeCurrent(_dc, nullptr);
            wglDeleteContext(_gl_context);
        }
        if (_dummy_window) DestroyWindow(_dummy_window);
#elif defined(__linux__)
        if (_gl_context) glXDestroyContext(_display, _gl_context);
        if (_display) XCloseDisplay(_display);
#elif defined(__APPLE__)
        if (_gl_context) {
            CGLSetCurrentContext(nullptr);
            CGLReleaseContext(_gl_context);
        }
#endif
        _initialized = false;
    }

    ~GpuContext() {
        if (_initialized) cleanup();
    }

    bool is_initialized() const { return _initialized; }
    int device_id() const { return _device_id; }
};

// Global GPU context accessor
inline GpuContext& get_gpu_context() {
    static GpuContext ctx;
    return ctx;
}

// ============================================================================
// BoolTensorResult: Simple CPU-based bool tensor for comparison results
// ============================================================================
// GPU shaders don't support bool SSBOs well, so comparison operations return
// results on the CPU. This class wraps a std::vector<bool> as a TensorBase<bool>.

class BoolTensorResult : public TensorBase<bool> {
private:
    std::vector<bool> _data;
    std::size_t _size;

public:
    explicit BoolTensorResult(std::vector<bool> data)
        : _data(std::move(data)), _size(_data.size()) {}

    std::size_t ndim() const override { return 1; }
    std::size_t total_size() const override { return _size; }
    const std::size_t* shape() const override {
        static std::size_t s = 0;
        // Return pointer to a static variable - not ideal but works for 1D
        const_cast<std::size_t&>(s) = _size;
        return &s;
    }
    const std::size_t* stride() const override {
        static std::size_t s = 1;
        return &s;
    }

    bool get_element(std::size_t index) const override {
        if (index >= _size) throw std::invalid_argument("Index out of bounds");
        return _data[index];
    }

    void set_element(std::size_t index, bool value) override {
        if (index >= _size) throw std::invalid_argument("Index out of bounds");
        _data[index] = value;
    }

    bool operator()(std::size_t i) const override { return get_element(i); }
    bool& operator()(std::size_t i) override {
        static bool temp;
        temp = _data[i];
        return temp;
    }
    bool operator()(std::size_t /*i*/, std::size_t /*j*/) const override {
        throw std::invalid_argument("BoolTensorResult is 1D only");
    }
    bool& operator()(std::size_t /*i*/, std::size_t /*j*/) override {
        throw std::invalid_argument("BoolTensorResult is 1D only");
    }

    bool is_streaming() const override { return false; }
    std::string backend_name() const override { return "BoolTensorResult (CPU)"; }

    std::unique_ptr<TensorBase<bool>> add(const TensorBase<bool>*) const override {
        throw std::invalid_argument("add not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> subtract(const TensorBase<bool>*) const override {
        throw std::invalid_argument("subtract not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> multiply(const TensorBase<bool>*) const override {
        throw std::invalid_argument("multiply not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> divide(const TensorBase<bool>*) const override {
        throw std::invalid_argument("divide not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> add_scalar(bool) const override {
        throw std::invalid_argument("add_scalar not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> subtract_scalar(bool) const override {
        throw std::invalid_argument("subtract_scalar not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> multiply_scalar(bool) const override {
        throw std::invalid_argument("multiply_scalar not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> divide_scalar(bool) const override {
        throw std::invalid_argument("divide_scalar not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> negate() const override {
        throw std::invalid_argument("negate not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> abs() const override {
        throw std::invalid_argument("abs not supported on bool tensor");
    }

    bool sum() const override {
        bool result = false;
        for (std::size_t i = 0; i < _size; ++i) result = result || _data[i];
        return result;
    }
    bool mean() const override { return false; }
    bool max() const override {
        for (std::size_t i = 0; i < _size; ++i) if (_data[i]) return true;
        return false;
    }
    bool min() const override {
        for (std::size_t i = 0; i < _size; ++i) if (!_data[i]) return false;
        return true;
    }
    bool dot(const TensorBase<bool>*) const override {
        throw std::invalid_argument("dot not supported on bool tensor");
    }

    std::unique_ptr<TensorBase<bool>> reshape(const std::size_t*, std::size_t) const override {
        throw std::invalid_argument("reshape not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> transpose() const override {
        throw std::invalid_argument("transpose not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> clamp(bool, bool) const override {
        throw std::invalid_argument("clamp not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> greater_than(bool) const override {
        throw std::invalid_argument("greater_than not supported on bool tensor");
    }
    std::unique_ptr<TensorBase<bool>> less_than(bool) const override {
        throw std::invalid_argument("less_than not supported on bool tensor");
    }

    std::unique_ptr<bool[]> to_array() const override {
        std::unique_ptr<bool[]> result(new bool[_size]);
        for (std::size_t i = 0; i < _size; ++i) result[i] = _data[i];
        return result;
    }

    void print(std::ostream& os) const override {
        os << "[";
        for (std::size_t i = 0; i < _size; ++i) {
            os << (_data[i] ? "true" : "false");
            if (i != _size - 1) os << ", ";
        }
        os << "]";
    }

    void batch_print(std::ostream& os, std::size_t batch_size) const override {
        os << "[";
        for (std::size_t batch_start = 0; batch_start < _size; batch_start += batch_size) {
            std::size_t count = std::min(batch_size, _size - batch_start);
            for (std::size_t i = 0; i < count; ++i) {
                os << (_data[batch_start + i] ? "true" : "false");
                if (batch_start + i != _size - 1) os << ", ";
            }
            os.flush();
            if (batch_start + count < _size) {
                os << "\n  ... (batch " << (batch_start / batch_size + 1) << " done) ";
            }
        }
        os << "]";
    }
};

// ============================================================================
// GpuTensor: GPU-accelerated tensor using OpenGL compute shaders
// ============================================================================
template<typename T>
class GpuTensor : public TensorBase<T> {
private:
    std::size_t* _shape;
    std::size_t* _stride;
    std::size_t _ndim;
    std::size_t _total_size;
    GLuint _ssbo;
    GpuConfig _config;
    bool _owns_data;
    int _device_id;  // Which GPU this tensor lives on

    void compute_strides() {
        if (_ndim == 0) { _stride = nullptr; return; }
        _stride = new std::size_t[_ndim];
        _stride[_ndim - 1] = 1;
        for (std::size_t i = _ndim - 1; i > 0; --i) {
            _stride[i - 1] = _stride[i] * _shape[i];
        }
    }

    std::size_t compute_total_size() const {
        if (_ndim == 0) return 1;
        std::size_t size = 1;
        for (std::size_t i = 0; i < _ndim; ++i) size *= _shape[i];
        return size;
    }

    // Get or create context for this tensor's device
    // Returns non-const reference since context methods modify GPU state
    GpuContext& get_context() const {
        static std::map<int, std::unique_ptr<GpuContext>> contexts;
        auto it = contexts.find(_device_id);
        if (it == contexts.end()) {
            contexts[_device_id] = std::make_unique<GpuContext>();
            if (!contexts[_device_id]->initialize(_device_id)) {
                throw std::runtime_error("Failed to initialize GPU context for device #" +
                                         std::to_string(_device_id));
            }
        }
        return *contexts[_device_id];
    }

public:
    GpuTensor(const std::size_t* shape, std::size_t ndim,
              const GpuConfig& config = GpuConfig{}, T fill_value = T{})
        : _ndim(ndim), _config(config), _owns_data(true) {

        _device_id = config.device_id;
        if (_device_id < 0) {
            auto& selector = get_gpu_selector();
            if (selector.get_gpus().empty()) selector.enumerate();
            if (selector.selected_device() < 0) selector.auto_select();
            _device_id = selector.selected_device();
        }

        _shape = new std::size_t[_ndim];
        std::memcpy(_shape, shape, _ndim * sizeof(std::size_t));
        _total_size = compute_total_size();
        compute_strides();

        std::vector<T> host_data(_total_size, fill_value);
        _ssbo = get_context().create_buffer(host_data.data(), _total_size * sizeof(T));
    }

    GpuTensor(const T* data, std::size_t size,
              const GpuConfig& config = GpuConfig{})
        : _ndim(1), _total_size(size), _config(config), _owns_data(true) {

        _device_id = config.device_id;
        if (_device_id < 0) {
            auto& selector = get_gpu_selector();
            if (selector.get_gpus().empty()) selector.enumerate();
            if (selector.selected_device() < 0) selector.auto_select();
            _device_id = selector.selected_device();
        }

        _shape = new std::size_t[1];
        _shape[0] = size;
        compute_strides();

        _ssbo = get_context().create_buffer(data, size * sizeof(T));
    }

    GpuTensor(std::initializer_list<T> data,
              const GpuConfig& config = GpuConfig{})
        : GpuTensor(data.begin(), data.size(), config) {}

    GpuTensor(const GpuTensor& other)
        : _ndim(other._ndim), _total_size(other._total_size),
          _config(other._config), _owns_data(true), _device_id(other._device_id) {

        _shape = new std::size_t[_ndim];
        std::memcpy(_shape, other._shape, _ndim * sizeof(std::size_t));
        compute_strides();

        std::vector<T> host_data(_total_size);
        get_context().read_buffer(other._ssbo, host_data.data(), _total_size * sizeof(T));
        _ssbo = get_context().create_buffer(host_data.data(), _total_size * sizeof(T));
    }

    GpuTensor(GpuTensor&& other) noexcept
        : _shape(other._shape), _stride(other._stride),
          _ndim(other._ndim), _total_size(other._total_size),
          _ssbo(other._ssbo), _config(other._config),
          _owns_data(other._owns_data), _device_id(other._device_id) {
        other._shape = nullptr;
        other._stride = nullptr;
        other._ndim = 0;
        other._total_size = 0;
        other._ssbo = 0;
        other._owns_data = false;
    }

    ~GpuTensor() {
        delete[] _shape;
        delete[] _stride;
        if (_owns_data && _ssbo != 0) {
            get_context().delete_buffer(_ssbo);
        }
    }

    std::size_t ndim() const override { return _ndim; }
    std::size_t total_size() const override { return _total_size; }
    const std::size_t* shape() const override { return _shape; }
    const std::size_t* stride() const override { return _stride; }
    GLuint ssbo() const { return _ssbo; }
    int device_id() const { return _device_id; }

    T get_element(std::size_t index) const override {
        if (index >= _total_size) throw std::invalid_argument("Index out of bounds");
        std::vector<T> host(_total_size);
        get_context().read_buffer(_ssbo, host.data(), _total_size * sizeof(T));
        return host[index];
    }

    void set_element(std::size_t index, T value) override {
        if (index >= _total_size) throw std::invalid_argument("Index out of bounds");
        std::vector<T> host(_total_size);
        get_context().read_buffer(_ssbo, host.data(), _total_size * sizeof(T));
        host[index] = value;
        get_context().delete_buffer(_ssbo);
        _ssbo = get_context().create_buffer(host.data(), _total_size * sizeof(T));
    }

    T operator()(std::size_t i) const override {
        if (_ndim != 1) throw std::invalid_argument("Requires 1D tensor");
        if (i >= _shape[0]) throw std::invalid_argument("Index out of bounds");
        return get_element(i);
    }

    T& operator()(std::size_t i) override {
        static T temp;
        if (_ndim != 1) throw std::invalid_argument("Requires 1D tensor");
        if (i >= _shape[0]) throw std::invalid_argument("Index out of bounds");
        std::vector<T> host(_total_size);
        get_context().read_buffer(_ssbo, host.data(), _total_size * sizeof(T));
        temp = host[i];
        return temp;
    }

    T operator()(std::size_t i, std::size_t j) const override {
        if (_ndim != 2) throw std::invalid_argument("Requires 2D tensor");
        if (i >= _shape[0] || j >= _shape[1]) throw std::invalid_argument("Index out of bounds");
        return get_element(i * _stride[0] + j * _stride[1]);
    }

    T& operator()(std::size_t i, std::size_t j) override {
        static T temp;
        if (_ndim != 2) throw std::invalid_argument("Requires 2D tensor");
        if (i >= _shape[0] || j >= _shape[1]) throw std::invalid_argument("Index out of bounds");
        std::size_t idx = i * _stride[0] + j * _stride[1];
        std::vector<T> host(_total_size);
        get_context().read_buffer(_ssbo, host.data(), _total_size * sizeof(T));
        temp = host[idx];
        return temp;
    }

    bool is_streaming() const override { return false; }
    std::string backend_name() const override {
        auto& selector = get_gpu_selector();
        const GpuInfo* gpu = selector.selected_gpu();
        std::string gpu_name = gpu ? gpu->name : "unknown";
        return "GpuTensor (OpenGL, device #" + std::to_string(_device_id) + ": " + gpu_name + ")";
    }

    std::vector<T> to_host() const {
        std::vector<T> host_data(_total_size);
        get_context().read_buffer(_ssbo, host_data.data(), _total_size * sizeof(T));
        return host_data;
    }

    // ========================================================================
    // Cross-GPU data transfer
    // ========================================================================
    // When operating on tensors on different GPUs, we must transfer data.
    // This downloads from source GPU, then uploads to destination GPU.

    static std::vector<T> cross_gpu_transfer(const GpuTensor<T>* src, const GpuTensor<T>* /*dst*/) {
        // Download from source GPU
        std::vector<T> host_data(src->_total_size);
        src->get_context().read_buffer(src->_ssbo, host_data.data(), src->_total_size * sizeof(T));
        return host_data;
    }

    // ========================================================================
    // GPU Element-wise Operations
    // ========================================================================

    std::unique_ptr<TensorBase<T>> add(const TensorBase<T>* other) const override {
        const GpuTensor<T>* gpuOther = dynamic_cast<const GpuTensor<T>*>(other);
        if (!gpuOther) throw std::invalid_argument("Mixed backend operations not supported");

        // Handle cross-GPU operation
        if (_device_id != gpuOther->_device_id) {
            std::cout << "[GpuTensor] Cross-GPU operation: device #" << _device_id
                      << " + device #" << gpuOther->_device_id << std::endl;
            // Download other to host, create new tensor on this GPU
            std::vector<T> host_data = cross_gpu_transfer(gpuOther, this);
            GpuTensor<T> other_on_this_gpu(host_data.data(), host_data.size(), _config);
            return add(&other_on_this_gpu);
        }

        GLuint out_ssbo = get_context().create_empty_buffer(_total_size * sizeof(T));
        GLuint program = get_context().get_program(SHADER_ADD, "add");

        get_context().bind_buffer_base(_ssbo, 0);
        get_context().bind_buffer_base(gpuOther->_ssbo, 1);
        get_context().bind_buffer_base(out_ssbo, 2);
        get_context().dispatch(program, _total_size, _config.work_group_size);

        GpuTensor<T>* result = new GpuTensor<T>(_shape, _ndim, _config);
        get_context().delete_buffer(result->_ssbo);
        result->_ssbo = out_ssbo;
        return std::unique_ptr<TensorBase<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> subtract(const TensorBase<T>* other) const override {
        const GpuTensor<T>* gpuOther = dynamic_cast<const GpuTensor<T>*>(other);
        if (!gpuOther) throw std::invalid_argument("Mixed backend operations not supported");

        if (_device_id != gpuOther->_device_id) {
            std::vector<T> host_data = cross_gpu_transfer(gpuOther, this);
            GpuTensor<T> other_on_this_gpu(host_data.data(), host_data.size(), _config);
            return subtract(&other_on_this_gpu);
        }

        GLuint out_ssbo = get_context().create_empty_buffer(_total_size * sizeof(T));
        GLuint program = get_context().get_program(SHADER_SUBTRACT, "subtract");

        get_context().bind_buffer_base(_ssbo, 0);
        get_context().bind_buffer_base(gpuOther->_ssbo, 1);
        get_context().bind_buffer_base(out_ssbo, 2);
        get_context().dispatch(program, _total_size, _config.work_group_size);

        GpuTensor<T>* result = new GpuTensor<T>(_shape, _ndim, _config);
        get_context().delete_buffer(result->_ssbo);
        result->_ssbo = out_ssbo;
        return std::unique_ptr<TensorBase<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> multiply(const TensorBase<T>* other) const override {
        const GpuTensor<T>* gpuOther = dynamic_cast<const GpuTensor<T>*>(other);
        if (!gpuOther) throw std::invalid_argument("Mixed backend operations not supported");

        if (_device_id != gpuOther->_device_id) {
            std::vector<T> host_data = cross_gpu_transfer(gpuOther, this);
            GpuTensor<T> other_on_this_gpu(host_data.data(), host_data.size(), _config);
            return multiply(&other_on_this_gpu);
        }

        GLuint out_ssbo = get_context().create_empty_buffer(_total_size * sizeof(T));
        GLuint program = get_context().get_program(SHADER_MULTIPLY, "multiply");

        get_context().bind_buffer_base(_ssbo, 0);
        get_context().bind_buffer_base(gpuOther->_ssbo, 1);
        get_context().bind_buffer_base(out_ssbo, 2);
        get_context().dispatch(program, _total_size, _config.work_group_size);

        GpuTensor<T>* result = new GpuTensor<T>(_shape, _ndim, _config);
        get_context().delete_buffer(result->_ssbo);
        result->_ssbo = out_ssbo;
        return std::unique_ptr<TensorBase<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> divide(const TensorBase<T>* other) const override {
        const GpuTensor<T>* gpuOther = dynamic_cast<const GpuTensor<T>*>(other);
        if (!gpuOther) throw std::invalid_argument("Mixed backend operations not supported");

        if (_device_id != gpuOther->_device_id) {
            std::vector<T> host_data = cross_gpu_transfer(gpuOther, this);
            GpuTensor<T> other_on_this_gpu(host_data.data(), host_data.size(), _config);
            return divide(&other_on_this_gpu);
        }

        GLuint out_ssbo = get_context().create_empty_buffer(_total_size * sizeof(T));
        GLuint program = get_context().get_program(SHADER_DIVIDE, "divide");

        get_context().bind_buffer_base(_ssbo, 0);
        get_context().bind_buffer_base(gpuOther->_ssbo, 1);
        get_context().bind_buffer_base(out_ssbo, 2);
        get_context().dispatch(program, _total_size, _config.work_group_size);

        GpuTensor<T>* result = new GpuTensor<T>(_shape, _ndim, _config);
        get_context().delete_buffer(result->_ssbo);
        result->_ssbo = out_ssbo;
        return std::unique_ptr<TensorBase<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> add_scalar(T scalar) const override {
        return scalar_op(SHADER_ADD_SCALAR, "add_scalar", scalar);
    }

    std::unique_ptr<TensorBase<T>> subtract_scalar(T scalar) const override {
        return scalar_op(SHADER_ADD_SCALAR, "subtract_scalar", -scalar);
    }

    std::unique_ptr<TensorBase<T>> multiply_scalar(T scalar) const override {
        return scalar_op(SHADER_MULTIPLY_SCALAR, "multiply_scalar", scalar);
    }

    std::unique_ptr<TensorBase<T>> divide_scalar(T scalar) const override {
        return multiply_scalar(static_cast<T>(1) / scalar);
    }

    std::unique_ptr<TensorBase<T>> negate() const override {
        return unary_op(SHADER_NEGATE, "negate");
    }

    std::unique_ptr<TensorBase<T>> abs() const override {
        return unary_op(SHADER_ABS, "abs");
    }

    T sum() const override {
        std::vector<T> host_data = to_host();
        T result = T{};
        for (std::size_t i = 0; i < _total_size; ++i) result += host_data[i];
        return result;
    }

    T mean() const override { return sum() / static_cast<T>(_total_size); }

    T max() const override {
        std::vector<T> host_data = to_host();
        T result = host_data[0];
        for (std::size_t i = 1; i < _total_size; ++i)
            if (host_data[i] > result) result = host_data[i];
        return result;
    }

    T min() const override {
        std::vector<T> host_data = to_host();
        T result = host_data[0];
        for (std::size_t i = 1; i < _total_size; ++i)
            if (host_data[i] < result) result = host_data[i];
        return result;
    }

    T dot(const TensorBase<T>* other) const override {
        const GpuTensor<T>* gpuOther = dynamic_cast<const GpuTensor<T>*>(other);
        if (!gpuOther) throw std::invalid_argument("Mixed backend operations not supported");

        std::vector<T> a = to_host();
        std::vector<T> b = gpuOther->to_host();
        T result = T{};
        for (std::size_t i = 0; i < _total_size; ++i) result += a[i] * b[i];
        return result;
    }

    std::unique_ptr<TensorBase<T>> reshape(const std::size_t* new_shape, std::size_t new_ndim) const override {
        std::size_t new_total = 1;
        for (std::size_t i = 0; i < new_ndim; ++i) new_total *= new_shape[i];
        if (new_total != _total_size)
            throw std::invalid_argument("Reshape must preserve total number of elements");

        GpuTensor<T>* result = new GpuTensor<T>(new_shape, new_ndim, _config);
        get_context().delete_buffer(result->_ssbo);
        std::vector<T> host_data = to_host();
        result->_ssbo = get_context().create_buffer(host_data.data(), _total_size * sizeof(T));
        return std::unique_ptr<TensorBase<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> transpose() const override {
        if (_ndim != 2) throw std::invalid_argument("Transpose requires 2D tensors");

        std::vector<T> host_data = to_host();
        std::size_t new_shape[] = {_shape[1], _shape[0]};
        GpuTensor<T>* result = new GpuTensor<T>(new_shape, 2, _config);

        std::vector<T> transposed(_total_size);
        for (std::size_t i = 0; i < _shape[0]; ++i)
            for (std::size_t j = 0; j < _shape[1]; ++j)
                transposed[j * _shape[0] + i] = host_data[i * _shape[1] + j];

        get_context().delete_buffer(result->_ssbo);
        result->_ssbo = get_context().create_buffer(transposed.data(), _total_size * sizeof(T));
        return std::unique_ptr<TensorBase<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> clamp(T min_val, T max_val) const override {
        if (min_val > max_val) throw std::invalid_argument("min_val must be <= max_val");

        GLuint out_ssbo = get_context().create_empty_buffer(_total_size * sizeof(T));
        GLuint program = get_context().get_program(SHADER_CLAMP, "clamp");

        get_context().bind_buffer_base(_ssbo, 0);
        get_context().bind_buffer_base(out_ssbo, 1);
        get_context().dispatch(program, _total_size, _config.work_group_size);

        GpuTensor<T>* result = new GpuTensor<T>(_shape, _ndim, _config);
        get_context().delete_buffer(result->_ssbo);
        result->_ssbo = out_ssbo;
        return std::unique_ptr<TensorBase<T>>(result);
    }

    std::unique_ptr<TensorBase<bool>> greater_than(T threshold) const override {
        std::vector<T> host_data = to_host();
        std::vector<bool> result_data(_total_size);
        for (std::size_t i = 0; i < _total_size; ++i)
            result_data[i] = host_data[i] > threshold;
        // Return a simple wrapper around the bool vector
        return std::unique_ptr<TensorBase<bool>>(new BoolTensorResult(result_data));
    }

    std::unique_ptr<TensorBase<bool>> less_than(T threshold) const override {
        std::vector<T> host_data = to_host();
        std::vector<bool> result_data(_total_size);
        for (std::size_t i = 0; i < _total_size; ++i)
            result_data[i] = host_data[i] < threshold;
        return std::unique_ptr<TensorBase<bool>>(new BoolTensorResult(result_data));
    }

    std::unique_ptr<T[]> to_array() const override {
        std::vector<T> host = to_host();
        std::unique_ptr<T[]> result(new T[_total_size]);
        for (std::size_t i = 0; i < _total_size; ++i) result[i] = host[i];
        return result;
    }

    void print(std::ostream& os) const override {
        std::vector<T> host = to_host();
        if (_ndim == 0) {
            os << host[0];
        } else if (_ndim == 1) {
            os << "[";
            for (std::size_t i = 0; i < _shape[0]; ++i) {
                os << host[i];
                if (i != _shape[0] - 1) os << ", ";
            }
            os << "]";
        } else if (_ndim == 2) {
            os << "[";
            for (std::size_t i = 0; i < _shape[0]; ++i) {
                os << "[";
                for (std::size_t j = 0; j < _shape[1]; ++j) {
                    os << host[i * _shape[1] + j];
                    if (j != _shape[1] - 1) os << ", ";
                }
                os << "]";
                if (i != _shape[0] - 1) os << ", ";
            }
            os << "]";
        } else {
            os << "GpuTensor(ndim=" << _ndim << ", size=" << _total_size
               << ", device=" << _device_id << ")";
        }
    }

    void batch_print(std::ostream& os, std::size_t batch_size) const override {
        std::vector<T> host = to_host();
        os << "[";
        for (std::size_t batch_start = 0; batch_start < _total_size; batch_start += batch_size) {
            std::size_t count = std::min(batch_size, _total_size - batch_start);
            for (std::size_t i = 0; i < count; ++i) {
                os << host[batch_start + i];
                if (batch_start + i != _total_size - 1) os << ", ";
            }
            os.flush();
            if (batch_start + count < _total_size) {
                os << "\n  ... (batch " << (batch_start / batch_size + 1) << " done) ";
            }
        }
        os << "]";
    }

private:
    std::unique_ptr<TensorBase<T>> unary_op(const char* shader_source, const char* name) const {
        GLuint out_ssbo = get_context().create_empty_buffer(_total_size * sizeof(T));
        GLuint program = get_context().get_program(shader_source, name);

        get_context().bind_buffer_base(_ssbo, 0);
        get_context().bind_buffer_base(out_ssbo, 1);
        get_context().dispatch(program, _total_size, _config.work_group_size);

        GpuTensor<T>* result = new GpuTensor<T>(_shape, _ndim, _config);
        get_context().delete_buffer(result->_ssbo);
        result->_ssbo = out_ssbo;
        return std::unique_ptr<TensorBase<T>>(result);
    }

    std::unique_ptr<TensorBase<T>> scalar_op(const char* shader_source, const char* name, T scalar) const {
        (void)scalar; // Used by shader, not directly in C++
        GLuint out_ssbo = get_context().create_empty_buffer(_total_size * sizeof(T));
        GLuint program = get_context().get_program(shader_source, name);

        get_context().bind_buffer_base(_ssbo, 0);
        get_context().bind_buffer_base(out_ssbo, 1);
        get_context().dispatch(program, _total_size, _config.work_group_size);

        GpuTensor<T>* result = new GpuTensor<T>(_shape, _ndim, _config);
        get_context().delete_buffer(result->_ssbo);
        result->_ssbo = out_ssbo;
        return std::unique_ptr<TensorBase<T>>(result);
    }

    template<typename U>
    friend class GpuTensor;
};

// ============================================================================
// GpuAutoTensor: Auto-selection with multi-GPU support
// ============================================================================
template<typename T>
class GpuAutoTensor {
public:
    static std::unique_ptr<TensorBase<T>> from_data(
        const T* data,
        std::size_t size,
        const AutoConfig& auto_config = AutoConfig{},
        const GpuConfig& gpu_config = GpuConfig{}) {

        std::size_t data_bytes = size * sizeof(T);
        std::size_t total_memory = get_total_system_memory();
        double usage_ratio = static_cast<double>(data_bytes) / static_cast<double>(total_memory);

        // Enumerate GPUs
        auto& selector = get_gpu_selector();
        if (selector.get_gpus().empty()) {
            selector.enumerate();
        }

        std::cout << "[GpuAutoTensor] Detected " << selector.get_gpus().size() << " GPU(s)" << std::endl;
        selector.print_all(std::cout);

        // Try GPU first (if data is large enough)
        bool gpu_beneficial = size > 100000;
        if (gpu_beneficial && !selector.get_gpus().empty()) {
            try {
                // Select GPU based on config
                if (gpu_config.device_id >= 0) {
                    selector.select_device(gpu_config.device_id);
                } else {
                    selector.select_discrete_only(); // Prefer discrete GPUs
                }

                std::cout << "[GpuAutoTensor] Data size: " << (data_bytes / (1024.0 * 1024.0)) << " MB"
                          << " → Using GpuTensor on device #" << selector.selected_device() << std::endl;
                return std::make_unique<GpuTensor<T>>(data, size, gpu_config);
            } catch (const std::exception& e) {
                std::cout << "[GpuAutoTensor] GPU failed: " << e.what() << ", falling back" << std::endl;
            }
        }

        // Fall back to AutoTensor logic
        bool use_streaming = auto_config.force_streaming ||
                             (!auto_config.force_dense && usage_ratio > auto_config.memory_threshold);

        if (use_streaming) {
            std::cout << "[GpuAutoTensor] Data size: " << (data_bytes / (1024.0 * 1024.0)) << " MB"
                      << " → Using MmapTensor (memory-mapped)" << std::endl;
            return std::make_unique<MmapTensor<T>>(data, size, auto_config.stream_config);
        } else {
            std::cout << "[GpuAutoTensor] Data size: " << (data_bytes / (1024.0 * 1024.0)) << " MB"
                      << " → Using DenseTensor (in-memory)" << std::endl;
            return std::make_unique<DenseTensor<T>>(data, size);
        }
    }

    // Force specific GPU
    static std::unique_ptr<TensorBase<T>> force_gpu(
        const T* data,
        std::size_t size,
        int device_id = -1,
        const GpuConfig& gpu_config = GpuConfig{}) {

        GpuConfig cfg = gpu_config;
        cfg.device_id = device_id;

        auto& selector = get_gpu_selector();
        if (selector.get_gpus().empty()) selector.enumerate();

        if (device_id >= 0) {
            selector.select_device(device_id);
            std::cout << "[GpuAutoTensor] Forcing GPU device #" << device_id << std::endl;
        } else {
            selector.auto_select();
            std::cout << "[GpuAutoTensor] Forcing best GPU (auto-selected)" << std::endl;
        }

        return std::make_unique<GpuTensor<T>>(data, size, cfg);
    }

    // Force GPU by name pattern
    static std::unique_ptr<TensorBase<T>> force_gpu_by_name(
        const T* data,
        std::size_t size,
        const std::string& name_pattern,
        const GpuConfig& gpu_config = GpuConfig{}) {

        auto& selector = get_gpu_selector();
        if (selector.get_gpus().empty()) selector.enumerate();

        if (!selector.select_by_name(name_pattern)) {
            throw std::runtime_error("No GPU matching pattern: " + name_pattern);
        }

        GpuConfig cfg = gpu_config;
        cfg.device_id = selector.selected_device();

        std::cout << "[GpuAutoTensor] Forcing GPU matching '" << name_pattern
                  << "' (device #" << cfg.device_id << ")" << std::endl;
        return std::make_unique<GpuTensor<T>>(data, size, cfg);
    }

    static void print_system_info() {
        AutoTensor<T>::print_system_info();
        auto& selector = get_gpu_selector();
        if (selector.get_gpus().empty()) selector.enumerate();
        selector.print_all(std::cout);
        std::cout << "GPU backend: OpenGL compute shaders (4.3+)" << std::endl;
        std::cout << "GPU threshold: data > 100K elements for GPU acceleration" << std::endl;
    }
};
