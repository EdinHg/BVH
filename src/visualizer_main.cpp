#include <iostream>
#include <string>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "mesh/obj_loader.hpp"
#include "bvh/recursive_bvh.hpp"
#include "bvh/lbvh_builder.hpp"
#include "visualization/leaf_cost.hpp"
#include "engine/shader.hpp"
#include "engine/camera.hpp"
#include "engine/mesh_renderer.hpp"
#include "benchmark/timer.hpp"
#include "benchmark/sah_cost.hpp"

// Global state
static Camera g_camera;
static bool g_dragging = false;
static bool g_panning = false;
static bool g_firstMouse = true;
static double g_lastX = 0, g_lastY = 0;
static bool g_heatmapEnabled = true;
static bool g_needsUpload = true;

// Input state
static bool g_keyW = false, g_keyS = false, g_keyA = false, g_keyD = false;
static bool g_keySpace = false, g_keyShift = false;

static void printControls() {
    std::cout << "\nControls:\n";
    std::cout << "  WASD        - Move\n";
    std::cout << "  Space/Shift - Up/Down\n";
    std::cout << "  Mouse       - Look around\n";
    std::cout << "  Middle-drag - Pan\n";
    std::cout << "  Scroll      - Zoom in/out\n";
    std::cout << "  H           - Toggle heatmap\n";
    std::cout << "  ESC         - Exit\n\n";
}

static void cursorCallback(GLFWwindow*, double x, double y) {
    if (g_firstMouse) {
        g_lastX = x;
        g_lastY = y;
        g_firstMouse = false;
        return;
    }
    
    float dx = static_cast<float>(x - g_lastX);
    float dy = static_cast<float>(y - g_lastY);
    
    g_lastX = x;
    g_lastY = y;
    
    // Always rotate camera with mouse movement
    if (g_panning) {
        g_camera.pan(-dx, dy);
    } else {
        g_camera.rotate(-dx * 0.003f, -dy * 0.003f);
    }
}

static void mouseButtonCallback(GLFWwindow*, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_dragging = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        g_panning = (action == GLFW_PRESS);
    }
}

static void scrollCallback(GLFWwindow*, double, double yoff) {
    g_camera.zoom(static_cast<float>(yoff));
}

static void keyCallback(GLFWwindow* window, int key, int, int action, int) {
    bool pressed = (action != GLFW_RELEASE);

    if (key == GLFW_KEY_W) g_keyW = pressed;
    if (key == GLFW_KEY_S) g_keyS = pressed;
    if (key == GLFW_KEY_A) g_keyA = pressed;
    if (key == GLFW_KEY_D) g_keyD = pressed;
    if (key == GLFW_KEY_SPACE) g_keySpace = pressed;
    if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) g_keyShift = pressed;

    if (key == GLFW_KEY_H && action == GLFW_PRESS) {
        g_heatmapEnabled = !g_heatmapEnabled;
        g_needsUpload = true;
        std::cout << "Heatmap: " << (g_heatmapEnabled ? "ON" : "OFF") << "\n";
    }

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: bvh_visualizer <model.obj> [--density] [--lbvh]\n";
        return 1;
    }

    std::string modelPath = argv[1];
    bool densityOnly = false;
    bool useLBVH = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--density") densityOnly = true;
        if (std::string(argv[i]) == "--lbvh") useLBVH = true;
    }

    // Load mesh
    TriangleMesh mesh;
    try {
        mesh = loadOBJ(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Loaded " << mesh.size() << " triangles\n";

    // Build BVH
    Timer timer;
    BVHResult bvh;
    timer.start();
    if (useLBVH) {
        LBVHBuilder builder;
        bvh = builder.build(mesh);
        std::cout << "Using LBVH (Morton code) builder\n";
    } else {
        RecursiveBVH builder(4);
        bvh = builder.build(mesh);
        std::cout << "Using Recursive BVH builder\n";
    }
    timer.stop();

    float sahCost = calculateSAHCost(bvh.nodes);
    std::cout << "BVH built in " << timer.elapsedMs() << " ms, SAH cost: " << sahCost << "\n";

    // Compute scene bounds and position camera
    AABB sceneBounds;
    for (size_t i = 0; i < mesh.size(); ++i) {
        sceneBounds.expand(mesh.getBounds(i));
    }
    Vec3 center = sceneBounds.center();
    float size = sceneBounds.extent().length();

    // Position camera to view the whole model
    g_camera.setPosition(center.x, center.y, center.z + size * 1.5f);
    g_camera.yaw = 3.14159f;  // Look towards -Z (towards model)
    g_camera.pitch = 0.0f;
    g_camera.speed = size * 0.5f;

    // Precompute colors
    std::vector<Vec3> heatmapColors = computeLeafCostColors(mesh, bvh, densityOnly);

    // Default white color for non-heatmap mode
    std::vector<Vec3> whiteColors(mesh.size(), Vec3(0.8f, 0.8f, 0.8f));

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "BVH Visualizer (H=toggle heatmap)", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, cursorCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return 1;
    }

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.15f, 0.15f, 0.2f, 1.0f);

    Shader shader;
    shader.compile(VERTEX_SHADER, FRAGMENT_SHADER);
    
    Shader wireframeShader;
    wireframeShader.compile(WIREFRAME_VERTEX_SHADER, WIREFRAME_FRAGMENT_SHADER);

    MeshRenderer renderer;
    GLint mvpLoc = shader.loc("uMVP");
    GLint mvpWireLoc = wireframeShader.loc("uMVP");

    printControls();

    double lastTime = glfwGetTime();

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float deltaTime = static_cast<float>(currentTime - lastTime);
        lastTime = currentTime;

        glfwPollEvents();

        // Camera movement
        g_camera.move(g_keyW, g_keyS, g_keyA, g_keyD, g_keySpace, g_keyShift, deltaTime);

        // Re-upload mesh if needed (heatmap toggled)
        if (g_needsUpload) {
            renderer.upload(mesh, g_heatmapEnabled ? heatmapColors : whiteColors);
            g_needsUpload = false;
        }

        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        if (w > 0 && h > 0) {
            glViewport(0, 0, w, h);

            Mat4 proj = g_camera.projMatrix(static_cast<float>(w) / h);
            Mat4 view = g_camera.viewMatrix();
            Mat4 mvp = proj * view;

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Use wireframe shader when heatmap is off for better edge visibility
            if (g_heatmapEnabled) {
                shader.use();
                glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, mvp.data());
            } else {
                wireframeShader.use();
                glUniformMatrix4fv(mvpWireLoc, 1, GL_FALSE, mvp.data());
            }
            renderer.draw();
        }

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
