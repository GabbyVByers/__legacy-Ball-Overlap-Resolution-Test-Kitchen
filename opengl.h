#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <string>
#include <tuple>

#include "simulationstate.h"

class InteropOpenGL
{
public:

    int screenWidth = -1;
    int screenHeight = -1;
    GLFWwindow* window = nullptr;
    dim3 PIXLS_threadsPerBlock = 0;
    dim3 PIXLS_blocksPerGrid = 0;
    GLuint PBO = 0;
    cudaGraphicsResource* cudaPBO = nullptr;
    GLuint textureId = 0;
    GLuint shader = 0;
    GLuint quadVAO = 0;
    GLuint quadVBO = 0;

	InteropOpenGL(int width, int height, std::string title, bool fullScreen)
	{
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        if (fullScreen)
        {
            GLFWmonitor* primary = glfwGetPrimaryMonitor();
            screenWidth = glfwGetVideoMode(primary)->width;
            screenHeight = glfwGetVideoMode(primary)->height;
            window = glfwCreateWindow(screenWidth, screenHeight, title.c_str(), primary, nullptr);
        }
        else
        {
            screenWidth = width;
            screenHeight = height;
            window = glfwCreateWindow(screenWidth, screenHeight, title.c_str(), nullptr, nullptr);
        }

        PIXLS_threadsPerBlock = dim3(32, 32);
        PIXLS_blocksPerGrid = dim3((screenWidth / 32) + 1, (screenHeight / 32) + 1);

        glfwMakeContextCurrent(window);
        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        glGenBuffers(1, &PBO);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, screenWidth * screenHeight * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        cudaGraphicsGLRegisterBuffer(&cudaPBO, PBO, cudaGraphicsMapFlagsWriteDiscard);
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screenWidth, screenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        const char* vertexShaderSrc =
            R"glsl(
                #version 330 core
                layout (location = 0) in vec2 aPos;
                layout (location = 1) in vec2 aTex;
                out vec2 texCoord;
                void main()
                {
                    gl_Position = vec4(aPos.xy, 0.0, 1.0);
                    texCoord = aTex;
                }
            )glsl";

        const char* fragmentShaderSrc =
            R"glsl(
                #version 330 core
                in vec2 texCoord;
                out vec4 fragColor;
                uniform sampler2D screenTexture;
                void main()
                {
                    fragColor = texture(screenTexture, texCoord);
                }
            )glsl";

        GLuint vert = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert, 1, &vertexShaderSrc, nullptr);
        glCompileShader(vert);
        GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag, 1, &fragmentShaderSrc, nullptr);
        glCompileShader(frag);
        shader = glCreateProgram();
        glAttachShader(shader, vert);
        glAttachShader(shader, frag);
        glLinkProgram(shader);
        glDeleteShader(vert);
        glDeleteShader(frag);

        float quadVertices[] =
        {
            -1.0f,  1.0f,   0.0f, 1.0f,
            -1.0f, -1.0f,   0.0f, 0.0f,
             1.0f, -1.0f,   1.0f, 0.0f,
            -1.0f,  1.0f,   0.0f, 1.0f,
             1.0f, -1.0f,   1.0f, 0.0f,
             1.0f,  1.0f,   1.0f, 1.0f
        };

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);

        initImGui();
        enableVSYNC();
	}

    ~InteropOpenGL() { free(); }

    void free()
    {
        cudaGraphicsUnregisterResource(cudaPBO);
        glDeleteBuffers(1, &PBO);
        glDeleteTextures(1, &textureId);
        glDeleteVertexArrays(1, &quadVAO);
        glDeleteBuffers(1, &quadVBO);
        glDeleteProgram(shader);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void executeKernels(SimulationState& simState);
    void initImGui();
    void renderImGui(SimulationState& simState);
    void processUserInput();

    bool isAlive() const { return !glfwWindowShouldClose(window); }
    void enableVSYNC() { glfwSwapInterval(1); }
    void disableVSYNC() { glfwSwapInterval(0); }
    
    std::tuple<int, int> getScreenDim()
    {
        return std::tuple<int, int>(screenWidth, screenHeight);
    }

    void renderFullScreenQuad()
    {
        cudaGraphicsUnmapResources(1, &cudaPBO, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader);
        glBindVertexArray(quadVAO);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    void swapBuffers() const
    {
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
};

