#pragma once

#include <glad/glad.h>
#include <string>
#include <stdexcept>

class Shader {
public:
    GLuint program = 0;

    Shader() = default;

    void compile(const char* vertexSrc, const char* fragmentSrc) {
        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vertexSrc, nullptr);
        glCompileShader(vs);
        checkCompile(vs, "vertex");

        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fragmentSrc, nullptr);
        glCompileShader(fs);
        checkCompile(fs, "fragment");

        program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        checkLink(program);

        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    void use() const { glUseProgram(program); }

    GLint loc(const char* name) const {
        return glGetUniformLocation(program, name);
    }

    ~Shader() {
        if (program) glDeleteProgram(program);
    }

private:
    void checkCompile(GLuint shader, const char* type) {
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[512];
            glGetShaderInfoLog(shader, 512, nullptr, log);
            throw std::runtime_error(std::string(type) + " shader error: " + log);
        }
    }

    void checkLink(GLuint prog) {
        GLint success;
        glGetProgramiv(prog, GL_LINK_STATUS, &success);
        if (!success) {
            char log[512];
            glGetProgramInfoLog(prog, 512, nullptr, log);
            throw std::runtime_error(std::string("Link error: ") + log);
        }
    }
};

// Simple vertex color shader
inline const char* VERTEX_SHADER = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uMVP;

out vec3 vColor;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
}
)";

inline const char* FRAGMENT_SHADER = R"(
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
)";

// Wireframe shader - uses flat shading for better edge visibility
inline const char* WIREFRAME_VERTEX_SHADER = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

uniform mat4 uMVP;

out vec3 vColor;
out vec3 vPos;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
    vPos = aPos;
}
)";

inline const char* WIREFRAME_FRAGMENT_SHADER = R"(
#version 330 core
in vec3 vColor;
in vec3 vPos;

out vec4 FragColor;

void main() {
    // Compute flat-shaded normal from screen-space derivatives
    vec3 fdx = dFdx(vPos);
    vec3 fdy = dFdy(vPos);
    vec3 normal = normalize(cross(fdx, fdy));
    
    // Simple directional lighting to highlight edges
    vec3 lightDir = normalize(vec3(0.5, 0.7, 0.3));
    float diff = max(dot(normal, lightDir), 0.0) * 0.6 + 0.4;
    
    FragColor = vec4(vColor * diff, 1.0);
}
)";
