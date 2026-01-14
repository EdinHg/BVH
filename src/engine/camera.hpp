#pragma once

#include <cmath>

struct Mat4 {
    float m[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    const float* data() const { return m; }

    static Mat4 identity() { return Mat4(); }

    static Mat4 perspective(float fov, float aspect, float near, float far) {
        Mat4 r;
        float tanHalf = std::tan(fov * 0.5f);
        r.m[0] = 1.0f / (aspect * tanHalf);
        r.m[5] = 1.0f / tanHalf;
        r.m[10] = -(far + near) / (far - near);
        r.m[11] = -1.0f;
        r.m[14] = -(2.0f * far * near) / (far - near);
        r.m[15] = 0.0f;
        return r;
    }

    static Mat4 lookAt(float eyeX, float eyeY, float eyeZ,
                       float cenX, float cenY, float cenZ,
                       float upX, float upY, float upZ) {
        float fx = cenX - eyeX, fy = cenY - eyeY, fz = cenZ - eyeZ;
        float flen = std::sqrt(fx*fx + fy*fy + fz*fz);
        if (flen > 0) { fx /= flen; fy /= flen; fz /= flen; }

        float sx = fy * upZ - fz * upY;
        float sy = fz * upX - fx * upZ;
        float sz = fx * upY - fy * upX;
        float slen = std::sqrt(sx*sx + sy*sy + sz*sz);
        if (slen > 0) { sx /= slen; sy /= slen; sz /= slen; }

        float ux = sy * fz - sz * fy;
        float uy = sz * fx - sx * fz;
        float uz = sx * fy - sy * fx;

        Mat4 r;
        r.m[0] = sx;  r.m[4] = sy;  r.m[8]  = sz;  r.m[12] = -(sx*eyeX + sy*eyeY + sz*eyeZ);
        r.m[1] = ux;  r.m[5] = uy;  r.m[9]  = uz;  r.m[13] = -(ux*eyeX + uy*eyeY + uz*eyeZ);
        r.m[2] = -fx; r.m[6] = -fy; r.m[10] = -fz; r.m[14] = (fx*eyeX + fy*eyeY + fz*eyeZ);
        r.m[3] = 0;   r.m[7] = 0;   r.m[11] = 0;   r.m[15] = 1;
        return r;
    }

    Mat4 operator*(const Mat4& b) const {
        Mat4 r;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                r.m[j*4+i] = 0;
                for (int k = 0; k < 4; ++k) {
                    r.m[j*4+i] += m[k*4+i] * b.m[j*4+k];
                }
            }
        }
        return r;
    }
};

// Free-fly camera with WASD + mouse look
class Camera {
public:
    // Position
    float posX = 0.0f, posY = 0.0f, posZ = 3.0f;
    // Euler angles (radians)
    float yaw = 3.14159f;   // Looking towards -Z initially
    float pitch = 0.0f;
    // Movement speed
    float speed = 1.0f;

    void setPosition(float x, float y, float z) {
        posX = x; posY = y; posZ = z;
    }

    void rotate(float dYaw, float dPitch) {
        yaw += dYaw;
        pitch += dPitch;
        // Clamp pitch to avoid gimbal lock
        if (pitch > 1.5f) pitch = 1.5f;
        if (pitch < -1.5f) pitch = -1.5f;
    }

    // Get forward direction
    void getForward(float& fx, float& fy, float& fz) const {
        fx = std::cos(pitch) * std::sin(yaw);
        fy = std::sin(pitch);
        fz = std::cos(pitch) * std::cos(yaw);
    }

    // Get right direction
    void getRight(float& rx, float& ry, float& rz) const {
        rx = std::sin(yaw - 1.5708f);
        ry = 0.0f;
        rz = std::cos(yaw - 1.5708f);
    }

    // Move camera based on input (FPS-style: horizontal movement follows mouse look)
    void move(bool forward, bool backward, bool left, bool right, bool up, bool down, float deltaTime) {
        // Get horizontal forward direction (ignoring pitch for WASD movement)
        float hfx = std::sin(yaw);
        float hfz = std::cos(yaw);
        
        // Get right direction
        float rx = std::sin(yaw - 1.5708f);
        float rz = std::cos(yaw - 1.5708f);

        float moveSpeed = speed * deltaTime;

        if (forward)  { posX += hfx * moveSpeed; posZ += hfz * moveSpeed; }
        if (backward) { posX -= hfx * moveSpeed; posZ -= hfz * moveSpeed; }
        if (left)     { posX -= rx * moveSpeed; posZ -= rz * moveSpeed; }
        if (right)    { posX += rx * moveSpeed; posZ += rz * moveSpeed; }
        if (up)       { posY += moveSpeed; }
        if (down)     { posY -= moveSpeed; }
    }

    // Zoom by moving camera forward/backward
    void zoom(float delta) {
        float fx, fy, fz;
        getForward(fx, fy, fz);
        
        float zoomSpeed = speed * delta * 0.1f;
        posX += fx * zoomSpeed;
        posY += fy * zoomSpeed;
        posZ += fz * zoomSpeed;
    }

    // Pan in screen space
    void pan(float dx, float dy) {
        float rx, ry, rz;
        getRight(rx, ry, rz);
        
        float panSpeed = speed * 0.001f;
        posX += rx * dx * panSpeed;
        posZ += rz * dx * panSpeed;
        posY += dy * panSpeed;
    }

    Mat4 viewMatrix() const {
        float fx, fy, fz;
        const_cast<Camera*>(this)->getForward(fx, fy, fz);
        return Mat4::lookAt(posX, posY, posZ,
                            posX + fx, posY + fy, posZ + fz,
                            0, 1, 0);
    }

    Mat4 projMatrix(float aspect) const {
        return Mat4::perspective(1.0f, aspect, 0.01f, 1000.0f);
    }
};
