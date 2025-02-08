#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define GLFW_INCLUDE_NONE
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include "gdt/math/AffineSpace.h"

#include "boundingBox.h"
#include "transferFunction.h"
#include "rapidcsv.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace gdt;

struct GLFWindow {
  GLFWindow(){};
  GLFWindow(const std::string &title, int2 frame_size);
  GLFWindow(const std::string &title, int2 frame_size, bool renderGUI);
  GLFWindow(int2 frame_size);
  ~GLFWindow();

  /*! put pixels on the screen ... */
  // virtual void draw()
  void draw(uint8_t* pixels);
  // { /* empty - to be subclassed by user */ }

  /*! callback that window got resized */
  virtual void resize(const vec2i& newSize)
  {
      fbSize.x = newSize.x;
      fbSize.y = newSize.y;
      renderParamsChanged = true;
  }
  //{ /* empty - to be subclassed by user */ }

  virtual void key(int key, int action, int mods)
  {}
  
  /*! callback that window got resized */
  virtual void mouseMotion(float xPos, float yPos)
  {}
  
  virtual void mouseScroll(float xOffset, float yOffset)
  {}

  /*! callback that window got resized */
  virtual void mouseButton(int button, int action, int mods)
  {}

  inline vec2i getMousePos() const
  {
    double x,y;
    glfwGetCursorPos(handle,&x,&y);
    return vec2i((int)x, (int)y);
  }

  /*! re-render the frame - typically part of draw(), but we keep
    this a separate function so render() can focus on optix
    rendering, and now have to deal with opengl pixel copies
    etc */
  virtual void render() 
  { /* empty - to be subclassed by user */ }

  GLuint shaderProgram;
  GLuint VBO, VAO;
  // Create and bind the VBO for the rectangle
  float vertices[12] = {
      -1.0f, -1.0f, 0.0f,   // Bottom-left vertex
       1.0f, -1.0f, 0.0f,   // Bottom-right vertex
       1.0f,  1.0f, 0.0f,   // Top-right vertex
      -1.0f,  1.0f, 0.0f    // Top-left vertex
  };

  float texCoords[8] = {
      0.0f, 0.0f,   // Bottom-left texture coordinate
      1.0f, 0.0f,   // Bottom-right texture coordinate
      1.0f, 1.0f,   // Top-right texture coordinate
      0.0f, 1.0f    // Top-left texture coordinate
  };
  void generateAndbindbuffer();
  std::string readShaderCode(const char* a);
  GLuint compileShader(GLenum shaderType, const char* shaderCode);
  void setUniformShaderVariable();
  void compileAndLinkShader();

  void renderShaderOutput(uint8_t* frame, vec2f eye, vec2i fovSize, std::string& filePath);

  /*! opens the actual window, and runs the window's events to
    completion. This function will only return once the window
    gets closed */

  void renderImgui();

  void otherRenderingParams();

  void tabs();

  int2 getframeSize();
  int2 fbSize;

  void setFrameIndex(int index) { frame_index = index; }
  void setAspectRatio(float aspectRatio_) { aspectRatio = aspectRatio_; if (aspectRatio != 1.f) { foveatedRendering = true; } }
  void setEyePos(float2 eyePos_) { eyePos = eyePos_; }
  void terminateGUI();

  /*! the glfw window handle */
  GLFWwindow *handle { nullptr };
  GLuint                                      fbTexture {0};

  bool renderGUI = false;

  float shininess = (float)28.0f;
  bool enableHeadLight = true;
  float attenuationConstA = (float)1.0f;
  float attenuationConstB = (float)0.001f;
  float attenuationConstC = (float)0.0f;
  float ambientConstant = (float)0.3f;
  float phongThreshold = 0.0f;

  int frame_index = 1;
  int envToneMap = 3;
  float gammaEnv = (float)2.2f;
  float exposureEnv = (float)1.0f;

  float2 focalLength = make_float2(0.5f, 0.5f);
  float2 eyePos = make_float2(0.5f, 0.5f);                     // eye Positon in frame normalized coordinates,
  float aspectRatio = (float)1.f;                           // originFramebuffer / smallerFramebuffer


  bool lightsChanged = false;
  bool tfChanged = false;
  bool renderParamsChanged = true;

  bool accumulateFrames = true;
  bool denoiseImage = false;
  bool enablePhongShading = true;
  bool foveatedRendering_ = false;
  bool foveatedRendering = false;

  float fontSize = 1.5f;

  glm::vec3 cameraPosition;
  float cameraFOV = 10.f;
  float cameraFocalLength = 28.f;
  float cameraApertureRadius = 0.0f;
  bool cameraParamsChanged = false;

  float currentFrameTime = 0.0f, lastFrameTime = 0.0f, deltaTime = 0.0f;

  bool CloseApplication = false;
};

struct CameraFrame
{
    glm::vec3 position, up, front, right;

    void setOrientation(const glm::vec3& position_, const glm::vec3& target, const glm::vec3& up_)
    {
        position = position_;
        front = normalize(target - position);
        right = normalize(cross(front, up_));
        up = cross(right, front);
    }

	float3 get_from() const { return make_float3(position.x, position.y, position.z); }
	float3 get_front() const { return make_float3(front.x, front.y, front.z); }
	float3 get_up() const { return make_float3(up.x, up.y, up.z); }

	void set_from(const glm::vec3& from) { position = from; }
	void set_front(const glm::vec3& f) { front = f; }
	void set_up(const glm::vec3& u) { up = u; }

    void updateFrontVector(glm::vec3 front_)
    {
		front = front_;
		right = glm::normalize(glm::cross(front, up));
		up = glm::normalize(glm::cross(right, front));
    }
};

struct GLFCameraWindow : public GLFWindow {
    GLFCameraWindow(const std::string &title,
                  const glm::vec3 &camera_from,
                  const glm::vec3 &camera_at,
                  const glm::vec3 &camera_up,
                  const int2 &frame_size,
                  const float worldScale)
        : GLFWindow(title, frame_size), position(camera_from)
    {
        cameraFrame.setOrientation(camera_from,camera_at,camera_up);
        position = camera_from;
        up = camera_up;
        front = glm::normalize(camera_at - camera_from);
        cameraFrame.updateFrontVector(front);
    }

    // Virtual function to handle keyboard input
    virtual void key(int key, int action, int mods) override
    {
        if (key == GLFW_KEY_ESCAPE) {
            CloseApplication = true;
            return;
        }
        if (!RightButtonPressed) return;

        float velocity = speed * deltaTime;

        switch (key) {
        case GLFW_KEY_W:
            position += front * velocity;
            break;
        case GLFW_KEY_S:
		    position -= front * velocity;
		    break;
        case GLFW_KEY_A:
            position -= glm::normalize(glm::cross(front, up)) * velocity;
            break;
        case GLFW_KEY_D:
            position += glm::normalize(glm::cross(front, up)) * velocity;
            break;
        case GLFW_KEY_Q:
            position -= up * velocity;
            break;
        case GLFW_KEY_E:
		    position += up * velocity;
		    break;
        }
        cameraFrame.set_from(position);

	    renderParamsChanged = true;
    }
  
    virtual void mouseMotion(float xPos, float yPos) override
    {
        if (aspectRatio != 1.f) {
            eyePos.x = xPos / fbSize.x;
            eyePos.y = 1.f - (yPos / fbSize.y);
        }
        if (!RightButtonPressed) return;
        if (firstMouseMovement) {
            lastX = xPos;
            lastY = yPos;
            firstMouseMovement = false;
            return;
        }

        float xoffset = xPos - lastX;
        float yoffset = lastY - yPos; // Reversed
        lastX = xPos, lastY = yPos;

        yaw += xoffset * sensitivity;
        pitch += yoffset * sensitivity;

        // Clamp pitch
        pitch = glm::clamp(pitch, -89.0f, 89.0f);

        // Calculate new front vector
        glm::vec3 direction;
        direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        direction.y = sin(glm::radians(pitch));
        direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(direction);

        cameraFrame.updateFrontVector(front);

        renderParamsChanged = true;
    }

    virtual void mouseScroll(float xoffset, float yoffset)
    {
        if (!RightButtonPressed) return;

        speed = yoffset > 0 ? speed + 50.f : speed - 50.f;
        speed = glm::clamp(speed, 128.0f, 8992.0f);
    }
  
    virtual void mouseButton(int button, int action, int mods) override
    {
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                RightButtonPressed = true;
                firstMouseMovement = true;
            }
            else if (action == GLFW_RELEASE) {
                RightButtonPressed = false;
            }
        }
    }

    int keyPressed;
    CameraFrame cameraFrame;

    /*
    * Mouse input variables
    */
    bool RightButtonPressed = false;
    bool firstMouseMovement = true;
    float lastX, lastY;
    float speed = 128.f, sensitivity = 0.1f;
    float yaw = -90.f, pitch = 0.f;
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

};
