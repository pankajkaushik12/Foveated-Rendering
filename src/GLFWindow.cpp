// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "GLFWindow.h"

using namespace gdt;

#define pi 3.14159265358979323846f
//static void windowPosCallback(GLFWwindow* window, int xpos, int ypos) {
//    // Get the position of the window
//    int uiWindowX, uiWindowY;
//    glfwGetWindowPos(window, &uiWindowX, &uiWindowY);
//
//    // Update the position of the ImGui UI window
//    ImGui::SetNextWindowPos(ImVec2(uiWindowX, uiWindowY), ImGuiCond_Always);
//}

bool RightMouseButtonPressed = false;

std::string GLFWindow::readShaderCode(const char* shaderPath) {
    std::ifstream shaderFile(shaderPath);
    std::stringstream shaderStream;
    shaderStream << shaderFile.rdbuf();
    return shaderStream.str();
}

GLuint GLFWindow::compileShader(GLenum shaderType, const char* shaderCode) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderCode, nullptr);
    glCompileShader(shader);

    // Check shader compilation status.
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

static void glfw_error_callback(int error, const char* description)
{
  fprintf(stderr, "Error: %s\n", description);
}

/*! callback for a window resizing event */
static void glfwindow_reshape_cb(GLFWwindow* window, int width, int height )
{
  GLFWindow *gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
  assert(gw);
  gw->resize(vec2i(width,height));
// assert(GLFWindow::current);
//   GLFWindow::current->resize(vec2i(width,height));
}

static void glfwindow_key_cb(GLFWwindow *window, int key, int scancode, int action, int mods) 
{
    GLFWindow *gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);

    gw->key(key, action, mods);
}

static void glfwindow_mouseMotion_cb(GLFWwindow *window, double xPos, double yPos) 
{
    GLFWindow* gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);

	ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        gw->mouseMotion(float(xPos), float(yPos));
    }
}

static void glfwindow_mouseScroll_cb(GLFWwindow* window, double xoffset, double yoffset) {
    GLFWindow* gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);

    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        gw->mouseScroll(float(xoffset), float(yoffset));
    }
}

static void glfwindow_mouseButton_cb(GLFWwindow *window, int button, int action, int mods) 
{    
    GLFWindow* gw = static_cast<GLFWindow*>(glfwGetWindowUserPointer(window));
    assert(gw);

	ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
		gw->mouseButton(button, action, mods);
    }
}


GLFWindow::~GLFWindow()
{
  glfwDestroyWindow(handle);
  glfwTerminate();
}

 const char* setGLSLVersion() {
 #if defined(IMGUI_IMPL_OPENGL_ES2)
     // GL ES 2.0 + GLSL 100
     const char* glsl_version = "#version 100";
     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
     glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
 #elif defined(__APPLE__)
     // GL 3.2 + GLSL 150
     const char* glsl_version = "#version 150";
     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
     glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
     glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
 #else
     // GL 3.0 + GLSL 130
     const char* glsl_version = "#version 130";
     glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
     glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
     //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
     //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
 #endif
     return glsl_version;
 }

GLFWindow::GLFWindow(const std::string &title, int2 frame_size)
{
  glfwSetErrorCallback(glfw_error_callback);
    
  if (!glfwInit())
    exit(EXIT_FAILURE);
    
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    
  handle = glfwCreateWindow(frame_size.x, frame_size.y, title.c_str(), NULL, NULL);
  if (!handle) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
    
  glfwSetWindowUserPointer(handle, this);
  glfwMakeContextCurrent(handle);
  glfwSwapInterval( 1 );

  GLenum err = glewInit();
  
  fbSize = make_int2(frame_size.x, frame_size.y);

  // glfwSetWindowUserPointer(window, GLFWindow::current);
  glfwSetFramebufferSizeCallback(handle, glfwindow_reshape_cb);
  glfwSetMouseButtonCallback(handle, glfwindow_mouseButton_cb);
  glfwSetKeyCallback(handle, glfwindow_key_cb);
  glfwSetCursorPosCallback(handle, glfwindow_mouseMotion_cb);
  glfwSetScrollCallback(handle, glfwindow_mouseScroll_cb);

  compileAndLinkShader();             // Compiling and linking shaders
  generateAndbindbuffer();            // Generating and binding buffers

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;

  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(fontSize);  // Adjust the scaling factor as needed
  ImGui::GetIO().FontGlobalScale = fontSize;
  //ImGui::StyleColorsLight();

  // Setup Platform/Renderer backends
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(handle, true);
  ImGui_ImplOpenGL3_Init(setGLSLVersion());
  //std::cout << "OpenGL version: " << setGLSLVersion() << std::endl;

}

GLFWindow::GLFWindow(const std::string& title, int2 frame_size, bool renderGUI_)
{
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    handle = glfwCreateWindow(frame_size.x, frame_size.y, title.c_str(), NULL, NULL);
    if (!handle) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    glfwSwapInterval(1);

    GLenum err = glewInit();

    fbSize = make_int2(frame_size.x, frame_size.y);
    renderGUI = renderGUI_;

    compileAndLinkShader();             // Compiling and linking shaders
    generateAndbindbuffer();            // Generating and binding buffers

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(fontSize);  // Adjust the scaling factor as needed
    ImGui::GetIO().FontGlobalScale = fontSize;
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(handle, true);
    ImGui_ImplOpenGL3_Init(setGLSLVersion());
    //std::cout << "OpenGL version: " << setGLSLVersion() << std::endl;
}

GLFWindow::GLFWindow(int2 frame_size)
{
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    handle = glfwCreateWindow(frame_size.x, frame_size.y, "Foveated Image", NULL, NULL);
    if (!handle) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(handle, this);
    glfwMakeContextCurrent(handle);
    glfwSwapInterval(1);

    GLenum err = glewInit();

    fbSize = make_int2(frame_size.x, frame_size.y);

    compileAndLinkShader();             // Compiling and linking shaders
    generateAndbindbuffer();            // Generating and binding buffers
}

void GLFWindow::tabs() {

    if (ImGui::Checkbox("Foveated Rendering", &foveatedRendering_)) {
		foveatedRendering = !foveatedRendering;
        accumulateFrames = !accumulateFrames;
        if (!foveatedRendering) {
            aspectRatio = 1.0f;
        }
		renderParamsChanged = true;
	}
}

void GLFWindow::otherRenderingParams() {

    if (ImGui::CollapsingHeader("Environment Tone mapping")) {
        const char* options[] = { "Exposure", "Gamma", "Reinhard", "Filmic" };
        ImGui::Text("Choose a tone map method");
        if (ImGui::Combo("Options", &envToneMap, options, 4)) {
            renderParamsChanged = true;
        }

        // Handle selected option
        switch (envToneMap) {
        case 0:
            if (ImGui::DragFloat("Exposure Constant env", &exposureEnv, 0.1f, 0.1f, 10.0f)) {
                renderParamsChanged = true;
            }
            break;
        case 1:
            if (ImGui::DragFloat("Gamma Constant env", &gammaEnv, 0.1f, 0.1f, 10.0f)) {
                renderParamsChanged = true;
            }
            break;
        case 2:
            if (ImGui::DragFloat("Exposure Constant env", &exposureEnv, 0.1f, 0.1f, 10.0f)) {
                renderParamsChanged = true;
            }
            break;
        case 3:
            if (ImGui::DragFloat("Exposure Constant env", &exposureEnv, 0.1f, 0.1f, 10.0f)) {
                renderParamsChanged = true;
            }
            break;
        }
    }
    
    if (enablePhongShading)
    {
        if (ImGui::CollapsingHeader("Phong Shading Params"))
        {
            ImGui::Text("Phong gradient magnitude threshold");
            if (ImGui::DragFloat("Phong gradient magnitude threshold", &phongThreshold, 0.001f, 0.0f, 1.0f)) {
                renderParamsChanged = true;
            }
            ImGui::Separator();

            std::string s = "Ambient Constant";
            if (ImGui::DragFloat(s.c_str(), &ambientConstant, 0.1f, 0.0f, 1.0f))
            {
                renderParamsChanged = true;
            }
            ImGui::Separator();

            ImGui::Text("Attenuation constant A");
            if (ImGui::DragFloat("Attenuation constant A", &attenuationConstA, 0.1f, 0.0f, 1.0f)) {
                renderParamsChanged = true;
            }
            ImGui::Separator();
            
            ImGui::Text("Attenuation constant B");
            if (ImGui::DragFloat("Attenuation constant B", &attenuationConstB, 0.1f, 0.0f, 1.0f)) {
                renderParamsChanged = true;
            }
            ImGui::Separator();

            ImGui::Text("Attenuation constant C");
            if (ImGui::DragFloat("Attenuation constant C", &attenuationConstC, 0.1f, 0.0f, 1.0f)) {
				renderParamsChanged = true;
			}

            ImGui::Text("Shininess");
            if (ImGui::DragFloat("Shininess", &shininess, 1, 1, 200)) {
                renderParamsChanged = true;
            }
            ImGui::Separator();
        }
	}
}

void GLFWindow::renderImgui() {

    ImGui::Begin("Information");
    
    ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Frame Index : %d", frame_index);
    ImGui::Separator();
    ImGui::Text("Frame Size %d x %d", fbSize.x, fbSize.y);
    ImGui::Separator();

    if (renderGUI) { ImGui::End();  return; }

    tabs();
    ImGui::Separator();

    //otherRenderingParams();
    //ImGui::Separator();

    ImGui::End();

    if (foveatedRendering)
    {
        ImGui::Begin("Foveated Rendering Params");

        std::string s = "Focal length ";
        if (ImGui::DragFloat2(s.c_str(), &focalLength.x, 0.001f, 0.1f, 1.0f))
        {
            renderParamsChanged = true;
        }

        s = "Eye Position ";
        if (ImGui::DragFloat2(s.c_str(), &eyePos.x, 0.001f, 0.01f, 1.0f))
        {
            renderParamsChanged = true;
        }
        ImGui::Text("Eye pos: %f, %f", eyePos.x * fbSize.x, eyePos.y * fbSize.y);

        s = "Aspect raio of original frame and foveated render frame size ";
        if (ImGui::DragFloat(s.c_str(), &aspectRatio, 0.01f, 0.1f, 1.0f))
        {
            renderParamsChanged = true;
        }

        ImGui::End();
    }
}

void GLFWindow::draw(uint8_t* pixels)
{
    currentFrameTime = float(glfwGetTime());
    deltaTime = currentFrameTime - lastFrameTime;
    if (!glfwWindowShouldClose(handle)) {
		vec2i renderFrameSize = vec2i(int(float(fbSize.x) * aspectRatio), int(float(fbSize.y) * aspectRatio));

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        renderImgui();

        if (fbTexture == 0) {
			glGenTextures(1, &fbTexture);
		}

		glUseProgram(shaderProgram);

		glBindTexture(GL_TEXTURE_2D, fbTexture);
		GLenum texFormat = GL_RGBA;
		GLenum texelType = GL_UNSIGNED_BYTE;
		glTexImage2D(GL_TEXTURE_2D, 0, texFormat, renderFrameSize.x, renderFrameSize.y, 0, GL_RGBA, texelType, pixels);

		glDisable(GL_LIGHTING);
		glColor3f(1, 1, 1);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		setUniformShaderVariable();
		glUniform1i(glGetUniformLocation(shaderProgram, "textureSampler"), 0);

		glDisable(GL_DEPTH_TEST);
		glViewport(0, 0, fbSize.x, fbSize.y);

		glBindVertexArray(VAO);
		glDrawArrays(GL_QUADS, 0, 4);
		glBindVertexArray(0);
		
        ImGui::Render();
        ImGui::EndFrame();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(handle);
        glfwSwapInterval(0);
        glfwPollEvents();
	}
    else {
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		glfwDestroyWindow(handle);
		glfwTerminate();
	}
    lastFrameTime = currentFrameTime;
}

void GLFWindow::terminateGUI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(handle);
    glfwTerminate();
}

int2 GLFWindow::getframeSize(){
    return fbSize;
}

void GLFWindow::compileAndLinkShader() {

    // Load and compile vertex shader.
    std::string vertexShaderCode = readShaderCode("../shaders/vertex.glsl");
    const char* vertexShaderSource = vertexShaderCode.c_str();
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    if (vertexShader == 0) {
        std::cout << "Vertex shader compilation failed" << std::endl;
        exit(-1);
    }

    // Load and compile fragment shader.
    std::string fragmentShaderCode = readShaderCode("../shaders/fragment.glsl");
    const char* fragmentShaderSource = fragmentShaderCode.c_str();
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    if (fragmentShader == 0) {
        std::cout << "Fragment shader compilation failed" << std::endl;
        exit(-1);
    }

    // Link vertex and fragment shader into shader program and use it.
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check program linking status.
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader program linking error: " << infoLog << std::endl;
        exit(-1);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void GLFWindow::generateAndbindbuffer() {

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices) + sizeof(texCoords), nullptr, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(vertices), sizeof(texCoords), texCoords);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), static_cast<void*>(nullptr));
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(sizeof(vertices)));
    glEnableVertexAttribArray(1);

}

void GLFWindow::setUniformShaderVariable() {
    GLint useTextureLocation = glGetUniformLocation(shaderProgram, "foveatedRendering");
    GLint eyeLocation = glGetUniformLocation(shaderProgram, "eyePos");
    GLint foveaSizeLocation = glGetUniformLocation(shaderProgram, "focalLength");

    // Set the value of the uniform bool variable
    glUniform1i(useTextureLocation, foveatedRendering);

    // Set the value of the uniform vec3 variable
    glUniform2f(eyeLocation, eyePos.x, eyePos.y);

    // Set the value of the uniform float variable
    glUniform2f(foveaSizeLocation, focalLength.x, focalLength.y);

}

void GLFWindow::renderShaderOutput(uint8_t* pixels, vec2f eye, vec2i fovSize, std::string& outputFile) {

    if (!glfwWindowShouldClose(handle)) {

        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &fbTexture);
        glBindTexture(GL_TEXTURE_2D, fbTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fovSize.x, fovSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glDisable(GL_LIGHTING);
        glColor3f(1, 1, 1);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Framebuffer is not complete!" << std::endl;
            return;
        }

        glViewport(0, 0, fbSize.x, fbSize.y);
        glUseProgram(shaderProgram);

        glUniform1i(glGetUniformLocation(shaderProgram, "foveatedRendering"), true);
        glUniform2f(glGetUniformLocation(shaderProgram, "eyePos"), eye.x, eye.y);
        glUniform2f(glGetUniformLocation(shaderProgram, "focalLength"), 0.5f, 0.5f);
        glUniform1i(glGetUniformLocation(shaderProgram, "textureSampler"), 0);

        // Render to FBO
        glBindVertexArray(VAO);
        glDrawArrays(GL_QUADS, 0, 4);
        glBindVertexArray(0);

        std::vector<unsigned char> outputPixels(fbSize.x * fbSize.y * 4);
        glReadPixels(0, 0, fbSize.x, fbSize.y, GL_RGBA, GL_UNSIGNED_BYTE, outputPixels.data());

        glfwSwapBuffers(handle);
        glfwSwapInterval(0);
        glfwPollEvents();
    }
    else {
        glfwDestroyWindow(handle);
        glfwTerminate();
    }
}
