cmake_minimum_required(VERSION 3.5)

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif (POLICY CMP0077)

project(FoveatedRendering)

set(executable_name ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set (CUDA_PROPAGATE_HOST_FLAGS ON)

set(gdt_dir ${PROJECT_SOURCE_DIR}/depends/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/depends/cmake/")
include("configure_build_type")
include("configure_optix")

find_package(CUDAToolkit 11.7 REQUIRED)
find_package(ImGui REQUIRED)
mark_as_advanced(CUDA_SDK_ROOT_DIR)

include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/depends/tinyply")


set(OpenGL_GL_PREFERENCE LEGACY)

find_package(OpenGL REQUIRED)
set(glew $ENV{GLEW_DIR})
set(glfw $ENV{GLFW_DIR})

include_directories( 
    headers/
    depends/
    ${glew}/include 
    ${glfw}/include
    $ENV{GLM_DIR}/
    ${OptiX_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/depends/stb
)

set(SOURCES
    src/main.cpp
    src/boundingBox.cpp
    src/pathTracer.cpp
    src/launchParams.cpp
    src/cudaBuffer.cpp
    src/camera.cpp
    src/texture.cpp
    src/optixPipeline.cpp
    src/imageProcessing.cpp
    src/transferFunction.cpp
    src/Model.cpp
    src/GLFWindow.cpp
)

set(SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/devicePrograms.cu
)

cuda_compile_and_embed(embedded_ptx_code ${CMAKE_CURRENT_SOURCE_DIR}/shaders/devicePrograms.cu)

add_executable("${executable_name}"
    ${embedded_ptx_code}
    ${SOURCES}
  )

cmake_path(SET glfw_lib_dir ${glfw}/lib-vc2022)
cmake_path(SET glew_lib_dir ${glew}/lib/Release/x64)

target_link_directories("${executable_name}" PUBLIC 
    $ENV{OpenEXR_DIR}/lib
    "${glfw_lib_dir}" 
    "${glew_lib_dir}"
)

target_link_libraries("${executable_name}" PRIVATE
    ImGUI
    gdt
    tinyply
    glfw3
    glew32s
    ${OPENGL_gl_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
)

