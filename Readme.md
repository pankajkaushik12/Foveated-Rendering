# Foveated Rendering with CUDA OptiX and OpenGL

This project implements **foveated rendering** based on the paper *"Rectangular Mapping-based Foveated Rendering"*. The rendering pipeline consists of two passes:

1. **First Pass (CUDA OptiX)** - Ray tracing is performed using NVIDIA OptiX.
2. **Second Pass (OpenGL Shader)** - The OptiX output is processed in OpenGL to display the final result.

## Features
- Efficient foveated rendering using a **rectangular mapping approach**
- Optimized for performance by reducing detail outside the fovea
- CUDA OptiX for high-quality ray tracing
- **Cursor-based foveation**: The foveal point follows the cursor in real-time

## Dependencies
- **CUDA** (OptiX)
- **OpenGL**
- **GLEW**
- **GLFW**

## Building and Running

1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install the dependencies (OpenGL, GLFW, GLEW)
3. Set the path to GLEW and GLFW as environment variable in GLEW_DIR & GLFW_DIR.
4. Use the CMake to build and generate the solution. Open the generated solution in Microsoft Visual Studio.
5. Run the project.

## Control
- Move the cursor to change the focus point. (Cursor point is treated a focus point).
- Enable the *Foveated rendering* and change the slider to adjust the size of the temp buffer.

## References
- *"Rectangular Mapping-based Foveated Rendering"* - [Paper Link](https://ieeexplore.ieee.org/document/9756831)

