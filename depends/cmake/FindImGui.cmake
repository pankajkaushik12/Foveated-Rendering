
set(imgui $ENV{IMGUI_DIR})
cmake_path(SET imgui_ ${imgui})

set(IMGUI_SOURCES
    ${imgui_}/imgui.cpp
    ${imgui_}/imgui_draw.cpp
    ${imgui_}/imgui_tables.cpp
    ${imgui_}/imgui_widgets.cpp
    ${imgui_}/backends/imgui_impl_glfw.cpp
    ${imgui_}/backends/imgui_impl_opengl3.cpp
)

add_library(ImGUI STATIC ${IMGUI_SOURCES})

target_include_directories(ImGUI PUBLIC ${imgui} ${imgui}/backends)
