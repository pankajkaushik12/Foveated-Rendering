#include "gdt/math/vec.h"
#include "imageProcessing.h"
#include <random>
#include <chrono>

#include "GLFWindow.h"
#include "pathTracer.h"
#include "Model.h"

void startRendering(Model& model) {

    pathTracer pTracer(&model);


    GLFCameraWindow* frame;

    frame = new GLFCameraWindow("Path Tracing", pTracer.getcameraPosition(), glm::vec3(0.f, 0.f, 0.f), glm::vec3(.0f, 1.f, .0f), pTracer.getframeSize(), model.getModelSpan());

	uint8_t* pixels = nullptr;
	ImageProcessing imageProcessing;
    std::string savedImage_folder = "";

    while (!frame->CloseApplication) {
        if (frame->renderParamsChanged) {

            if (pTracer.getframeSize() != frame->getframeSize() || pTracer.getaspectRatio() != frame->aspectRatio)
            {
                pTracer.setaspectRatio(frame->aspectRatio);
                pTracer.setframeSize(frame->getframeSize());
                pTracer.setRenderFrameSize(make_int2(frame->getframeSize().x * frame->aspectRatio, frame->getframeSize().y * frame->aspectRatio));
                pTracer.renderFrameResize();
            }

            pTracer.setCamera(frame->cameraFrame.get_from(), frame->cameraFrame.get_front(), frame->cameraFrame.get_up(), frame->cameraFOV, frame->cameraFocalLength, frame->cameraApertureRadius);
            frame->renderParamsChanged = false;
            pTracer.setframeIndex(0);

            frame->cameraPosition = pTracer.getcameraPosition();
        }

        pTracer.setframeIndex(pTracer.getframeIndex() + 1);
        frame->frame_index = pTracer.getframeIndex();

        if (glfwWindowShouldClose(frame->handle)) break;

        pTracer.setRenderingParams(frame);

        pTracer.renderVolume();

        int2 frameSize = pTracer.getframeSize();
        pixels = pTracer.getFrame();

        if (frame->keyPressed == 'S') {
            std::string path = savedImage_folder + std::to_string(pTracer.getframeIndex()) + ".png";
            imageProcessing.saveImage(pixels, path, frameSize.x, frameSize.y, 3);
            frame->keyPressed = 0;
        }

        frame->draw(pixels);
    }
}

void startApp(const std::string& modelPath) {

    auto start = std::chrono::high_resolution_clock::now();
    Model model(modelPath);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Loading time: " << duration.count() << "ms" << std::endl;

    startRendering(model);
}

int main(int argc, char** argv) {
    
    const std::string modelPath = "../data/sponza/sponza.obj";
    startApp(modelPath);

    return 0;
}

