#pragma once

#include <iostream>
#include <vector>
#include "rapidcsv.h"
#include "cutil_math.h"

#include "optixPipeline.h"
#include "boundingBox.h"
#include "texture.h"
#include "camera.h"
#include "Model.h"
#include "transferFunction.h"

#include "LaunchParams.h"
#include "gdt/math/vec.h"
#include "gdt/gdt.h"
#include <fstream>
#include <iostream>
#include <string>

#include "dataStructs.h"
#include <omp.h>

#include "GLFWindow.h"
#include <GL/gl.h>

#include "glm/gtc/type_ptr.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/string_cast.hpp"

using namespace gdt;

enum renderingMethod {
    RayMarching,
    SingleScattering,
    MultipleScattering,
    NUM_RENDERING_METHODS
};

enum toneMapmethod {
    Exposure,
    Gamma,
    Reinhard,
    Filmic,
    NUM_TONEMAP_METHODS
};

class pathTracer
{
private:
    std::string                                 environmentMapPath = "../data/EnvMaps/Forest.png";
    std::string                                 segmenetFilePath;
    std::string                                 segmentationMeshesFolder;

    Texture									    *Textures;
    Model                                       *model;
    Optixpipeline                               optixPipeline;
    Camera                                      camera;
    LaunchParams                                launchParams;

    CUstream postProcessStream = nullptr;
    CUmodule    module;
    CUfunction  kernelFun;

    CUDABuffer colorBuffer;
    CUDABuffer yuvBuffer;
    CUDABuffer denoisedColorBuffer;
    CUDABuffer noisedColorBuffer;
    
    int2                                        renderFrameSize;            // Smaller buffer after first pass of foveated rendering
    int2                                        renderFrameSizeFull;        // Full buffer after second pass of foveated rendering

    bool transpose = false; 

    uint8_t*                                    framePixels = nullptr;

    CUDABuffer d_cryo;
    CUDABuffer d_bioscopeAlpha;
    CUDABuffer d_l1;
    CUDABuffer d_mappedArray;
    CUDABuffer d_segmentsToRender;

    std::vector<cudaArray_t>         textureArrays;

    CUDABuffer                                  deviceLights;

    float                                       initialVolumeSpacing;
    float                                       currentVolumeSpacing;
    int                                         totalBlocks;

public:
    pathTracer() { }
    pathTracer(Texture *textures_, Model *model_, std::string environmentMapPath, std::string segmentedFilePath);
    pathTracer(Model* model_);
    
    void initCamera();
    void initOptix();

    //void setLaunchCamera(int eye);
    void setLaunchCamera();
    void getAxisFromVector(float3& gaze, float3& up, float3& right);

    void postProcess();

    void createTextures();
   
    //void resetLaunchParams();
    bool isPNGFile(const std::string& filename);
    bool isEXRFile(const std::string& filename);

    void renderFrameResize();
    //void postProcessing(int eye);

    bool renderVolume();
    uint8_t* getFrame();
    void setCamera(float3 pos, float3 front, float3 up, float fov, float fDistance, float apertureRadius);
    void denoiseImage();

    void renderEyePos();

    int2 getEyePosition() { return make_int2(launchParams.eyePos.x * renderFrameSizeFull.x, launchParams.eyePos.y * renderFrameSizeFull.y); }

    void setframeIndex(int findex) { launchParams.frameIndex = findex; }
    LaunchParams getLaunchParams() { return launchParams; }
    int getframeIndex() { return launchParams.frameIndex; }
    float getaspectRatio() { return launchParams.fractionToRender; }
    void setaspectRatio(float aspect) { launchParams.fractionToRender = aspect; }
    int2 getframeSize() { return renderFrameSizeFull; }
    void setframeSize(int2 size) { renderFrameSizeFull = size; launchParams.frame.fullFrameSize = renderFrameSizeFull; }
    int2 getRenderFrameSize() { return renderFrameSize; }
    void setRenderFrameSize(int2 size) { renderFrameSize = size; }
    glm::vec3 getcameraPosition() { float3 pos = camera.getcamPos(); return glm::vec3(pos.x, pos.y, pos.z); }
    void setcameraPosition(float3 pos) { camera.setcamPos(pos); }
    void setgazeVector(float3 gaze) { camera.setgazeVector(gaze); }
    void setcameraFOV(float fov) { camera.setfovVertical(fov); }
    float getcameraFOV() { return camera.getfovVertical(); }
    void setcameraFocalLength(float focal) { camera.setfocalDistance(focal); }
    void setcameraApertureRadius(float aperture) { camera.setapertureRadius(aperture); }

    ~pathTracer();

    void setRenderingParams(GLFCameraWindow* frame);

};


