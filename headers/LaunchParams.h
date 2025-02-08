#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"
#include "curand.h"
#include "curand_kernel.h"
#include "dataStructs.h"

using namespace gdt;

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

struct LaunchParams
{
    int renderMode;
    struct {
        float4* noisedColorBuffer;
        float4* denoisedColorBuffer;
        uint8_t* colorBuffer;
        int2     size;
        int2 fullFrameSize;
    } frame;

    struct {
        float3 position;
        float3 horizontal;
        float3 vertical;
        float focalDistance;
        float apertureRadius;
        float4 viewMatrix[4];
        float4 invView[4];
        float4 invProj[4];
    } camPos;

    int uniqueSegmentCount;

    bool renderEyePos = false;

    float aspectRatio = 1.0;
    curandState_t *d_rand_state;

    int phasesamplingMethod=0;
    int lightSampleMethod=1;
    int envToneMap=3;
    float2 focalLength;
    float2 eyePos;
    float fractionToRender;
    bool foveatedRendering;
    
    float gammaEnv=2.2f;
    float exposureEnv=1.f;
    bool accumulateFrames = false;
    bool denoiseImage = false;

    bool enablePhongShading = false;
    bool enableHeadLight = true;
    float ambientConstant=0.3f;
    float attenuationConstA=1.f;
    float attenuationConstB=0.001f;
    float attenuationConstC=0.0f;
    float phongThreshold=0.0f;
    float shininess = 28.f;

    int samplePerPixel=1;
    int frameIndex = 0;

    float worldScale;

    int cudaStreamID;
    int totalCudaStreams;

    OptixTraversableHandle traversable;
};


