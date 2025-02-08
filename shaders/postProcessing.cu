#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include "LaunchParams.h"
#include "Utility.h"

using namespace gdt;

extern "C" __global__ void generatingFoveatedFrame (LaunchParams* params, cudaTextureObject_t texture, int width, int height) {

    int x_ = blockIdx.x * blockDim.x + threadIdx.x;
    int y_ = blockIdx.y * blockDim.y + threadIdx.y;
    vec2i pixelID = vec2i(x_, y_);

    if (pixelID.x >= width || pixelID.y >= height) return;

    float originWidth = width;
    float originHeight = height;
    int renderedFrameX = params->frame.size.x;
    int renderedFrameY = params->frame.size.y;

    float fx = params->focalLength.x;                                              //optixLaunchParams.camera.fovx;
    float fy = params->focalLength.y;                                                 //optixLaunchParams.camera.fovy;

    vec2f cursorPos = vec2f(params->eyePos.x, params->eyePos.y);

    float maxDxPos = 1.0 - cursorPos[0];
    float maxDyPos = 1.0 - cursorPos[1];
    float maxDxNeg = cursorPos[0];
    float maxDyNeg = cursorPos[1];

    float norDxPos = fx * maxDxPos / (fx + maxDxPos);
    float norDyPos = fy * maxDyPos / (fy + maxDyPos);
    float norDxNeg = fx * maxDxNeg / (fx + maxDxNeg);
    float norDyNeg = fy * maxDyNeg / (fy + maxDyNeg);

    vec2f tc = vec2f(float(pixelID.x) / originWidth, float(pixelID.y) / originHeight) - cursorPos;
    float x, y;

    if (tc.x >= 0) {
        x = fx * tc.x / (fx + tc.x); //>0
        x = x / norDxPos;
        x = x * maxDxPos + cursorPos.x;
    }
    else {
        x = fx * tc.x / (fx - tc.x); //<0
        x = x / norDxNeg;
        x = x * maxDxNeg + cursorPos.x;
    }

    if (tc.y >= 0) {
        y = fy * tc.y / (fy + tc.y);
        y = y / norDyPos;
        y = y * maxDyPos + cursorPos.y;
    }
    else {
        y = fy * tc.y / (fy - tc.y);
        y = y / norDyNeg;
        y = y * maxDyNeg + cursorPos.y;
    }


    //int TID = y_ * width + x_;

    uint32_t color = tex2D<uint32_t>(texture, x * renderedFrameX, y * renderedFrameY);

    // params->frame.finalColorBuffer[TID] = color;

}

// Function to convert float4* input image pointer to uint32_t* output image pointer
extern "C" __global__ void float4ToUint32(uint32_t *outColor, float4 *denoisedColor, int width, int height) {
	int x_ = blockIdx.x * blockDim.x + threadIdx.x;
	int y_ = blockIdx.y * blockDim.y + threadIdx.y;
	vec2i pixelID = vec2i(x_, y_);
	if (pixelID.x >= width || pixelID.y >= height) return;
	int TID = y_ * width + x_;
	float4 color = denoisedColor[TID];
	uint32_t r = (uint32_t)(color.x * 255.0f);
	uint32_t g = (uint32_t)(color.y * 255.0f);
	uint32_t b = (uint32_t)(color.z * 255.0f);
	uint32_t a = (uint32_t)(color.w * 255.0f);
	outColor[TID] = (a << 24) | (b << 16) | (g << 8) | r;
}
