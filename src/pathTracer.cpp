#include "pathTracer.h"
#include "cuda_texture_types.h"
#include "cuda_runtime.h"

pathTracer::pathTracer(Model* model_) : model(model_)
{
    createTextures();		  // Create textures for intensity volume, gradient volume and rgba volume

    initCamera();			  // Initialize camera with initial position, direction, field of view and frame size

    initOptix();
}

pathTracer::~pathTracer()
{
    cudaFree(launchParams.d_rand_state);
    launchParams.d_rand_state = nullptr;
    
    CUDA_SYNC_CHECK();

    if (deviceLights.d_ptr != nullptr) { cudaFree(deviceLights.d_ptr); deviceLights.d_ptr = nullptr; }

    CUDA_SYNC_CHECK();

    CUDA_SYNC_CHECK();

    if (colorBuffer.d_ptr != nullptr) { cudaFree(colorBuffer.d_ptr); colorBuffer.d_ptr = nullptr; }
    if (denoisedColorBuffer.d_ptr != nullptr) { cudaFree(denoisedColorBuffer.d_ptr); denoisedColorBuffer.d_ptr = nullptr; }
    if (noisedColorBuffer.d_ptr != nullptr) { cudaFree(noisedColorBuffer.d_ptr); noisedColorBuffer.d_ptr = nullptr; }

    CUDA_SYNC_CHECK();

    if (d_cryo.d_ptr != nullptr) { cudaFree(d_cryo.d_ptr); d_cryo.d_ptr = nullptr; }
    if (d_bioscopeAlpha.d_ptr != nullptr) { cudaFree(d_bioscopeAlpha.d_ptr); d_bioscopeAlpha.d_ptr = nullptr; }
    if (d_l1.d_ptr != nullptr) { cudaFree(d_l1.d_ptr); d_l1.d_ptr = nullptr; }
    if (d_mappedArray.d_ptr != nullptr) { cudaFree(d_mappedArray.d_ptr); d_mappedArray.d_ptr = nullptr; }
    if (d_segmentsToRender.d_ptr != nullptr) { cudaFree(d_segmentsToRender.d_ptr); d_segmentsToRender.d_ptr = nullptr; }

    CUDA_SYNC_CHECK();

    if (postProcessStream != nullptr) {
        cuModuleUnload(module);
		cudaStreamDestroy(postProcessStream);
		postProcessStream = nullptr;
	}
    CUDA_SYNC_CHECK();
}

bool pathTracer::renderVolume() {

    if (renderFrameSize.x <= 0 || renderFrameSize.y <= 0) return false;

    bool success = optixPipeline.launchOptix(launchParams);
    if (!success) {
        printf("Optix launch failed\n");
        return false;
    }
    CUDA_SYNC_CHECK();

    if (launchParams.denoiseImage) { denoiseImage(); }
    return true;
}

uint8_t* pathTracer::getFrame(){
    if (framePixels != NULL && colorBuffer.d_ptr != nullptr)
    {
        CUDA_SYNC_CHECK();
        cudaMemcpy(framePixels, colorBuffer.d_ptr, renderFrameSize.x * renderFrameSize.y * 4, cudaMemcpyDeviceToHost);
        CUDA_SYNC_CHECK();
        return framePixels;
    }
    return nullptr;
}

void pathTracer::postProcess() {
    auto startTime = std::chrono::high_resolution_clock::now();
    if (postProcessStream == nullptr)
    {
        cudaStreamCreate(&postProcessStream);
        cuModuleLoad(&module, "../build/cuda_compile_ptx_2_generated_postProcessing.cu.ptx");
        CUDA_SYNC_CHECK();
        CUresult result = cuModuleGetFunction(&kernelFun, module, "float4ToUint32");
        CUDA_SYNC_CHECK();
    }
    uint32_t* outputColor = (uint32_t*)launchParams.frame.colorBuffer;
    float4* denoisedColor = (float4*)launchParams.frame.denoisedColorBuffer;
        
    void* args[] = { &outputColor, &denoisedColor, &renderFrameSize.x, &renderFrameSize.y};
        
    int2 blockDim = make_int2(16, 16);
    const int gridDimX = (renderFrameSize.x + blockDim.x - 1) / blockDim.x;
    const int gridDimY = (renderFrameSize.y + blockDim.y - 1) / blockDim.y;
    cuLaunchKernel(kernelFun, gridDimX, gridDimY, 1, blockDim.x, blockDim.y, 1, 0, postProcessStream, args, 0);
    CUDA_SYNC_CHECK();
    cudaStreamSynchronize(postProcessStream);
    CUDA_SYNC_CHECK();
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    std::cout << "Time taken for post processing: " << elapsed.count() * 1000 << " ms" << std::endl;
}

void pathTracer::setLaunchCamera() {
    launchParams.camPos.apertureRadius = camera.getapertureRadius();
    launchParams.camPos.focalDistance = camera.getfocalDistance();

    launchParams.camPos.position = camera.getcamPos();

    glm::mat4 viewMatrix = camera.getViewMatrix();
    launchParams.camPos.horizontal = make_float3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]);
    launchParams.camPos.vertical = make_float3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);

    camera.getInverseViewMatrix(launchParams.camPos.invView, transpose);
    camera.getInverseProjectionMatrix(launchParams.camPos.invProj);
    camera.getViewMatrix(launchParams.camPos.viewMatrix);
}

void pathTracer::renderFrameResize(){
    if (framePixels != nullptr) {
		delete[] framePixels;
		framePixels = nullptr;
	}

    framePixels = new uint8_t[renderFrameSize.x * renderFrameSize.y * 4];
    
    noisedColorBuffer.resize(renderFrameSize.x * renderFrameSize.y * sizeof(float4));
    denoisedColorBuffer.resize(renderFrameSize.x * renderFrameSize.y * sizeof(float4));
    colorBuffer.resize(renderFrameSize.x * renderFrameSize.y * sizeof(uint8_t) * 4);
    yuvBuffer.resize(renderFrameSize.x * renderFrameSize.y * sizeof(uint8_t) * 3);

    launchParams.frame.size = renderFrameSize;
    launchParams.frame.fullFrameSize = renderFrameSizeFull;
    launchParams.frame.colorBuffer = (uint8_t*)colorBuffer.d_pointer();
    launchParams.frame.noisedColorBuffer = (float4*)noisedColorBuffer.d_pointer();
    launchParams.frame.denoisedColorBuffer = (float4*)denoisedColorBuffer.d_pointer();

    cudaMalloc(&launchParams.d_rand_state, sizeof(curandState_t));
}

// Setting rendering parameters in debug mode
void pathTracer::setRenderingParams(GLFCameraWindow* frame)
{
    launchParams.renderMode = 0;
    launchParams.accumulateFrames = frame->accumulateFrames;
    launchParams.denoiseImage = frame->denoiseImage;

    launchParams.gammaEnv = frame->gammaEnv;
    launchParams.exposureEnv = frame->exposureEnv;

    launchParams.enablePhongShading = frame->enablePhongShading;
    launchParams.shininess = frame->shininess;
    launchParams.envToneMap = frame->envToneMap;

    launchParams.ambientConstant = frame->ambientConstant;
    launchParams.enableHeadLight = frame->enableHeadLight;
    launchParams.attenuationConstA = frame->attenuationConstA;
    launchParams.attenuationConstB = frame->attenuationConstB;
    launchParams.attenuationConstC = frame->attenuationConstC;
    launchParams.phongThreshold = frame->phongThreshold;

    launchParams.foveatedRendering = frame->foveatedRendering;
    launchParams.eyePos = frame->eyePos;
    launchParams.focalLength = frame->focalLength;
}

void pathTracer::setCamera(float3 pos, float3 front, float3 up, float fov, float fDistance, float apertureRadius) {
    float3 gaze, right, at;

	at = pos + 10.f * front;
    gaze = normalize(front);                               // origin - from, always looks at origin
    getAxisFromVector(gaze, up, right);

    camera.setcamPos(pos);
    camera.setgazeVector(gaze);
    camera.setupVector(up);
    camera.setrightVector(right);

    camera.setapertureRadius(apertureRadius);
    camera.setfocalDistance(fDistance);
    camera.setfovVertical(fov);

    transpose = false;
    float aspectRatio = launchParams.frame.size.y == 0 ? 1.0f : (float)launchParams.frame.size.x / (float)launchParams.frame.size.y;
    camera.setViewMatrix(glm::lookAt(glm::vec3(pos.x, pos.y, pos.z), glm::vec3(at.x, at.y, at.z), glm::vec3(up.x, up.y, up.z)));
    camera.setProjectionMatrix(glm::perspective(glm::radians(fov), aspectRatio, 0.1f, 50.0f));

    setLaunchCamera();
}

void pathTracer::renderEyePos()
{
    launchParams.renderEyePos = true;
}

void pathTracer::createTextures() {
	
    Textures->createTexture(model, textureArrays);
}

void pathTracer::initCamera() {
    //float3 position = make_float3(0, 0, - VolumeDimension.z / 2.f - 200.f);
    float3 position = make_float3(-1293.07f, 154.681f, -0.7304f);
    float3 gaze = normalize(make_float3(0, 0, 0) - position);
    float fov = 90.f;
    int2 size = make_int2(1920, 1080);

    setCamera(position, make_float3(0, 0, 0), make_float3(0, 1, 0), fov, 1.f, 0.f);
    camera.setframeSize(size);

    renderFrameSizeFull = camera.getframeSize();
    launchParams.fractionToRender = 1.f;
    renderFrameSize = renderFrameSizeFull;
    launchParams.aspectRatio = (float)renderFrameSizeFull.x / (float)renderFrameSizeFull.y;
    renderFrameResize();
}

void pathTracer::initOptix() {
    optixPipeline.initOptix(model);

    for (int i = 0; i < totalStreams; i++) {
        optixPipeline.createContext(i);

        //std::cout << "#osc: setting up module ..." << std::endl;
        optixPipeline.createModule(i);

        //std::cout << "#osc: creating raygen programs ..." << std::endl;
        optixPipeline.createRaygenProgram(i);

        //std::cout << "#osc: creating miss programs ..." << std::endl;
        optixPipeline.createMissPrograms(i);

        //std::cout << "#osc: creating hitgroup programs ..." << std::endl;
        optixPipeline.createHitgroupPrograms(i);

        launchParams.traversable = optixPipeline.buildAccelWithMeshes(i);

        //std::cout << "#osc: setting up optix pipeline ..." << std::endl;
        optixPipeline.createPipeline(i);
    }

    optixPipeline.buildSBTWithMeshes();

    optixPipeline.allocateLaunchParamsBuffer();
}

void pathTracer::denoiseImage() {
    optixPipeline.denoiseImage(noisedColorBuffer.d_pointer(), denoisedColorBuffer.d_pointer(), launchParams, 0);
}

void pathTracer::getAxisFromVector(float3& gaze, float3& up, float3& right) {
    up = make_float3(0, 1, 0);
    gaze = normalize(gaze);
    right = normalize(cross(gaze, up));
    up = normalize(cross(right, gaze));
}

bool pathTracer::isPNGFile(const std::string& fileName) {
    size_t dotPosition = fileName.find_last_of('.');
    if (dotPosition != std::string::npos) {
        std::string extension = fileName.substr(dotPosition + 1);
        return extension == "png" || extension == "PNG";
    }
    return false;
}

bool pathTracer::isEXRFile(const std::string& fileName) {
    size_t dotPosition = fileName.find_last_of('.');
    if (dotPosition != std::string::npos) {
        std::string extension = fileName.substr(dotPosition + 1);
        return extension == "exr" || extension == "EXR";
    }
    return false;
}
