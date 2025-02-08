#pragma once

#include "LaunchParams.h"
#include "cudaBuffer.h"
#include "Model.h"

#include <thread>

#define totalStreams 4

class Optixpipeline
{
private:
    OptixTraversableHandle                      accelHandle{ 0 };
    std::vector<OptixDeviceContext>             optixContexts;
    CUcontext                                   cudaContext;
    std::vector<CUstream>                       streams;
    cudaDeviceProp                              deviceProps;
    OptixFunctionTable                          g_optixFunctionTable;
    OptixDeviceContextOptions                   options = {};
    
    OptixModule                                 module;
    OptixModuleCompileOptions                   moduleCompileOptions = {};
    std::vector<OptixPipeline>                  pipelines;
    OptixPipelineCompileOptions                 pipelineCompileOptions = {};
    OptixPipelineLinkOptions                    pipelineLinkOptions = {};

    OptixShaderBindingTable                     sbt = {};

    std::vector<OptixProgramGroup>              raygenPGs;
    std::vector<OptixProgramGroup>              missPGs;
    std::vector<OptixProgramGroup>              hitgroupPGs;

    OptixDenoiser                               denoiser;
    OptixDenoiserParams                         denoiserParams;
    CUDABuffer                                  denoiserScratch;
    OptixDenoiserSizes                          denoiserReturnSizes;
    CUDABuffer                                  denoiserState;
    std::vector<CUDABuffer>                     launchBuffers;
    Model*                                      model;

    std::vector<CUDABuffer> vertexBuffers;
    std::vector<CUDABuffer> indexBuffers;
    std::vector<CUDABuffer> normalBuffers;
    std::vector<CUDABuffer> texcoordBuffers;
    std::vector<int3> index;
    std::vector<float3> vertex;

    CUevent startEvent, stopEvent;
    std::vector<CUevent> kernelEvents;

    CUdeviceptr outputBuffer;

    CUDABuffer asBuffer;
    CUDABuffer raygenRecordsBuffer;
    CUDABuffer missRecordsBuffer;
    CUDABuffer hitgroupRecordsBuffer;

    std::vector<TriangleMesh*> meshes;

public:
    Optixpipeline() { denoiser = nullptr; };
    ~Optixpipeline();
    void initOptix(Model* model);
    OptixTraversableHandle buildAccel(int stream);
    OptixTraversableHandle buildAccelWithMeshes(int stream);
    //OptixTraversableHandle updateAccel(int stream, std::vector<vec3f> vertex, std::vector<vec3i> index);
    void createContext(int stream);
    void createModule(int stream);
    void createRaygenProgram(int stream);
    void createMissPrograms(int stream);
    void createHitgroupPrograms(int stream);
    void createPipeline(int stream);
    void allocateLaunchParamsBuffer();
    bool launchOptix(LaunchParams launchParams);
    void launchOptixParallel(LaunchParams launchParams, int stream);
    void buildSBTWithMeshes();
    void denoiseImage(CUdeviceptr input_pointer, CUdeviceptr output_pointer, LaunchParams launchParams, int stream);
    void createDenoiser(LaunchParams launchParams, int stream);
    //CUstream getCudaStream();
    OptixResult optixInitHandle(void** h_ptr);

};


struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    
    // data here
    void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    
    // data here
    void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    
    // data here
    //int objectID;
    TriangleMeshSBTData data;
};


