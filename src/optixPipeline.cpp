#include "optixPipeline.h"

extern "C" char embedded_ptx_code[];

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
    //fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

Optixpipeline::~Optixpipeline()
{
    CUDA_SYNC_CHECK();
    CUresult cuRes;
    cuRes = cuCtxSetCurrent(cudaContext);
    if (cuRes != CUDA_SUCCESS) { printf("Error setting current context\n"); }

    CUDA_SYNC_CHECK();
    cuRes = cuCtxSynchronize();
    if (cuRes != CUDA_SUCCESS) { printf("Error synchronizing context\n"); }
    CUDA_SYNC_CHECK();

    // Release traversable handles
    if (asBuffer.d_ptr != nullptr) { cudaFree(asBuffer.d_ptr); asBuffer.d_ptr = nullptr; }
    if (raygenRecordsBuffer.d_ptr != nullptr) { cudaFree(raygenRecordsBuffer.d_ptr); raygenRecordsBuffer.d_ptr = nullptr; }
    if (missRecordsBuffer.d_ptr != nullptr) { cudaFree(missRecordsBuffer.d_ptr); missRecordsBuffer.d_ptr = nullptr; }
    if (hitgroupRecordsBuffer.d_ptr != nullptr) { cudaFree(hitgroupRecordsBuffer.d_ptr); hitgroupRecordsBuffer.d_ptr = nullptr; }
    CUDA_SYNC_CHECK();

    // Release launch buffers
    for (int stream = 0; stream < launchBuffers.size(); stream++)
    {
        if (launchBuffers[stream].d_ptr != nullptr) { cudaFree(launchBuffers[stream].d_ptr); launchBuffers[stream].d_ptr = nullptr; }
    }
    launchBuffers.clear();
    CUDA_SYNC_CHECK();


    if (denoiserScratch.d_ptr != nullptr) { cudaFree(denoiserScratch.d_ptr); denoiserScratch.d_ptr = nullptr; }
    if (denoiserState.d_ptr != nullptr) { cudaFree(denoiserState.d_ptr); denoiserState.d_ptr = nullptr; }
    CUDA_SYNC_CHECK();

    if (denoiser != nullptr) {
        OPTIX_CHECK(g_optixFunctionTable.optixDenoiserDestroy(denoiser));
    };
    CUDA_SYNC_CHECK();

    for (int stream = 0; stream < pipelines.size(); stream++)
    {
        if (pipelines[stream] != nullptr) {
            OPTIX_CHECK(g_optixFunctionTable.optixPipelineDestroy(pipelines[stream]));
        }
    }
    pipelines.clear();

    // Optix context
    for (int i = 0; i < optixContexts.size(); i++)
    {
        OPTIX_CHECK(g_optixFunctionTable.optixDeviceContextDestroy(optixContexts[i]));
    }

     //Releasing raygen, miss and hitgroup programs
    //for (int i = 0; i < raygenPGs.size(); i++) {
    //    OPTIX_CHECK(g_optixFunctionTable.optixProgramGroupDestroy(raygenPGs[i]));
    //}
    //for (int i = 0; i < missPGs.size(); i++) {
    //    OPTIX_CHECK(g_optixFunctionTable.optixProgramGroupDestroy(missPGs[i]));
    //}
    //for (int i = 0; i < hitgroupPGs.size(); i++)
    //{
    //    OPTIX_CHECK(g_optixFunctionTable.optixProgramGroupDestroy(hitgroupPGs[i]));
    //}
    raygenPGs.clear();
    missPGs.clear();
    hitgroupPGs.clear();

    // Releasing module
    OPTIX_CHECK(g_optixFunctionTable.optixModuleDestroy(module));

    // Releasing vertex, index and normal buffers
    for (int i = 0; i < vertexBuffers.size(); i++) {
        if (vertexBuffers[i].d_ptr != nullptr) { cudaFree(vertexBuffers[i].d_ptr); vertexBuffers[i].d_ptr = nullptr; }
    }
    CUDA_SYNC_CHECK();
    vertexBuffers.clear();
    for (int i = 0; i < indexBuffers.size(); i++) {
        if (indexBuffers[i].d_ptr != nullptr) { cudaFree(indexBuffers[i].d_ptr); indexBuffers[i].d_ptr = nullptr; }
    }
    CUDA_SYNC_CHECK();
    indexBuffers.clear();
    for (int i = 0; i < normalBuffers.size(); i++) {
        if (normalBuffers[i].d_ptr != nullptr) { cudaFree(normalBuffers[i].d_ptr); normalBuffers[i].d_ptr = nullptr; }
    }
    CUDA_SYNC_CHECK();
    normalBuffers.clear();   
    
    meshes.clear();

    // Release cuda streams
    for (int i = 0; i < streams.size(); i++)
    {
        if (streams[i] != nullptr)
        {
            cuRes = cuStreamDestroy(streams[i]);
            if (cuRes != CUDA_SUCCESS)
            {
                std::cerr << "Error destroying stream " << i << " : " << cuRes << std::endl;
                exit(2);
            }
        }
        if (kernelEvents[i] != nullptr) {
            CUresult res = cuEventDestroy(kernelEvents[i]);
            if (res != CUDA_SUCCESS) {
                std::cerr << "Error destroying event " << i << " : " << res << std::endl;
                exit(2);
            }
        }
    }
    streams.clear();
    kernelEvents.clear();
    CUDA_SYNC_CHECK();
}

void Optixpipeline::initOptix(Model* model_)
{
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error("No CUDA capable devices found!");
    //std::cout << "Found " << numDevices << " CUDA devices" << std::endl;

    void* handle;
    OptixResult res = optixInitHandle(&handle);
    if (res != OPTIX_SUCCESS)
    {
        std::cout << "Optix handle could not be generated\n";
        exit(2);
    }

    streams.resize(totalStreams);
    optixContexts.resize(totalStreams);
    pipelines.resize(totalStreams);
    launchBuffers.resize(totalStreams);
    kernelEvents.resize(totalStreams);

    model = model_;
    meshes = model->getMeshes();
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
}

void Optixpipeline::createContext(int stream) {

    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));

    //CUDA_CHECK(StreamCreate(&streams[stream]));
    cudaStreamCreateWithFlags(&streams[stream], cudaStreamNonBlocking);
    // check if the stream was created successfully
    if (streams[stream] == nullptr)
    {
		std::cout << "Stream " << stream << " could not be created\n";
		exit(2);
	}

    cudaEventCreateWithFlags(&kernelEvents[stream], cudaEventDisableTiming);

    cudaGetDeviceProperties(&deviceProps, deviceID);
    //std::cout << "Running on device: " << deviceProps.name << std::endl;

    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);


    OPTIX_CHECK(g_optixFunctionTable.optixDeviceContextCreate(cudaContext, &options, &optixContexts[stream]));
    OPTIX_CHECK(g_optixFunctionTable.optixDeviceContextSetLogCallback(optixContexts[stream], context_log_cb, nullptr, 4));

}

//CUstream Optixpipeline::getCudaStream() {
//    return stream;
//}

void Optixpipeline::createModule(int stream) {

    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    //moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 20;

    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK(g_optixFunctionTable.optixModuleCreate(optixContexts[stream],
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log, &sizeof_log,
        &module
    ));
    //if (sizeof_log > 1) PRINT(log);
}

void Optixpipeline::createRaygenProgram(int stream) {
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(g_optixFunctionTable.optixProgramGroupCreate(optixContexts[stream],
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &raygenPGs[0]
    ));
    //if (sizeof_log > 1) PRINT(log);
}

void Optixpipeline::createMissPrograms(int stream)
{
    // we do a single ray gen program in this example:
    missPGs.resize(RAY_TYPE_COUNT);

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;

    pgDesc.miss.entryFunctionName = "__miss__radiance";
    OPTIX_CHECK(g_optixFunctionTable.optixProgramGroupCreate(optixContexts[stream],
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[RADIANCE_RAY_TYPE]
    ));
    //if (sizeof_log > 1) PRINT(log);

    pgDesc.miss.entryFunctionName = "__miss__shadow";
    OPTIX_CHECK(g_optixFunctionTable.optixProgramGroupCreate(optixContexts[stream],
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &missPGs[SHADOW_RAY_TYPE]
    ));
}

void Optixpipeline::createHitgroupPrograms(int stream)
{
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(RAY_TYPE_COUNT);

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    
    OPTIX_CHECK(g_optixFunctionTable.optixProgramGroupCreate(optixContexts[stream],
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[RADIANCE_RAY_TYPE]
    ));
    //if (sizeof_log > 1) PRINT(log);

    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";

    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";


    OPTIX_CHECK(g_optixFunctionTable.optixProgramGroupCreate(optixContexts[stream],
        &pgDesc,
        1,
        &pgOptions,
        log, &sizeof_log,
        &hitgroupPGs[SHADOW_RAY_TYPE]
    ));
}

void Optixpipeline::createPipeline(int stream)
{
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(g_optixFunctionTable.optixPipelineCreate(optixContexts[stream],
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log, &sizeof_log,
        &pipelines[stream]
    ));
    //if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(g_optixFunctionTable.optixPipelineSetStackSize
    (/* [in] The pipeline to configure the stack size for */
        pipelines[stream],
        /* [in] The direct stack size requirement for direct
        callables invoked from IS or AH. */
        2 * 1024,
        /* [in] The direct stack size requirement for direct
        callables invoked from RG, MS, or CH.  */
        2 * 1024,
        /* [in] The continuation stack requirement. */
        2 * 1024,
        /* [in] The maximum depth of a traversable graph
        passed to trace. */
        1));
    //if (sizeof_log > 1) PRINT(log);
}

void Optixpipeline::buildSBTWithMeshes()
{
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        RaygenRecord rec;
        OPTIX_CHECK(g_optixFunctionTable.optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    if (raygenRecordsBuffer.d_ptr != 0) raygenRecordsBuffer.free();
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(g_optixFunctionTable.optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    if (missRecordsBuffer.d_ptr != 0) missRecordsBuffer.free();
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < numObjects; meshID++) {
        for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++) {
            auto mesh = meshes[meshID];

            HitgroupRecord rec;
            OPTIX_CHECK(g_optixFunctionTable.optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
            if (mesh->diffuseTextureID >= 0 && mesh->diffuseTextureID < model->getTextures().size()) {
                rec.data.hasTexture = true;
                rec.data.texture = model->getTextures()[mesh->diffuseTextureID]->textureObject;
            }
            else {
                rec.data.hasTexture = false;
            }
            rec.data.color = mesh->diffuse;
            rec.data.vertex = (float3*)vertexBuffers[meshID].d_pointer();
            rec.data.normal = (float3*)normalBuffers[meshID].d_pointer();
            rec.data.index = (int3*)indexBuffers[meshID].d_pointer();
            rec.data.texcoord = (float2*)texcoordBuffers[meshID].d_pointer();
            hitgroupRecords.push_back(rec);
        }
    }
    if (hitgroupRecordsBuffer.d_ptr != 0) hitgroupRecordsBuffer.free();
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();

    missRecords.clear();
    raygenRecords.clear();
    hitgroupRecords.clear();
}

OptixTraversableHandle Optixpipeline::buildAccelWithMeshes(int stream)
{
    const int numMeshes = (int)meshes.size();
    vertexBuffers.resize(numMeshes);
    indexBuffers.resize(numMeshes);
    normalBuffers.resize(numMeshes);
    texcoordBuffers.resize(numMeshes);

    // ==================================================================
    // triangle inputs
    // ==================================================================
    std::vector<OptixBuildInput> triangleInput(numMeshes);
    std::vector<CUdeviceptr> d_vertices(numMeshes);
    std::vector<CUdeviceptr> d_indices(numMeshes);
    std::vector<uint32_t> triangleInputFlags(numMeshes);

    TriangleMesh* mesh;
    for (int meshID = 0; meshID < numMeshes; meshID++) {
        // upload the model to the device: the builder
        mesh = meshes[meshID];
        
        if (vertexBuffers[meshID].d_ptr != nullptr) vertexBuffers[meshID].free();
        if (indexBuffers[meshID].d_ptr != nullptr) indexBuffers[meshID].free();
        if (normalBuffers[meshID].d_ptr != nullptr) normalBuffers[meshID].free();
        if (texcoordBuffers[meshID].d_ptr != nullptr) texcoordBuffers[meshID].free();

        vertexBuffers[meshID].alloc_and_upload(mesh->vertex);
        indexBuffers[meshID].alloc_and_upload(mesh->index);
        if (!mesh->texcoord.empty()) { texcoordBuffers[meshID].alloc_and_upload(mesh->texcoord); }
        if (!mesh->normal.empty()) { normalBuffers[meshID].alloc_and_upload(mesh->normal); }

        triangleInput[meshID] = {};
        triangleInput[meshID].type
            = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices[meshID] = vertexBuffers[meshID].d_pointer();
        d_indices[meshID] = indexBuffers[meshID].d_pointer();

        triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(float3);
        triangleInput[meshID].triangleArray.numVertices = (int)mesh->vertex.size();
        triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

        triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(int3);
        triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh->index.size();
        triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(g_optixFunctionTable.optixAccelComputeMemoryUsage(optixContexts[stream], &accelOptions, triangleInput.data(), (int)numMeshes, &blasBufferSizes));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    //if (outputBuffer.d_ptr != nullptr) outputBuffer.free(); // << the UNcompacted, temporary output buffer
    //outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
    cudaMalloc((void**)&outputBuffer, blasBufferSizes.outputSizeInBytes);
    CUDA_SYNC_CHECK();

    OPTIX_CHECK(g_optixFunctionTable.optixAccelBuild(optixContexts[stream], streams[stream], &accelOptions, triangleInput.data(), (int)numMeshes, tempBuffer.d_pointer(), tempBuffer.sizeInBytes, outputBuffer,
                                                            blasBufferSizes.outputSizeInBytes, &accelHandle, &emitDesc, 1 ));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    if (asBuffer.d_ptr != nullptr) asBuffer.free(); // << the UNcompacted, temporary output buffer
    asBuffer.alloc(compactedSize);

    OPTIX_CHECK(g_optixFunctionTable.optixAccelCompact(optixContexts[stream], streams[stream], accelHandle, asBuffer.d_pointer(), asBuffer.sizeInBytes, &accelHandle));
    CUDA_SYNC_CHECK();

    //outputBuffer.free(); // << the UNcompacted, temporary output buffer
    // clearing the outpur buffer
    cudaFree((void*)outputBuffer);
    tempBuffer.free();
    compactedSizeBuffer.free();

    mesh = nullptr;

    return accelHandle;
}

void Optixpipeline::allocateLaunchParamsBuffer() {
    for (int i = 0; i < totalStreams; i++) {
		launchBuffers[i].alloc(sizeof(LaunchParams));
	}
}

bool Optixpipeline::launchOptix(LaunchParams launchParams) {
    cudaEventRecord(startEvent, 0);
    launchParams.totalCudaStreams = totalStreams;
    if (totalStreams == 1)
    {
        launchParams.cudaStreamID = -1;
        cudaMemcpyAsync((void*)launchBuffers[0].d_pointer(), &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice, streams[0]);
        OPTIX_CHECK(g_optixFunctionTable.optixLaunch(/*! pipeline we're launching launch: */
            pipelines[0], streams[0],
            /*! parameters and SBT */
            launchBuffers[0].d_pointer(),
            launchBuffers[0].sizeInBytes,
            &sbt,
            /*! dimensions of the launch: */
            launchParams.frame.size.x,
            launchParams.frame.size.y,
            1
        ));
    }
    else if (totalStreams == 4){
        for (int i = 0; i < totalStreams - 1; i++)
        {
            launchParams.cudaStreamID = i;
            cudaMemcpyAsync((void*)launchBuffers[i].d_pointer(), &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice, streams[i]);
            OPTIX_CHECK(g_optixFunctionTable.optixLaunch(/*! pipeline we're launching launch: */
                pipelines[i], streams[i],
                /*! parameters and SBT */
                launchBuffers[i].d_pointer(),
                launchBuffers[i].sizeInBytes,
                &sbt,
                /*! dimensions of the launch: */
                launchParams.frame.size.x / 2,
                launchParams.frame.size.y / 2,
                1
            ));
            cudaEventRecord(kernelEvents[i], streams[i]);
            cudaStreamWaitEvent(streams[totalStreams - 1], kernelEvents[i], 0);
        }
        launchParams.cudaStreamID = totalStreams - 1;
        cudaMemcpyAsync((void*)launchBuffers[totalStreams - 1].d_pointer(), &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice, streams[totalStreams - 1]);
        OPTIX_CHECK(g_optixFunctionTable.optixLaunch(/*! pipeline we're launching launch: */
            pipelines[totalStreams - 1], streams[totalStreams - 1],
            /*! parameters and SBT */
            launchBuffers[totalStreams - 1].d_pointer(),
            launchBuffers[totalStreams - 1].sizeInBytes,
            &sbt,
            /*! dimensions of the launch: */
            launchParams.frame.size.x / 2,
            launchParams.frame.size.y / 2,
            1
        ));

    }
    else if (totalStreams == 2) {

        for (int i = 0; i < totalStreams - 1; i++)
        {
            launchParams.cudaStreamID = i;
            cudaMemcpyAsync((void*)launchBuffers[i].d_pointer(), &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice, streams[i]);
            OPTIX_CHECK(g_optixFunctionTable.optixLaunch(/*! pipeline we're launching launch: */
                pipelines[i], streams[i],
                /*! parameters and SBT */
                launchBuffers[i].d_pointer(),
                launchBuffers[i].sizeInBytes,
                &sbt,
                /*! dimensions of the launch: */
                launchParams.frame.size.x / 2,
                launchParams.frame.size.y,
                1
            ));
            cudaEventRecord(kernelEvents[i], streams[i]);
            cudaStreamWaitEvent(streams[totalStreams - 1], kernelEvents[i], 0);
        }
        launchParams.cudaStreamID = totalStreams - 1;
        cudaMemcpyAsync((void*)launchBuffers[totalStreams - 1].d_pointer(), &launchParams, sizeof(LaunchParams), cudaMemcpyHostToDevice, streams[totalStreams - 1]);
        OPTIX_CHECK(g_optixFunctionTable.optixLaunch(/*! pipeline we're launching launch: */
            pipelines[totalStreams - 1], streams[totalStreams - 1],
            /*! parameters and SBT */
            launchBuffers[totalStreams - 1].d_pointer(),
            launchBuffers[totalStreams - 1].sizeInBytes,
            &sbt,
            /*! dimensions of the launch: */
            launchParams.frame.size.x / 2,
            launchParams.frame.size.y,
            1
        ));
    }
    else {
        printf("Incorrect number of streams\n");
        exit(0);
    }
    cudaEventRecord(stopEvent, 0);
    return true;
}

void Optixpipeline::launchOptixParallel(LaunchParams launchParams, int stream) {
    launchBuffers[stream].upload(&launchParams, 1);
    OPTIX_CHECK(g_optixFunctionTable.optixLaunch(/*! pipeline we're launching launch: */
        pipelines[stream], streams[launchParams.cudaStreamID],
        /*! parameters and SBT */
        launchBuffers[stream].d_pointer(),
        launchBuffers[stream].sizeInBytes,
        &sbt,
        /*! dimensions of the launch: */
        launchParams.frame.size.x / 2,
        launchParams.frame.size.y / 2,
        1
    ));
}

void Optixpipeline::createDenoiser(LaunchParams launchParams, int stream) {
    if (streams.size() == 0) {
        return;         // No streams, no denoiser
	}

    if (denoiser != nullptr) {
        OPTIX_CHECK(g_optixFunctionTable.optixDenoiserDestroy(denoiser));
    };

    OptixDenoiserOptions denoiserOptions = {};

    OPTIX_CHECK(g_optixFunctionTable.optixDenoiserCreate(optixContexts[stream], OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &denoiser));

    OPTIX_CHECK(g_optixFunctionTable.optixDenoiserComputeMemoryResources(denoiser, launchParams.frame.size.x, launchParams.frame.size.y,
        &denoiserReturnSizes));

    denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
        denoiserReturnSizes.withoutOverlapScratchSizeInBytes));

    denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

    // ------------------------------------------------------------------
    OPTIX_CHECK(g_optixFunctionTable.optixDenoiserSetup(denoiser, streams[stream],
        launchParams.frame.size.x, launchParams.frame.size.y,
        denoiserState.d_pointer(),
        denoiserState.size(),
        denoiserScratch.d_pointer(),
        denoiserScratch.size()));

    denoiserParams.hdrIntensity = (CUdeviceptr)0;
    //cudaMalloc((void**)&denoiserParams.hdrIntensity, sizeof(float));
    //CUDA_SYNC_CHECK();
}

void Optixpipeline::denoiseImage(CUdeviceptr input_pointer, CUdeviceptr output_pointer, LaunchParams launchParams, int stream) {
    
    if (denoiser == nullptr) { createDenoiser(launchParams, 0); }

    if (launchParams.accumulateFrames)
        denoiserParams.blendFactor = 1.f / (launchParams.frameIndex);
    else
        denoiserParams.blendFactor = 0.0f;

    // -------------------------------------------------------
    OptixImage2D inputLayer;
    //inputLayer.data = noisedColorBuffer[eye].d_pointer();
    inputLayer.data = input_pointer;
    /// Width of the image (in pixels)
    inputLayer.width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer.height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    //outputLayer.data = denoisedColorBuffer[eye].d_pointer();
    outputLayer.data = output_pointer;
    /// Width of the image (in pixels)
    outputLayer.width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    outputLayer.height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    OptixDenoiserGuideLayer denoiserGuideLayer = {};

    OptixDenoiserLayer denoiserLayer = {};
    denoiserLayer.input = inputLayer;
    denoiserLayer.output = outputLayer;

    //OPTIX_CHECK(g_optixFunctionTable.optixDenoiserComputeIntensity(denoiser, streams[stream], &inputLayer, denoiserParams.hdrIntensity,
    //    denoiserScratch.d_pointer(), denoiserReturnSizes.computeIntensitySizeInBytes));
    //// check hdr intensity
    //float hdrIntensityValue;
    //cudaMemcpy(&hdrIntensityValue, (void*)denoiserParams.hdrIntensity, sizeof(float), cudaMemcpyDeviceToHost);
    //CUDA_SYNC_CHECK();
    //printf("HDR Intensity: %f\n", hdrIntensityValue);

    OPTIX_CHECK(g_optixFunctionTable.optixDenoiserInvoke(denoiser,
        streams[stream],
        &denoiserParams,
        denoiserState.d_pointer(),
        denoiserState.size(),
        &denoiserGuideLayer,
        &denoiserLayer, 1,
        /*inputOffsetX*/0,
        /*inputOffsetY*/0,
        denoiserScratch.d_pointer(),
        denoiserScratch.size()));

    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
}

OptixResult Optixpipeline::optixInitHandle(void** handlePtr)
{
    // Make sure these functions get initialized to zero in case the DLL and function
    // table can't be loaded
    g_optixFunctionTable.optixGetErrorName = 0;
    g_optixFunctionTable.optixGetErrorString = 0;

    if (!handlePtr)
        return OPTIX_ERROR_INVALID_VALUE;

#ifdef _WIN32
    * handlePtr = optixLoadWindowsDll();
    if (!*handlePtr)
        return OPTIX_ERROR_LIBRARY_NOT_FOUND;

    void* symbol = GetProcAddress((HMODULE)*handlePtr, "optixQueryFunctionTable");
    if (!symbol)
        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
#else
    * handlePtr = dlopen("libnvoptix.so.1", RTLD_NOW);
    if (!*handlePtr)
        return OPTIX_ERROR_LIBRARY_NOT_FOUND;

    void* symbol = dlsym(*handlePtr, "optixQueryFunctionTable");
    if (!symbol)
        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
#endif

    OptixQueryFunctionTable_t* optixQueryFunctionTable = (OptixQueryFunctionTable_t*)symbol;

    return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &g_optixFunctionTable, sizeof(g_optixFunctionTable));
}
