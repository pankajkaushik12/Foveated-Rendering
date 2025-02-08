#include "texture.h"

void Texture::createTexture(Model* model, std::vector<cudaArray_t>& textureArrays)
{
    std::vector<TextureMap*> textures = model->getTextures();
    int numTextures = (int)textures.size();

    textureArrays.resize(numTextures);

    for (int textureID = 0; textureID < numTextures; textureID++) {
        auto texture = textures[textureID];

        cudaResourceDesc res_desc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t width = texture->width;
        int32_t height = texture->height;
        int32_t numComponents = 4;
        int32_t pitch = width * numComponents * sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();

        cudaArray_t& pixelArray = textureArrays[textureID];
        CUDA_CHECK(MallocArray(&pixelArray,
            &channel_desc,
            width, height));

        CUDA_CHECK(Memcpy2DToArray(pixelArray,
            /* offset */0, 0,
            texture->pixel,
            pitch, pitch, height,
            cudaMemcpyHostToDevice));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        // Create texture object
        CUDA_CHECK(CreateTextureObject(&texture->textureObject, &res_desc, &tex_desc, nullptr));
    }
}

cudaTextureObject_t Texture::create2Dtexture(unsigned char * image_data, int width, int height, int channel) {
    cudaArray* cuda_image_array;
    cudaChannelFormatDesc channel_desc;
    channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsignedNormalized8X4);

    cudaMallocArray(&cuda_image_array, &channel_desc, width, height);
    CUDA_SYNC_CHECK();

    cudaMemcpy2DToArray(cuda_image_array, 0, 0, image_data, width * sizeof(image_data[0]) * 4, width * sizeof(image_data[0]) * 4, height, cudaMemcpyHostToDevice);
    CUDA_SYNC_CHECK();
    
    cudaResourceDesc resource_desc;
    memset(&resource_desc, 0, sizeof(resource_desc));
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = cuda_image_array;

    cudaTextureDesc texture_desc;
    memset(&texture_desc, 0, sizeof(texture_desc));
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeNormalizedFloat;

    cudaCreateTextureObject(&textureObject, &resource_desc, &texture_desc, nullptr);
    CUDA_SYNC_CHECK();

    return textureObject;
}

cudaTextureObject_t Texture::create2Dtexture(float4* d_hdrMap, int width, int height, int channels) {
    cudaTextureObject_t textureObject;

    cudaArray* cuda_image_array;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&cuda_image_array, &channel_desc, width, height);
    CUDA_SYNC_CHECK();

    cudaMemcpy2DToArray(cuda_image_array, 0, 0, d_hdrMap, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyDeviceToDevice);
    CUDA_SYNC_CHECK();

    cudaResourceDesc resource_desc;
    memset(&resource_desc, 0, sizeof(resource_desc));
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = cuda_image_array;

    cudaTextureDesc texture_desc;
    memset(&texture_desc, 0, sizeof(texture_desc));
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&textureObject, &resource_desc, &texture_desc, nullptr);
    CUDA_SYNC_CHECK();

    return textureObject;
}

cudaTextureObject_t Texture::create3Dtexture(cudaArray_t cuda_image_array, cudaMemcpy3DParms &copyParams, cudaTextureDesc &textureDesc) {

    copyParams.dstArray = cuda_image_array;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    CUDA_SYNC_CHECK();

    cudaResourceDesc resource_desc;
    memset(&resource_desc, 0, sizeof(resource_desc));
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = cuda_image_array;
    CUDA_SYNC_CHECK();

    //textureDesc.addressMode[0] = cudaAddressModeBorder;
    //textureDesc.addressMode[1] = cudaAddressModeBorder;
    //textureDesc.addressMode[2] = cudaAddressModeBorder;
    textureDesc.addressMode[0] = cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaAddressModeClamp;
    textureDesc.addressMode[2] = cudaAddressModeClamp;

    cudaCreateTextureObject(&textureObject, &resource_desc, &textureDesc, nullptr);
    CUDA_SYNC_CHECK();

    return textureObject;
}

cudaTextureObject_t Texture::getTexturePtr(){
    return textureObject;
}

cudaTextureObject_t Texture::createIntensityTexture(float* hostData, float3 size, unsigned int channels) {
    int width = (int)size.x;
    int height = (int)size.y;
    int depth = (int)size.z;

    int numElements = int(width * height * depth * channels);

    float* deviceData;
    cudaMalloc(&deviceData, numElements * sizeof(float));
    cudaMemcpy(deviceData, hostData, numElements * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_SYNC_CHECK();

    cudaArray_t cuda_image_array;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
    CUDA_SYNC_CHECK();

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(deviceData, width * sizeof(float), width, height);
    copyParams.extent = make_cudaExtent(width, height, depth);

    cudaTextureDesc textureDesc{};
    memset(&textureDesc, 0, sizeof(textureDesc));
    textureDesc.filterMode = cudaFilterModeLinear;

    return create3Dtexture(cuda_image_array, copyParams, textureDesc);
}

cudaTextureObject_t Texture::createIntensityTexture(unsigned char* hostData, float3 size, unsigned int channels) {
    int width = (int)size.x;
    int height = (int)size.y;
    int depth = (int)size.z;

    int numElements = int(width * height * depth * channels);
    unsigned char* d_intensityBuffer = nullptr;

    cudaMalloc(&d_intensityBuffer, numElements * sizeof(unsigned char));
    cudaMemcpy(d_intensityBuffer, hostData, numElements * sizeof(unsigned char), cudaMemcpyHostToDevice);
    CUDA_SYNC_CHECK();

    cudaArray* cuda_image_array;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<unsigned char>();
    cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
    CUDA_SYNC_CHECK();

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr = make_cudaPitchedPtr(d_intensityBuffer, width * sizeof(unsigned char), width, height);
    copyParams.extent = make_cudaExtent(width, height, depth);

    cudaTextureDesc textureDesc{};
    memset(&textureDesc, 0, sizeof(textureDesc));
    textureDesc.filterMode = cudaFilterModeLinear;
    textureDesc.readMode = cudaReadModeNormalizedFloat;

    return create3Dtexture(cuda_image_array, copyParams, textureDesc);
}

cudaTextureObject_t Texture::createSegmentationTexture(unsigned short* hostData, float3 size, unsigned int channels) {
    int width = (int)size.x;
    int height = (int)size.y;
    int depth = (int)size.z;

    int numElements = int(width * height * depth * channels);
    bool useFloat = false;

    cudaArray* cuda_image_array;
    cudaChannelFormatDesc channel_desc;
    cudaMemcpy3DParms copyParams = { 0 };
    cudaTextureDesc textureDesc{};

    if (!useFloat) {
        cudaMalloc(&d_segmentationBuffer, numElements * sizeof(unsigned short));
        cudaMemcpy(d_segmentationBuffer, hostData, numElements * sizeof(unsigned short), cudaMemcpyHostToDevice);
        CUDA_SYNC_CHECK();

        channel_desc = cudaCreateChannelDesc<unsigned short>();
        cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
        CUDA_SYNC_CHECK();

        copyParams.srcPtr = make_cudaPitchedPtr(d_segmentationBuffer, width * sizeof(unsigned short), width, height);
        copyParams.extent = make_cudaExtent(width, height, depth);

        memset(&textureDesc, 0, sizeof(textureDesc));
        textureDesc.filterMode = cudaFilterModePoint;
        textureDesc.readMode = cudaReadModeElementType;
    }
    else {
        float *hostDataFloat = new float[numElements];
        for (int i = 0; i < numElements; i++) {
            hostDataFloat[i] = static_cast<float>(hostData[i]);
		}

        float *deviceData;
        cudaMalloc(&deviceData, numElements * sizeof(float));
        cudaMemcpy(deviceData, hostDataFloat, numElements * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_SYNC_CHECK();

        channel_desc = cudaCreateChannelDesc<float>();
        cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
        CUDA_SYNC_CHECK();

        copyParams.srcPtr = make_cudaPitchedPtr(deviceData, width * sizeof(float), width, height);
        copyParams.extent = make_cudaExtent(width, height, depth);

        memset(&textureDesc, 0, sizeof(textureDesc));
        textureDesc.filterMode = cudaFilterModeLinear;
        textureDesc.readMode = cudaReadModeElementType;
    }

    return create3Dtexture(cuda_image_array, copyParams, textureDesc);
}

cudaTextureObject_t Texture::createColorTexture(unsigned char* hostData, float3 size, unsigned int channels) {
    int width = int(size.x);
    int height = int(size.y);
    int depth = int(size.z);

    uint64_t numElements =  uint64_t(width) * uint64_t(height) * uint64_t(depth);
    bool useFloat = false;

    cudaArray* cuda_image_array;
    cudaChannelFormatDesc channel_desc;
    cudaMemcpy3DParms copyParams = { 0 };
    cudaTextureDesc textureDesc{};

    if (d_rgbaBuffer != nullptr) {
		cudaFree(d_rgbaBuffer);
        CUDA_SYNC_CHECK();
	}

    if (useFloat)
    {
        float4* rgbaData = new float4[numElements];
        for (int i = 0; i < numElements; i++) {
            rgbaData[i].x = hostData[i * 3 + 0] / 255.f;
            rgbaData[i].y = hostData[i * 3 + 1] / 255.f;
            rgbaData[i].z = hostData[i * 3 + 2] / 255.f;
            rgbaData[i].w = 1.f;
        }

        float4* deviceData;
        cudaMalloc(&deviceData, numElements * sizeof(float4));
        cudaMemcpy(deviceData, rgbaData, numElements * sizeof(float4), cudaMemcpyHostToDevice);
        CUDA_SYNC_CHECK();

        
        channel_desc = cudaCreateChannelDesc<float4>();
        cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
        CUDA_SYNC_CHECK();

        copyParams.srcPtr = make_cudaPitchedPtr(deviceData, width * sizeof(float4), width, height);
        copyParams.extent = make_cudaExtent(width, height, depth);

        memset(&textureDesc, 0, sizeof(textureDesc));
        textureDesc.filterMode = cudaFilterModeLinear;
    }
    else {
        uchar4* rgbaData = new uchar4[numElements];
        #pragma omp parallel for
        for (int i = 0; i < numElements; i++) {
            if (channels == 3)
            {
                rgbaData[i].x = hostData[i * 3 + 0];
                rgbaData[i].y = hostData[i * 3 + 1];
                rgbaData[i].z = hostData[i * 3 + 2];
                rgbaData[i].w = (unsigned char)255;
            }
            else {
                rgbaData[i].x = hostData[i * 4 + 0];
                rgbaData[i].y = hostData[i * 4 + 1];
                rgbaData[i].z = hostData[i * 4 + 2];
                rgbaData[i].w = hostData[i * 4 + 3];
            }
        }

        cudaMalloc(&d_rgbaBuffer, numElements * sizeof(uchar4));
        cudaMemcpy(d_rgbaBuffer, rgbaData, numElements * sizeof(uchar4), cudaMemcpyHostToDevice);
        CUDA_SYNC_CHECK();

        channel_desc = cudaCreateChannelDesc<uchar4>();
        cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
        CUDA_SYNC_CHECK();

        copyParams.srcPtr = make_cudaPitchedPtr(d_rgbaBuffer, width * sizeof(uchar4), width, height);
        copyParams.extent = make_cudaExtent(width, height, depth);

        memset(&textureDesc, 0, sizeof(textureDesc));
        textureDesc.filterMode = cudaFilterModeLinear;
        textureDesc.readMode = cudaReadModeNormalizedFloat;
        //textureDesc.filterMode = cudaFilterModePoint;                             //Change tex3D<uchar4> in kernerl and also normalize explicitly
        //textureDesc.readMode = cudaReadModeElementType;

        rgbaData = nullptr;
    }

    return create3Dtexture(cuda_image_array, copyParams, textureDesc);
}

cudaTextureObject_t Texture::createOpacityTexture(float* opacityrray, float3 size, unsigned int channels) {
    int width = (int)size.x;
    int height = (int)size.y;
    int depth = (int)size.z;

    int numElements = int(width * height * depth);

    if (d_opacityBuffer != nullptr) {
        cudaFree(d_opacityBuffer);
        CUDA_SYNC_CHECK();
    }

    cudaMalloc(&d_opacityBuffer, numElements * sizeof(float));
    cudaMemcpy(d_opacityBuffer, opacityrray, numElements * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_SYNC_CHECK();

    cudaArray* cuda_image_array = nullptr;
    cudaChannelFormatDesc channel_desc;
    cudaMemcpy3DParms copyParams = { 0 };
    cudaTextureDesc textureDesc{};

    channel_desc = cudaCreateChannelDesc<float>();
    if (cuda_image_array != nullptr) {
		cudaFreeArray(cuda_image_array);
		CUDA_SYNC_CHECK();
	}
    cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
    CUDA_SYNC_CHECK();

    copyParams.srcPtr = make_cudaPitchedPtr(d_opacityBuffer, width * sizeof(float), width, height);
    copyParams.extent = make_cudaExtent(width, height, depth);
    
    memset(&textureDesc, 0, sizeof(textureDesc));
    textureDesc.filterMode = cudaFilterModeLinear;
    textureDesc.readMode = cudaReadModeElementType;

    return create3Dtexture(cuda_image_array, copyParams, textureDesc);
}

cudaTextureObject_t Texture::createGradientTexture(float* gradientVolume, float3 size, unsigned int channels) {
    int width = (int)size.x;
    int height = (int)size.y;
    int depth = (int)size.z;

    int numElements = int(width * height * depth);
    bool useFloat = true;

    cudaArray* cuda_image_array;
    cudaChannelFormatDesc channel_desc;
    cudaMemcpy3DParms copyParams = { 0 };
    cudaTextureDesc textureDesc{};

    if (useFloat) {
        float4* hostData = new float4[numElements];
        #pragma omp parallel for
        for (int i = 0; i < numElements; i++) {
            hostData[i].x = gradientVolume[i * 3 + 0];
            hostData[i].y = gradientVolume[i * 3 + 1];
            hostData[i].z = gradientVolume[i * 3 + 2];
            hostData[i].w = 1.0f;
        }

        cudaMalloc(&d_gradientBuffer, numElements * sizeof(float4));
        cudaMemcpy(d_gradientBuffer, hostData, numElements * sizeof(float4), cudaMemcpyHostToDevice);
        CUDA_SYNC_CHECK();
        
        channel_desc = cudaCreateChannelDesc<float4>();
        cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
        CUDA_SYNC_CHECK();

        copyParams.srcPtr = make_cudaPitchedPtr(d_gradientBuffer, width * sizeof(float4), width, height);
        copyParams.extent = make_cudaExtent(width, height, depth);

        memset(&textureDesc, 0, sizeof(textureDesc));
        textureDesc.filterMode = cudaFilterModeLinear;
        //textureDesc.readMode = cudaReadModeNormalizedFloat;
    }
    else {
        uchar4* hostData = new uchar4[numElements];
        for (int i = 0; i < numElements; i++) {
            hostData[i].x = (unsigned char)(gradientVolume[i * 3 + 0] * 255.f);
            hostData[i].y = (unsigned char)(gradientVolume[i * 3 + 1] * 255.f);
            hostData[i].z = (unsigned char)(gradientVolume[i * 3 + 2] * 255.f);
            hostData[i].w = (unsigned char)0;
        }

        uchar4* deviceData;
        cudaMalloc(&deviceData, numElements * sizeof(uchar4));
        cudaMemcpy(deviceData, hostData, numElements * sizeof(uchar4), cudaMemcpyHostToDevice);
        CUDA_SYNC_CHECK();

        channel_desc = cudaCreateChannelDesc<uchar4>();
        cudaMalloc3DArray(&cuda_image_array, &channel_desc, make_cudaExtent(width, height, depth));
        CUDA_SYNC_CHECK();

        copyParams.srcPtr = make_cudaPitchedPtr(deviceData, width * sizeof(uchar4), width, height);
        copyParams.extent = make_cudaExtent(width, height, depth);

        memset(&textureDesc, 0, sizeof(textureDesc));
        textureDesc.filterMode = cudaFilterModeLinear;
        textureDesc.readMode = cudaReadModeNormalizedFloat;
    }

    return create3Dtexture(cuda_image_array, copyParams, textureDesc);
}

cudaTextureObject_t Texture::createEnvironmentTexture(unsigned char* data, int width, int height, int channels) 
{
    return create2Dtexture(data, width, height, channels);
}

cudaTextureObject_t Texture::createHDRMapTexture(float4* data, int width, int height, int channels)
{
    cudaMalloc(&d_hdrMapBuffer, width * height * sizeof(float4));
    cudaMemcpy(d_hdrMapBuffer, data, width * height * sizeof(float4), cudaMemcpyHostToDevice);
    CUDA_SYNC_CHECK();

	return create2Dtexture(d_hdrMapBuffer, width, height, channels);
}

cudaTextureObject_t Texture::create1Dtexture(float4* data, int length, colorProperty type) {
    cudaTextureObject_t texture;
    cudaArray* cuda_image_array;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&cuda_image_array, &channel_desc, length, 1);
    CUDA_SYNC_CHECK();

    if (type == Specular)
    {
  //      if (d_specularBuffer != nullptr) {
		//	cudaFree(d_specularBuffer);
  //          CUDA_SYNC_CHECK();
		//}
  //      cudaMalloc(&d_specularBuffer, length * sizeof(float4));
  //      cudaMemcpy(d_specularBuffer, data, length * sizeof(float4), cudaMemcpyHostToDevice);
  //      CUDA_SYNC_CHECK();

        cudaMemcpy2DToArray(cuda_image_array, 0, 0, data, length * sizeof(float4), length * sizeof(float4), 1, cudaMemcpyHostToDevice);
    }

    if (type == Diffuse)
    {
        //if (d_diffuseBuffer != nullptr) {
        //    cudaFree(d_diffuseBuffer);
        //    CUDA_SYNC_CHECK();
        //}
        //cudaMalloc(&d_diffuseBuffer, length * sizeof(float4));
        //cudaMemcpy(d_diffuseBuffer, data, length * sizeof(float4), cudaMemcpyHostToDevice);
        //CUDA_SYNC_CHECK();

        cudaMemcpy2DToArray(cuda_image_array, 0, 0, data, length * sizeof(float4), length * sizeof(float4), 1, cudaMemcpyHostToDevice);
    }

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuda_image_array;

    cudaTextureDesc textureDesc{};
    memset(&textureDesc, 0, sizeof(textureDesc));
    textureDesc.normalizedCoords = true;
    textureDesc.filterMode = cudaFilterModeLinear;
    textureDesc.readMode = cudaReadModeElementType;
    textureDesc.addressMode[0] = cudaAddressModeClamp;

    cudaCreateTextureObject(&texture, &resDesc, &textureDesc, nullptr);
    CUDA_SYNC_CHECK();

    return texture;
}

cudaTextureObject_t Texture::create1Dtexture(float* data, int length, colorProperty textureType) {
    cudaTextureObject_t texture;
    cudaArray* cuda_image_array;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
    cudaMallocArray(&cuda_image_array, &channel_desc, length, 1);
    CUDA_SYNC_CHECK();

    if (textureType == Roughness) {
        //if (d_roughnessBuffer != nullptr) {
        //    cudaFree(d_roughnessBuffer);
        //    CUDA_SYNC_CHECK();
        //}
        //cudaMalloc(&d_roughnessBuffer, length * sizeof(float));
        //cudaMemcpy(d_roughnessBuffer, data, length * sizeof(float), cudaMemcpyHostToDevice);
        //CUDA_SYNC_CHECK();

        cudaMemcpy2DToArray(cuda_image_array, 0, 0, data, length * sizeof(float), length * sizeof(float), 1, cudaMemcpyHostToDevice);
    }
    /*if (d_roughnessBuffer != nullptr) {
		cudaFree(d_roughnessBuffer);
        CUDA_SYNC_CHECK();
	}*/

    if (textureType == TF) {
        //if (d_tfBuffer != nullptr) {
        //    cudaFree(d_tfBuffer);
        //    CUDA_SYNC_CHECK();
        //}
        //cudaMalloc(&d_tfBuffer, length * sizeof(float));
        //cudaMemcpy(d_tfBuffer, data, length * sizeof(float), cudaMemcpyHostToDevice);
        //CUDA_SYNC_CHECK();

        cudaMemcpy2DToArray(cuda_image_array, 0, 0, data, length * sizeof(float), length * sizeof(float), 1, cudaMemcpyHostToDevice);
    }

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuda_image_array;

    cudaTextureDesc textureDesc{};
    memset(&textureDesc, 0, sizeof(textureDesc));
    textureDesc.normalizedCoords = true;
    textureDesc.filterMode = cudaFilterModeLinear;
    textureDesc.readMode = cudaReadModeElementType;
    textureDesc.addressMode[0] = cudaAddressModeClamp;

    cudaCreateTextureObject(&texture, &resDesc, &textureDesc, nullptr);
    CUDA_SYNC_CHECK();

    return texture;
}

cudaTextureObject_t Texture::createBlockTexture(unsigned char* data, cudaArray *cuda_image_array, int3 size, int channels, bool normalizedFloat)
{
    cudaMemcpy3DParms copyParams = { 0 };
    cudaTextureDesc textureDesc{};

    copyParams.srcPtr = channels == 1 ? make_cudaPitchedPtr(data, size.x * sizeof(unsigned char), size.x, size.y) : make_cudaPitchedPtr(reinterpret_cast<uchar4*>(data), size.x * sizeof(uchar4), size.x, size.y);
    copyParams.extent = make_cudaExtent(size.x, size.y, size.z);

    memset(&textureDesc, 0, sizeof(textureDesc));
    textureDesc.filterMode = normalizedFloat ? cudaFilterModeLinear : cudaFilterModePoint;
    textureDesc.readMode = normalizedFloat ? cudaReadModeNormalizedFloat : cudaReadModeElementType;

    return create3Dtexture(cuda_image_array, copyParams, textureDesc);
}