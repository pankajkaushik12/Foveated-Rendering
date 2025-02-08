#pragma once

#include <cuda_runtime.h>
#include "optix7.h"
#include "gdt/math/vec.h"
#include <string.h>
#include <omp.h>

#include "transferFunction.h"
#include "Model.h"
using namespace gdt;

class Texture{
  private:
	  cudaResourceDesc resourceDesc{};
	  cudaTextureObject_t textureObject{};
	  cudaChannelFormatDesc channelDesc{};

	  float* d_opacityBuffer = nullptr;
	  uchar4* d_rgbaBuffer = nullptr;
	  //unsigned char* d_intensityBuffer = nullptr;
	  unsigned short* d_segmentationBuffer = nullptr;

	  float4* d_gradientBuffer = nullptr;
	  float4* d_hdrMapBuffer = nullptr;

	  float4* d_diffuseBuffer = nullptr;
	  float4* d_specularBuffer = nullptr;
	  float* d_roughnessBuffer = nullptr;
	  float* d_tfBuffer = nullptr;

	  //cudaArray* cuda_image_array = nullptr;

	  float3 texSize{};
  public:
	  Texture() {};
	  ~Texture()
	  {
		  if (d_opacityBuffer) cudaFree(d_opacityBuffer);
		  if (d_rgbaBuffer) cudaFree(d_rgbaBuffer);
		  //if (d_intensityBuffer) cudaFree(d_intensityBuffer);
		  if (d_segmentationBuffer) cudaFree(d_segmentationBuffer);
		  if (d_gradientBuffer) cudaFree(d_gradientBuffer);
		  if (d_hdrMapBuffer) cudaFree(d_hdrMapBuffer);
		  if (d_diffuseBuffer) cudaFree(d_diffuseBuffer);
		  if (d_specularBuffer) cudaFree(d_specularBuffer);
		  if (d_roughnessBuffer) cudaFree(d_roughnessBuffer);
		  if (d_tfBuffer) cudaFree(d_tfBuffer);
		  cudaDestroyTextureObject(textureObject);
	  }

	  cudaTextureObject_t createIntensityTexture(float* data, float3 size, unsigned int channels);
	  cudaTextureObject_t createIntensityTexture(unsigned char* data, float3 size, unsigned int channels);
	  cudaTextureObject_t createSegmentationTexture(unsigned short* data, float3 size, unsigned int channels);
	  cudaTextureObject_t createOpacityTexture(float* opacityArray, float3 size, unsigned int channels);
	  cudaTextureObject_t createColorTexture(unsigned char* data, float3 size, unsigned int channels);
	  cudaTextureObject_t createGradientTexture(float* data, float3 size, unsigned int channels);
	  cudaTextureObject_t createEnvironmentTexture(unsigned char* data, int width, int height, int channel);
	  cudaTextureObject_t createHDRMapTexture(float4* data, int width, int height, int channel);
	  cudaTextureObject_t create2Dtexture(unsigned char *data, int width, int height, int channel);
	  cudaTextureObject_t create2Dtexture(float4* data, int width, int height, int channel);
	  cudaTextureObject_t create3Dtexture(cudaArray_t cuda_image_array, cudaMemcpy3DParms &params, cudaTextureDesc &textureDesc);
	  cudaTextureObject_t create1Dtexture(float* data, int length, colorProperty type);
	  cudaTextureObject_t create1Dtexture(float4* data, int length, colorProperty type);
	  cudaTextureObject_t getTexturePtr();

	  void createTexture(Model* model, std::vector<cudaArray_t> &textureArrays);

	  cudaTextureObject_t createBlockTexture(unsigned char* data, cudaArray *cuda_image_array, int3 size, int channels, bool readNormalizedFloat);
};
