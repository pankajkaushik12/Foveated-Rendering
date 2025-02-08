#pragma once
#include "curand.h"
#include "curand_kernel.h"

typedef struct {
    float3 origin;
    float3 direction;
    float3 color = make_float3(0, 0 ,0);
    float3 beta = make_float3(1, 1, 1);
    float transmittance = 1.f;
    float tmin = 1e-3f;
    float tmax = 1e20f;
    int rayDepth = 0;
} rayData;

// typedef union
typedef union
{
  rayData* ptr;
  uint2   dat;
}Payload;

__forceinline__ __device__ uint2 splitPointer(rayData* ptr)
{
  Payload payload;

  payload.ptr = ptr;

  return payload.dat;
}

__forceinline__ __device__ rayData* mergePointer(unsigned int p0, unsigned int p1)
{
  Payload payload;

  payload.dat.x = p0;
  payload.dat.y = p1;

  return payload.ptr;
}