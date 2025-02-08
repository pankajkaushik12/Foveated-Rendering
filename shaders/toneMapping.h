#pragma once

#include <optix_device.h>
#include "Utility.h"
#include "LaunchParams.h"

#define M_PIf 3.14159265358979323846f
#define InvPI 0.31830988618379067154f

extern "C" __constant__ LaunchParams optixLaunchParams;

__forceinline__ __device__ float luminance(float3& color) {
	return dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
}

__forceinline__ __device__ float3 gammaCorrect(float3& color, float gamma) {
	color = make_float3(powf(color.x, 1.0f / gamma), powf(color.y, 1.0f / gamma), powf(color.z, 1.0f / gamma));
	return color;
}

enum ToneMappingOperators {
	EXPONENTIAL,
	CLAMPPED,
	REINHARD,
	FILMIC,
	REINHARD_LUMINANCE,
	ACES_APPROX
};

__forceinline__ __device__ float3 toneMap(float3& color, float& gamma, float& exposure, int& toneMapType) {
	color = color * exposure;
	switch (toneMapType) {
	case ToneMappingOperators::EXPONENTIAL:
	{
		color.x = 1.0f - exp(-color.x);
		color.y = 1.0f - exp(-color.y);
		color.z = 1.0f - exp(-color.z);
		break;
	}
	case ToneMappingOperators::CLAMPPED:
	{
		color = clamp(color, 0.0f, 1.0f);
		break;
	}
	case ToneMappingOperators::REINHARD:
	{
		color = color / (color + 1.0f);
		break;
	}
	case ToneMappingOperators::FILMIC:
	{
		color = (color * (2.51f * color + 0.03f)) / (color * (2.43f * color + 0.59f) + 0.14f);
		break;
	}
	case ToneMappingOperators::REINHARD_LUMINANCE:
	{
		color = color * (1.0 / (luminance(color) + 1.0f));
		break;
	}
	case ToneMappingOperators::ACES_APPROX:
	{
		color = color * 0.6f;
		float a = 2.51f;
		float b = 0.03f;
		float c = 2.43f;
		float d = 0.59f;
		float e = 0.14f;
		color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
		break;
	}
	}

	return gammaCorrect(color, gamma);
}