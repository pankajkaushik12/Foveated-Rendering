#pragma once

#include "cuda_runtime.h"

////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef unsigned short ushort;

#define HD	__forceinline__ __host__ __device__

#include <math.h>

// float3 functions
HD float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HD float3& operator-=(float3& a, const float3& b) {
	a.x -= b.x;	a.y -= b.y;	a.z -= b.z;
	return a;
}

HD float3 operator-(const float3& a) {
	return make_float3(-a.x, -a.y, -a.z);
}

HD float3 operator-(float a, const float3& b) {
	return make_float3(a - b.x, a - b.y, a - b.z);
}

// float3 division
HD float3 operator/(const float3& a, float b) {
	return make_float3(a.x / b, a.y / b, a.z / b);
}

HD float3 operator/(const float3& a, const float3& b) {
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

HD float3& operator/=(float3& a, float b) {
	a.x /= b;	a.y /= b;	a.z /= b;
	return a;
}	


HD int3 operator/(const int3& a, int b) {
	return make_int3(a.x / b, a.y / b, a.z / b);
}

HD int3 operator+(const int3& a, int b) {
	return make_int3(a.x + b, a.y + b, a.z + b);
}

HD int3 operator*(const int3& a, int b) {
	return make_int3(a.x * b, a.y * b, a.z * b);
}

// float3 multiplication
HD float3 operator*(const float3& a, float b) {
	return make_float3(a.x * b, a.y * b, a.z * b);
}

HD float3 operator*(float a, const float3& b) {
	return make_float3(a * b.x, a * b.y, a * b.z);
}

HD float3 operator-(const float3& a, float b) {
	return make_float3(a.x - b, a.y - b, a.z - b);
}

HD float3 operator*(const float3& a, const float3& b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

HD float4 operator*(float a, const float4& b) {
	return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

HD float4 operator-(const float4& a, float b) {
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

HD float3& operator*=(float3& a, float b) {
	a.x *= b;	a.y *= b;	a.z *= b;
	return a;
}

// float2 multiplication
HD float2 operator*(const float2& a, const float& b) {
	return make_float2(a.x * b, a.y * b);
}

HD float2 operator*(const float2& a, float b) {
	return make_float2(a.x * b, a.y * b);
}

HD float2 operator*(float a, const float2& b) {
	return make_float2(a * b.x, a * b.y);
}

HD float2 operator*(const float2& a, const float2& b) {
	return make_float2(a.x * b.x, a.y * b.y);
}

HD float2 operator-(const float2& a, const float2& b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

HD float2 operator+(const float2 & a, const float2& b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

HD float2 operator-(const float2& a, float b) {
	return make_float2(a.x - b, a.y - b);
}

HD float2 operator/(const float2& a, float b) {
	return make_float2(a.x / b, a.y / b);
}

// float3 addition
HD float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HD float3& operator+=(float3& a, const float3& b) {
	a.x += b.x;	a.y += b.y;	a.z += b.z;
	return a;
}

HD float3& operator*=(float3& a, const float3& b) {
	a.x *= b.x;	a.y *= b.y;	a.z *= b.z;
	return a;
}

HD float3 operator+(const float3& a, float b) {
	return make_float3(a.x + b, a.y + b, a.z + b);
}

HD float3 operator+(float b, const float3& a) {
	return make_float3(a.x + b, a.y + b, a.z + b);
}

HD float4 operator+(const float4& a, float b) {
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

HD float4 operator*(const float4& a, float b) {
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

HD float4 operator/(const float4& a, float b) {
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

HD float4 operator*(const float4 matrix[4], const float4 vec)
{
	// matrix-vector multiplication (4x4 matrix times 4x1 vector)
	float4 result;
	result.x = matrix[0].x * vec.x + matrix[0].y * vec.y + matrix[0].z * vec.z + matrix[0].w * vec.w;
	result.y = matrix[1].x * vec.x + matrix[1].y * vec.y + matrix[1].z * vec.z + matrix[1].w * vec.w;
	result.z = matrix[2].x * vec.x + matrix[2].y * vec.y + matrix[2].z * vec.z + matrix[2].w * vec.w;
	result.w = matrix[3].x * vec.x + matrix[3].y * vec.y + matrix[3].z * vec.z + matrix[3].w * vec.w;
	return result;
}

HD bool operator==(const float3& a, const float3& b) {
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

HD bool operator!=(const float3& a, const float3& b) {
	return a.x != b.x || a.y != b.y || a.z != b.z;
}

HD bool operator!=(const int2& a, const int2& b) {
	return a.x != b.x || a.y != b.y;
}

HD float length(const float3& a) {
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

HD float length(const float2& a, const float2& b) {
	return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

HD float3 normalize(const float3& a) {
	float length = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
	return make_float3(a.x / length, a.y / length, a.z / length);
}

HD float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

HD float3 cross(const float3& a, const float3& b) {
	return make_float3(a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x);
}

HD float clamp(const float& a, float min, float max) {
	return fmaxf(min, fminf(a, max));
}

HD float3 clamp(const float3& a, float min, float max) {
	return make_float3(clamp(a.x, min, max), clamp(a.y, min, max), clamp(a.z, min, max));
}

HD float3 lerp(const float3& a, const float3& b, float t) {
	return make_float3(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z));
}

HD float lerp(const float& a, const float& b, float t) {
	return a + t * (b - a);
}
