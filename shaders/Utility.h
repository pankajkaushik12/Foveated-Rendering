#pragma once

#include <optix_device.h>
#include "random.h"
#include "cutil_math.h"
#include "PerRayData.h"
#include "LaunchParams.h"

#define M_PIf 3.14159265358979323846f
#define InvPI 0.31830988618379067154f
#define inf 1e20f
#define eps 1e-4f

// Reference white point (D65 standard illuminant)
#define Xn 95.047f // Reference X
#define Yn 100.000 // Reference Y
#define Zn 108.883 // Reference Z

extern "C" __constant__ LaunchParams optixLaunchParams;

__forceinline__ __device__ void rgbToyuv(uint8_t r, uint8_t g, uint8_t b, uint8_t& y, uint8_t& u, uint8_t& v);

__forceinline__ __device__ double gammaCorrection(double channel) {
    if (channel <= 0.04045) {
        return channel / 12.92;
    }
    else {
        return float(pow((channel + 0.055) / 1.055, 2.4));
    }
}

__forceinline__ __device__ float GetUniform1()
{
    return curand_uniform(optixLaunchParams.d_rand_state);
}

__forceinline__ __device__ float2 GetUniform2()
{
    return make_float2(GetUniform1(), GetUniform1());
}

__forceinline__ __device__ float3 RGBToXYZ(float3 rgb) {

    rgb.x = gammaCorrection(rgb.x);
    rgb.y = gammaCorrection(rgb.y);
    rgb.z = gammaCorrection(rgb.z);

    float3 xyz;
    xyz.x = 0.412453f * rgb.x + 0.357580f * rgb.y + 0.180423f * rgb.z;
    xyz.y = 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z;
    xyz.z = 0.019334f * rgb.x + 0.119193f * rgb.y + 0.950227f * rgb.z;
    return xyz;
}

__forceinline__ __device__ float3 toRGB(float3 xyz) {
    float3 rgb;
    rgb.x = 3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z;
    rgb.y = -0.969256f * xyz.x + 1.875992f * xyz.y + 0.041556f * xyz.z;
    rgb.z = 0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z;
    return rgb;
}

__forceinline__ __device__ float4 toRGB(float4 xyza) {
    float3 rgb = toRGB(make_float3(xyza.x, xyza.y, xyza.z));
    return make_float4(rgb.x, rgb.y, rgb.z, xyza.w);
}

__forceinline__ __device__ float3 XYZToLUV(float3 xyz)
{
    double X = xyz.x;
    double Y = xyz.y;
    double Z = xyz.z;

    double uPrime = (4 * X) / (X + 15 * Y + 3 * Z);
    double vPrime = (9 * Y) / (X + 15 * Y + 3 * Z);

    double uPrimeRef = (4 * Xn) / (Xn + 15 * Yn + 3 * Zn);
    double vPrimeRef = (9 * Yn) / (Xn + 15 * Yn + 3 * Zn);

    double L, U, V;
    if (Y / Yn > 0.008856) {
        L = 116 * cbrt(Y / Yn) - 16;
    }
    else {
        L = 903.3 * (Y / Yn);
    }

    U = 13 * L * (uPrime - uPrimeRef);
    V = 13 * L * (vPrime - vPrimeRef);
    
    return make_float3(L, U, V);
}

__forceinline__ __device__ float3 RGBToLUV(float3 rgb)
{
    float3 xyz = RGBToXYZ(rgb);
	return XYZToLUV(xyz);
}

__forceinline__ __device__ void rgbToyuv(uint8_t R, uint8_t G, uint8_t B, uint8_t& y, uint8_t& u, uint8_t& v) {
    y = (unsigned char)(0.299 * R + 0.587 * G + 0.114 * B);
    u = (unsigned char)(-0.14713 * R - 0.28886 * G + 0.436 * B + 128);
    v = (unsigned char)(0.615 * R - 0.51499 * G - 0.10001 * B + 128);
}

__forceinline__ __device__ float2 ConcentricSampleDisk(const float2& U)
{
    float r, theta;
    // Map uniform random numbers to $[-1,1]^2$
    float sx = 2 * U.x - 1;
    float sy = 2 * U.y - 1;
    // Map square to $(r,\theta)$
    // Handle degeneracy at the origin

    if (sx == 0.0 && sy == 0.0)
    {
        return make_float2(0.0f, 0.0f);
    }

    if (sx >= -sy)
    {
        if (sx > sy)
        {
            // Handle first region of disk
            r = sx;
            if (sy > 0.0)
                theta = sy / r;
            else
                theta = 8.0f + sy / r;
        }
        else
        {
            // Handle second region of disk
            r = sy;
            theta = 2.0f - sx / r;
        }
    }
    else
    {
        if (sx <= sy)
        {
            // Handle third region of disk
            r = -sx;
            theta = 4.0f - sy / r;
        }
        else
        {
            // Handle fourth region of disk
            r = -sy;
            theta = 6.0f + sx / r;
        }
    }

    theta *= M_PIf / 4.f;

    return make_float2(r * cosf(theta), r * sinf(theta));
}

__forceinline__ __device__ float3 CosineWeightedHemisphere(float2& U)
{
    float2 ret = ConcentricSampleDisk(U);
    return make_float3(ret.x, ret.y, sqrtf(max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y)));
}

__forceinline__ __device__ bool SameHemisphere(float3& Ww1, float3& Ww2)
{
    return Ww1.z * Ww2.z > 0.0f;
}

__forceinline__ __device__ float2 calculateIntersection(float3 camOrigin, float3 direction, float3 ExtentMin, float3 ExtentMax)
{
    float tymin, tymax, tzmin, tzmax, tmin, tmax;
    tmin = 0.0f;
    tmax = 1e20f;
    float3 invdir = { 1 / direction.x, 1 / direction.y, 1 / direction.z };

    tmin = invdir.x < 0 ? (ExtentMax.x - camOrigin.x) * invdir.x : (ExtentMin.x - camOrigin.x) * invdir.x;
    tmax = invdir.x < 0 ? (ExtentMin.x - camOrigin.x) * invdir.x : (ExtentMax.x - camOrigin.x) * invdir.x;

    tymin = invdir.y < 0 ? (ExtentMax.y - camOrigin.y) * invdir.y : (ExtentMin.y - camOrigin.y) * invdir.y;
    tymax = invdir.y < 0 ? (ExtentMin.y - camOrigin.y) * invdir.y : (ExtentMax.y - camOrigin.y) * invdir.y;

    tmin = tymin > tmin ? tymin : tmin;
    tmax = tymax < tmax ? tymax : tmax;

    tzmin = invdir.z < 0 ? (ExtentMax.z - camOrigin.z) * invdir.z : (ExtentMin.z - camOrigin.z) * invdir.z;
    tzmax = invdir.z < 0 ? (ExtentMin.z - camOrigin.z) * invdir.z : (ExtentMax.z - camOrigin.z) * invdir.z;

    tmin = tzmin > tmin ? tzmin : tmin;
    tmax = tzmax < tmax ? tzmax : tmax;

    return make_float2(tmin, tmax);
}

__forceinline__ __device__ float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

__forceinline__ __device__ float3 uniformSampleSphere(float r1, float r2) {
    float theta = 2 * M_PIf * r1;   // azimuthal angle  (0, 2pi)
    float phi = acos(1 - 2 * r2);   // polar angle      (0, pi)
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    return make_float3(x, y, z);
}

__forceinline__ __device__ bool intersectPlane(float3 planeNormal, float3 planePos, float width, float height, float3 rayOrigin, float3 rayDir, float t)
{
    // assuming vectors are all normalized
    float denom = dot(planeNormal, rayDir);
    if (denom > (float)1e-6) {
        return false;
    }
    float3 p0l0 = planePos - rayOrigin;
    t = dot(p0l0, planeNormal) / denom;
    if (t < 0) return false;

    if (t < 0.0f || abs(rayOrigin.x + t * rayDir.x - planePos.x) > width / 2.0f || abs(rayOrigin.y + t * rayDir.y - planePos.y) > height / 2.0f) {
        return false;
    }
    return true;
}

__forceinline__ __device__ float3 reflect(const float3& incidentVec, const float3& normal)
{
    return incidentVec - 2.f * dot(incidentVec, normal) * normal;
}

__forceinline__ __device__ float3 phongShading(float3 pos, float3 dir, float3 fColor, float3 normal)
{
    if (length(normal) < 0.1f)  return fColor;
    float3 ambient = optixLaunchParams.ambientConstant * fColor;
    float3 diffuse = make_float3(0.0f, 0.0f, 0.0f);
    float3 specular = make_float3(0.0f, 0.0f, 0.0f);

    float3 lightPos, lightColor, lightVec;
    float distance, attenuation, specularFactor;
    
    if (optixLaunchParams.enableHeadLight) {
        lightPos = optixLaunchParams.camPos.position;
        lightColor = make_float3(1.0f, 1.0f, 1.0f);
        lightVec = normalize(lightPos - pos); // scatterPos to ligthPos

        distance = length(lightPos - pos);
        attenuation = 1.0f / (optixLaunchParams.attenuationConstA + optixLaunchParams.attenuationConstB * distance + optixLaunchParams.attenuationConstC * distance * distance);
        
        diffuse += fmaxf(dot(normal, lightVec), 0.0f) * fColor * lightColor * attenuation;
        
        specularFactor = powf(fmaxf(dot(normalize(-dir), normalize(reflect(-lightVec, normal))), 0.0f), optixLaunchParams.shininess);
        specular += specularFactor * (fColor) * lightColor * attenuation;
    }

    return clamp(ambient + diffuse + specular, 0.0f, 1.0f);
}

__forceinline__ __device__ vec2i rectangularFoveatedInverseMapping(vec2i pixelID) {

    float renderFrameX = optixLaunchParams.frame.size.x;          // Dimension of the small buffer (Currenly it is rendered frame size)
    float renderFrameY = optixLaunchParams.frame.size.y;

    float fx = optixLaunchParams.focalLength.x;                                             //optixLaunchParams.camera.fovx;
    float fy = optixLaunchParams.focalLength.y;                                                 //optixLaunchParams.camera.fovy;

    float2 cursorPos = optixLaunchParams.eyePos;
    float aspectRatio = optixLaunchParams.fractionToRender;

    float maxDxPos = 1.0 - cursorPos.x;
    float maxDyPos = 1.0 - cursorPos.y;
    float maxDxNeg = cursorPos.x;
    float maxDyNeg = cursorPos.y;

    float norDxPos = fx * maxDxPos / (fx + maxDxPos);
    float norDyPos = fy * maxDyPos / (fy + maxDyPos);
    float norDxNeg = fx * maxDxNeg / (fx + maxDxNeg);
    float norDyNeg = fy * maxDyNeg / (fy + maxDyNeg);

    float2 tc = make_float2(float(pixelID.x) / renderFrameX, float(pixelID.y) / renderFrameY) - cursorPos;

    float x = tc.x > 0 ? tc.x / maxDxPos : tc.x / maxDxNeg;
    float y = tc.y > 0 ? tc.y / maxDyPos : tc.y / maxDyNeg;
    if (tc.x >= 0) {
        x *= norDxPos;
        x = fx * x / (fx - x);
        x += cursorPos.x;
    }
    else {
        x *= norDxNeg;
        x = fx * x / (fx + x);
        x += cursorPos.x;
    }
    if (tc.y >= 0) {
        y *= norDyPos;
        y = fy * y / (fy - y);
        y += cursorPos.y;
    }
    else {
        y *= norDyNeg;
        y = fy * y / (fy + y);
        y += cursorPos.y;
    }

    return vec2i(int(x * renderFrameX / aspectRatio), int(y * renderFrameY / aspectRatio));
}

__forceinline__ __device__ int2 getPixelIndex(int ix, int iy)
{
    switch (optixLaunchParams.totalCudaStreams) {
    case 1:
    {
        return make_int2(ix, iy);
    }
    case 2:
    {
        switch (optixLaunchParams.cudaStreamID) {
        case 0:
        {
            return make_int2(ix, iy);
        }
        case 1:
        {
            return make_int2(ix + optixLaunchParams.frame.size.x / 2, iy);
        }
        }
    }
    case 4:
    {
        switch (optixLaunchParams.cudaStreamID) {
        case 0:
        {
            return make_int2(ix, iy);
        }
        case 1:
        {
            ix = ix + optixLaunchParams.frame.size.x / 2;
            iy = iy;
            return make_int2(ix, iy);
        }
        case 2:
        {
            ix = ix;
            iy = iy + optixLaunchParams.frame.size.y / 2;
            return make_int2(ix, iy);
        }
        case 3:
        {
            ix = ix + optixLaunchParams.frame.size.x / 2;
            iy = iy + optixLaunchParams.frame.size.y / 2;
            return make_int2(ix, iy);
        }
        }
    }
    }
    return make_int2(ix, iy);
}
