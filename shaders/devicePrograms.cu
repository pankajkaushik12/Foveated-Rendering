#include "Utility.h"
#include "cutil_math.h"
#include "random.h"
#include "toneMapping.h"

#include "LaunchParams.h"

extern "C" __constant__ LaunchParams optixLaunchParams;

extern "C" __global__ void __closesthit__radiance()
{
    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    rayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());

    const int   primID = optixGetPrimitiveIndex();
    const int3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float3& A = sbtData.vertex[index.x];
    const float3& B = sbtData.vertex[index.y];
    const float3& C = sbtData.vertex[index.z];
    float3 Ng = cross(B - A, C - A);
    float3 Ns = (sbtData.normal) ? ((1.f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z]) : Ng;

    float3 diffuseColor = sbtData.color;
    if (sbtData.hasTexture && sbtData.texcoord) {
        const float2 tc = (1.f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];

        vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= make_float3(fromTexture.x, fromTexture.y, fromTexture.z);
    }

    thePrd->color = diffuseColor;

    const float3 surfPos = (1.f - u - v) * sbtData.vertex[index.x] + u * sbtData.vertex[index.y] + v * sbtData.vertex[index.z];

    /*if (thePrd->rayDepth == 0 && optixLaunchParams.numLights > 0)
    {
        for (int i = 0; i < optixLaunchParams.numLights; i++)
        {
            if (optixLaunchParams.lights[i].type != LightType::Point) continue;
            float3 lightDir = make_float3(sin(optixLaunchParams.lights[i].phi) * cos(optixLaunchParams.lights[i].theta), 
                                            sin(optixLaunchParams.lights[i].phi) * sin(optixLaunchParams.lights[i].theta),
                                            cos(optixLaunchParams.lights[i].phi));
            float3 lightPos = optixLaunchParams.lights[i].target + optixLaunchParams.lights[i].distance * lightDir;

            rayData ray;
            uint2 payload = splitPointer(&ray);
            ray.origin = surfPos + Ns * 1e-3f;
            ray.direction = normalize(lightPos - ray.origin);
            ray.tmax = length(lightPos - ray.origin);
            
            optixTrace(optixLaunchParams.traversable,
                ray.origin, //camera.position,
                ray.direction,
                ray.tmin,    // tmin
                ray.tmax,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                SHADOW_RAY_TYPE,             // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                SHADOW_RAY_TYPE,             // missSBTIndex 
                payload.x, payload.y);

            // If the Shadow Ray did not hit anything
            if (ray.color != make_float3(0, 0, 0)) {
                float3 lightColor = optixLaunchParams.lights[i].color;
                float3 lightIntensity = lightColor * optixLaunchParams.lights[i].intensity;
                float3 L = normalize(lightPos - surfPos);
                float3 diffuse = diffuseColor * fmax(dot(Ns, L), 0.0f);
                thePrd->color += thePrd->beta * diffuse * lightIntensity;
            }
        }
    }*/

    thePrd->color = optixLaunchParams.enablePhongShading ? phongShading(thePrd->origin + thePrd->direction * optixGetRayTmax(), thePrd->direction, thePrd->color, Ns) : thePrd->color;
}

extern "C" __global__ void __anyhit__radiance()
{

}

extern "C" __global__ void __miss__radiance()
{
    // The RadianceRay did not hit anything in the scene
    return;
}

extern "C" __global__ void __closesthit__shadow()
{
    /* not going to be used ... */
}

extern "C" __global__ void __anyhit__shadow()
{
    /* not going to be used ... */
}

extern "C" __global__ void __miss__shadow()
{
    rayData* thePrd = mergePointer(optixGetPayload_0(), optixGetPayload_1());
    thePrd->color = make_float3(1, 1, 1);
}

extern "C" __global__ void __raygen__renderFrame()
{
    int ix, iy;
    int2 pixelID = getPixelIndex(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    ix = pixelID.x;
    iy = pixelID.y;
    vec2i pixel = vec2i(ix, iy);
    if (optixLaunchParams.foveatedRendering) {
        pixel = rectangularFoveatedInverseMapping(pixel);
    }

    auto &camera = optixLaunchParams.camPos;

    rayData prd;
    uint2 payload = splitPointer(&prd);

    float3 rayDir, color = { 0, 0, 0 };
    float transmittance = 0.0f;
    float2 screen;
    float2 renderDimFull = make_float2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y) / optixLaunchParams.fractionToRender;
    for (int i = 0; i < optixLaunchParams.samplePerPixel; i++) {

        float2 u = make_float2(0.f, 0.f);// GetUniform2();
        screen = make_float2((pixel.x + u.x) / renderDimFull.x, (pixel.y + u.y) / renderDimFull.y);
        
        // Create the ray direction with inverse of view matrix and inverse projection matrix
        screen = 2.f * screen - 1.f;
        float4 rayDir4 = make_float4(screen.x, screen.y, -1.0f, 1.0f);
        rayDir4 = optixLaunchParams.camPos.invProj * rayDir4;
        rayDir4.w = 0.0f;
        rayDir4 = optixLaunchParams.camPos.invView * rayDir4;
        rayDir = normalize(make_float3(rayDir4.x, rayDir4.y, rayDir4.z));
        
        float3 LI = make_float3(0, 0, 0);

        if (optixLaunchParams.camPos.apertureRadius != 0.0f)
        {
            float2 LensUV = optixLaunchParams.camPos.apertureRadius * ConcentricSampleDisk(GetUniform2());

            LI = camera.horizontal * LensUV.x + camera.vertical * LensUV.y;
            rayDir = normalize((rayDir * optixLaunchParams.camPos.focalDistance) - LI);
        }

        prd.origin = camera.position + LI;
        prd.direction = rayDir;
        prd.color = make_float3(0.f, 0.f, 0.f);
        prd.transmittance = 1.0f;
        prd.tmin = 0.0f;
        prd.tmax = 1e20f;

        optixTrace(optixLaunchParams.traversable,
            prd.origin, //camera.position,
            prd.direction,
            prd.tmin,    // tmin
            prd.tmax,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RADIANCE_RAY_TYPE,             // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            RADIANCE_RAY_TYPE,             // missSBTIndex 
            payload.x, payload.y);

        color += prd.color;
        transmittance += prd.transmittance;
    }
    color /= optixLaunchParams.samplePerPixel;
    transmittance /= optixLaunchParams.samplePerPixel;

    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

    if (optixLaunchParams.accumulateFrames) {
        float3 prevColor = optixLaunchParams.frameIndex == 1 ? make_float3(0, 0, 0) :
                                    make_float3(optixLaunchParams.frame.noisedColorBuffer[fbIndex].x, optixLaunchParams.frame.noisedColorBuffer[fbIndex].y, optixLaunchParams.frame.noisedColorBuffer[fbIndex].z);
        color = prevColor + (color - prevColor) / (optixLaunchParams.frameIndex);
    }

    uint8_t r, g, b, a;
    float4 denoisedColor = optixLaunchParams.frameIndex == 1 ? make_float4(color.x, color.y, color.z, 1.0f) : optixLaunchParams.frame.denoisedColorBuffer[fbIndex];
    float3 col = optixLaunchParams.denoiseImage ? make_float3(denoisedColor.x, denoisedColor.y, denoisedColor.z) : color;
    float3 afterToneMap = toneMap(col, optixLaunchParams.gammaEnv, optixLaunchParams.exposureEnv, optixLaunchParams.envToneMap);
    r = static_cast<uint8_t>(afterToneMap.x * 255.0f);
    g = static_cast<uint8_t>(afterToneMap.y * 255.0f);
    b = static_cast<uint8_t>(afterToneMap.z * 255.0f);
    a = 0xff;

    if (optixLaunchParams.renderEyePos)
    {
        float2 pixelPos = make_float2(pixel.x / float(optixLaunchParams.frame.fullFrameSize.x), pixel.y / float(optixLaunchParams.frame.fullFrameSize.y));
        if (length(optixLaunchParams.eyePos, pixelPos) < 0.005f)
        {
            r = 255;
            g = 0;
            b = 0;
            a = 0xff;
        }
    }

    optixLaunchParams.frame.colorBuffer[fbIndex * 4] = r;
    optixLaunchParams.frame.colorBuffer[fbIndex * 4 + 1] = g;
    optixLaunchParams.frame.colorBuffer[fbIndex * 4 + 2] = b;
    optixLaunchParams.frame.colorBuffer[fbIndex * 4 + 3] = a;
    optixLaunchParams.frame.noisedColorBuffer[fbIndex] = make_float4(color.x, color.y, color.z, 1.0f);
}

