// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "gdt/math/AffineSpace.h"
#include "cutil_math.h"
#include "tinyply/source/tinyply.h"
#include <filesystem>
#include <vector>
#include <omp.h>

#include "imageProcessing.h"
#include "boundingBox.h"
namespace fs = std::filesystem;

using namespace gdt;
using namespace tinyply;

struct TriangleMesh {
    std::vector<float3> vertex;
    std::vector<float3> normal;
    std::vector<int3> index;
	std::vector<float2> texcoord;

    float3 MinBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MaxBound = -make_float3(FLT_MAX, FLT_MAX, FLT_MAX);

    float3              diffuse;
    int                diffuseTextureID{ -1 };
};

struct TextureMap
{
    ~TextureMap() {
		if (pixel) delete[] pixel;
    }
	uint32_t* pixel = nullptr;
	cudaTextureObject_t textureObject;
	int width, height;
	int nrChannels;
};

class Model {
private:
    std::string modelFolder;
    std::vector<TriangleMesh *> meshes;
    std::vector<TextureMap* > textures;
    float3 minBounds = make_float3(FLT_MAX, FLT_MAX, FLT_MAX), maxBounds = -make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float 									 worldScale;

    ImageProcessing imageProcessing;

public:
    Model() {};
    Model(std::string modelPath);
    ~Model()
    {
		for (auto mesh : meshes) delete mesh;
	}

    void setWorldScale(float scale);
    void setBounds(float3 min, float3 max);

    int loadTexture(std::vector<TextureMap*>& textures, std::map<std::string, int>& knownTextures, const std::string& inFileName, const std::string& modelPath);

    std::vector<TriangleMesh *> getMeshes();
    std::vector<TextureMap* > getTextures();

    float getWorldScale();

    void loadObj(std::string filePath);

	float getModelSpan();
};
