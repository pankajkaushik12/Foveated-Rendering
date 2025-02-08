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

#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

//std
#include <set>

namespace std {
  inline bool operator<(const tinyobj::index_t &a,
                        const tinyobj::index_t &b)
  {
    if (a.vertex_index < b.vertex_index) return true;
    if (a.vertex_index > b.vertex_index) return false;
    
    if (a.normal_index < b.normal_index) return true;
    if (a.normal_index > b.normal_index) return false;
    
    if (a.texcoord_index < b.texcoord_index) return true;
    if (a.texcoord_index > b.texcoord_index) return false;
    
    return false;
  }
}

int Model::loadTexture(std::vector<TextureMap*>& tetxures,
    std::map<std::string, int>& knownTextures,
    const std::string& inFileName,
    const std::string& modelPath)
{
    if (inFileName == "")
        return -1;

    if (knownTextures.find(inFileName) != knownTextures.end())
        return knownTextures[inFileName];

    std::string fileName = inFileName;
    // first, fix backspaces:
    for (auto& c : fileName)
        if (c == '\\') c = '/';
    fileName = modelPath + "/" + fileName;

    int width, height, channels;
    unsigned char* image;
    imageProcessing.loadImage(fileName, width, height, channels, image);

    int textureID = -1;
    if (image) {
        textureID = (int)textures.size();
        TextureMap* texture = new TextureMap;
        texture->width = width;
        texture->height = height;
        texture->pixel = (uint32_t*)image;

        /* iw - actually, it seems that stbi loads the pictures
           mirrored along the y axis - mirror them here */
        for (int y = 0; y < height / 2; y++) {
            uint32_t* line_y = texture->pixel + y * width;
            uint32_t* mirrored_y = texture->pixel + (height - 1 - y) * width;
            int mirror_y = height - 1 - y;
            for (int x = 0; x < width; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }

        textures.push_back(texture);
    }
    else {
        std::cout << GDT_TERMINAL_RED
            << "Could not load texture from " << fileName << "!"
            << GDT_TERMINAL_DEFAULT << std::endl;
    }

    knownTextures[inFileName] = textureID;
    return textureID;
}

Model::Model(std::string modelPath)
{
    modelFolder = modelPath.substr(0, modelPath.rfind('/') + 1);

    std::string extension = modelPath.substr(modelPath.find_last_of(".") + 1);

    if (extension == "obj") { loadObj(modelPath); } else { printf("Only .obj files are supported\n"); }
}

int addVertex(TriangleMesh* mesh, tinyobj::attrib_t& attributes, const tinyobj::index_t& idx, std::map<tinyobj::index_t, int>& knownVertices)
{
    if (knownVertices.find(idx) != knownVertices.end())
        return knownVertices[idx];

    const float3* vertex_array = (const float3*)attributes.vertices.data();
    const float3* normal_array = (const float3*)attributes.normals.data();
    const float2* texcoord_array = (const float2*)attributes.texcoords.data();

    int newID = (int)mesh->vertex.size();
    knownVertices[idx] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
        while (mesh->normal.size() < mesh->vertex.size())
            mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
        while (mesh->texcoord.size() < mesh->vertex.size())
            mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
        mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
        mesh->normal.resize(mesh->vertex.size());

    return newID;
}

std::vector<TriangleMesh*> Model::getMeshes()
{
	return meshes;
}

std::vector<TextureMap*> Model::getTextures()
{
    return textures;
}

void Model::setBounds(float3 min, float3 max)
{
    minBounds = min;
    maxBounds = max;
}

void Model::setWorldScale(float scale)
{
	worldScale = scale;
}

float Model::getWorldScale() {
	return worldScale;
}

void Model::loadObj(std::string objFile)
{
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err = "";

    bool readOK
        = tinyobj::LoadObj(&attributes,
            &shapes,
            &materials,
            &err,
            &err,
            objFile.c_str(),
            modelFolder.c_str(),
            /* triangulate */true);
    if (!readOK) {
        throw std::runtime_error("Could not read OBJ model from " + objFile + " : " + err);
    }

    if (materials.empty()) {
        throw std::runtime_error("could not parse materials ...");
    }

    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;

    std::map<std::string, int>      knownTextures;
    for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
        tinyobj::shape_t& shape = shapes[shapeID];

        std::set<int> materialIDs;
        for (auto faceMatID : shape.mesh.material_ids)
            materialIDs.insert(faceMatID);

        for (int materialID : materialIDs) {
            std::map<tinyobj::index_t, int> knownVertices;
            TriangleMesh* mesh = new TriangleMesh;

            for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                if (shape.mesh.material_ids[faceID] != materialID) continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                vec3i idx(addVertex(mesh, attributes, idx0, knownVertices), addVertex(mesh, attributes, idx1, knownVertices), addVertex(mesh, attributes, idx2, knownVertices));
                mesh->index.push_back(make_int3(idx.x, idx.y, idx.z));
                mesh->diffuse = (const float3&)materials[materialID].diffuse;
                mesh->diffuseTextureID = loadTexture(textures, knownTextures, materials[materialID].diffuse_texname, modelFolder);
            }

            if (mesh->vertex.empty())
                delete mesh;
            else
                meshes.push_back(mesh);
        }
    }


    for (auto mesh : meshes)
    {
        for (auto vtx : mesh->vertex)
        {
            minBounds = make_float3(fminf(minBounds.x, vtx.x), fminf(minBounds.y, vtx.y), fminf(minBounds.z, vtx.z));
			maxBounds = make_float3(fmaxf(maxBounds.x, vtx.x), fmaxf(maxBounds.y, vtx.y), fmaxf(maxBounds.z, vtx.z));
        }
    }
    std::cout << "created a total of " << meshes.size() << " meshes" << std::endl;    
}

float Model::getModelSpan()
{
	return length(maxBounds - minBounds);
}
