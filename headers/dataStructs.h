#ifndef DATASTRUCTS_H
#define DATASTRUCTS_H

struct segmentDict
{
    unsigned char key = 0;
    bool value = 0;
};

struct TriangleMeshSBTData {
    float3  color;
	float3* vertex;
    float3* normal;
    int3* index;

    float2* texcoord;
    bool hasTexture;
    cudaTextureObject_t texture;
};

#endif // !DATASTRUCTS_H