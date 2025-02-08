#include "boundingBox.h"

using namespace gdt;

//BoundingBox boundingBox;

float3 tofloat3(vec3f v)
{
    return make_float3(v.x, v.y, v.z);
}

int3 toint3(vec3i v)
{
	return make_int3(v.x, v.y, v.z);
}

void BoundingBox::setBoxDim(vec3f BoxDim_) {
    BoxDim = BoxDim_;
    makeBoundingBox(BoxDim);
}

void BoundingBox::makeBoundingBox(vec3f BoxDim){
    affine3f xfm;
    const vec3f &center = vec3f(0.f,0.f,0.f);
    const vec3f &size = vec3f(BoxDim.x, BoxDim.y, BoxDim.z);
    xfm.p = center - 0.5f*size;
    xfm.l.vx = vec3f(size.x,0.f,0.f);
    xfm.l.vy = vec3f(0.f,size.y,0.f);
    xfm.l.vz = vec3f(0.f,0.f,size.z);

    int firstVertexID = (int)vertex.size();
    if (vertex.size() != 0) { vertex.clear(); }
    if (index.size() != 0) { index.clear(); }

    vertex.push_back(tofloat3(xfmPoint(xfm,vec3f(0.f,0.f,0.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm,vec3f(1.f,0.f,0.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm,vec3f(0.f,1.f,0.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm,vec3f(1.f,1.f,0.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm,vec3f(0.f,0.f,1.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm,vec3f(1.f,0.f,1.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm,vec3f(0.f,1.f,1.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm,vec3f(1.f,1.f,1.f))));

    int indices[] = {
    0,2,1, 1,2,3, // Bottom face
    4,5,6, 5,7,6, // Top face
    0,1,4, 1,5,4, // Front face
    2,6,3, 3,6,7, // Back face
    0,4,2, 2,4,6, // Left face
    1,3,5, 3,7,5  // Right face
    };

    for (int i = 0; i < 12; i++) {
        index.push_back(make_int3(firstVertexID + indices[3 * i + 0], firstVertexID + indices[3 * i + 1], firstVertexID + indices[3 * i + 2]));
    }
}

std::vector<float3> BoundingBox::getVertices(){
    return vertex;
}

std::vector<int3> BoundingBox::getIndices(){
    return index;
}

void BoundingBox::alphaPartBBox(int3 offset, int3 size, std::vector<float3>& vertex, std::vector<int3>& index) {
    affine3f xfm;
    vec3f center, partSize;
    center.x = float(offset.x); center.y = float(offset.y), center.z = float(offset.z);
    const vec3f &size_ = BoxDim;
    xfm.p = center - 0.5f*size_;
    partSize.x = float(size.x); partSize.y = float(size.y); partSize.z = float(size.z);
    xfm.l.vx = vec3f(partSize.x, 0.f, 0.f);
    xfm.l.vy = vec3f(0.f, partSize.y, 0.f);
    xfm.l.vz = vec3f(0.f, 0.f, partSize.z);
    int firstVertexID = (int)vertex.size();
    if (vertex.size() != 0) { vertex.clear(); }
    if (index.size() != 0) { index.clear(); }
    vertex.push_back(tofloat3(xfmPoint(xfm, vec3f(0.f, 0.f, 0.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm, vec3f(1.f, 0.f, 0.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm, vec3f(0.f, 1.f, 0.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm, vec3f(1.f, 1.f, 0.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm, vec3f(0.f, 0.f, 1.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm, vec3f(1.f, 0.f, 1.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm, vec3f(0.f, 1.f, 1.f))));
    vertex.push_back(tofloat3(xfmPoint(xfm, vec3f(1.f, 1.f, 1.f))));
    int indices[] = {
    0,2,1, 1,2,3, // Bottom face
    4,5,6, 5,7,6, // Top face
    0,1,4, 1,5,4, // Front face
    2,6,3, 3,6,7, // Back face
    0,4,2, 2,4,6, // Left face
    1,3,5, 3,7,5  // Right face
	};
    for (int i = 0; i < 12; i++) {
        index.push_back(make_int3(firstVertexID + indices[3 * i + 0], firstVertexID + indices[3 * i + 1], firstVertexID + indices[3 * i + 2]));
    }
}

void BoundingBox::printVertices(){
    for(int i=0;i<vertex.size(); i++){
        printf("Vertex %d: %f %f %f\n", i, vertex[i].x, vertex[i].y, vertex[i].z);
    }
    std::cout<<"\n";
}

void BoundingBox::printIndicies(){
    for(int i=0;i<index.size(); i++){
        printf("Index %d: %d %d %d\n", i, index[i].x, index[i].y, index[i].z);
    }
    std::cout<<"\n";
}

vec3f BoundingBox::getBoxDim() {
    return BoxDim;
}

vec3f BoundingBox::getExtentMin() {
	return vec3f(BoxDim) / (float) 2.0;
}

vec3f BoundingBox::getExtentMax() {
	return vec3f(BoxDim) / (float) -2.0;
}