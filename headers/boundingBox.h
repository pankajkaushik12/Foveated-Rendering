#pragma once

#include "gdt/math/AffineSpace.h"
#include "cutil_math.h"
#include "vector"

using namespace gdt;

class BoundingBox{
    private:
        vec3f BoxDim;
        std::vector<float3> vertex;
        std::vector<int3> index;
    public:
        BoundingBox() {};
        ~BoundingBox()
        {
            vertex.clear();
            index.clear();
        }
        void makeBoundingBox(vec3f BoxDim);
        void setBoxDim(vec3f BoxDim);
        vec3f getBoxDim();
        vec3f getExtentMin();
        vec3f getExtentMax();
        void alphaPartBBox(int3 offset, int3 size, std::vector<float3> &vertex, std::vector<int3> &index);
        std::vector<float3> getVertices();
        std::vector<int3> getIndices();
        void printVertices();
        void printIndicies();
};
