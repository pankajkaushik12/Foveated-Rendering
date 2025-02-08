#pragma once
#include <vector>
#include "cutil_math.h"
#include "dataStructs.h"
#include "gdt/math/AffineSpace.h"

#include "glm/gtc/type_ptr.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/string_cast.hpp"

using namespace gdt;

class Camera{
	float fovVertical;
	int2 frameSize;
	float3 from;
	float3 gaze;
	float3 up = make_float3(0.0f, 1.0f, 0.0f);
	float3 right;
	float apertureRadius;
	float focalDistance;
	float aspectRatio;
	glm::mat4 viewMatrix;
	glm::mat4 projectionMatrix;
	glm::mat4 inverseProjectionMatrix;
	public:
	Camera(){};
	Camera(float3 from_, float3 gazeVector_): from(from_), gaze(gazeVector_){};

	void setfovVertical(float fov);			// set fov in degree
	void setframeSize(int2 size);
	void setcamPos(float3 form);
	void setgazeVector(float3 gazeVector);
	void setupVector(float3 upVector);
	void setrightVector(float3 rightVector);
	void setViewMatrix(glm::mat4 viewMatrix_);
	void setProjectionMatrix(glm::mat4 projectionMatrix_);
	void setInverseProjectionMatrix(glm::mat4 inverseProjectionMatrix_);
	void setapertureRadius(float apertureRadius_);
	void setfocalDistance(float focalDistance_);
	void setaspectRatio(float aspectRatio_);

	float3 getgazeVector();
	float3 getcamPos();
	float3 getupVector();
	float3 getrightVector();
	glm::mat4 getViewMatrix() { return viewMatrix; }
	glm::mat4 getProjectionMatrix() { return projectionMatrix; }
	glm::mat4 getInverseViewMatrix() { return glm::inverse(viewMatrix); }
	glm::mat4 getInverseProjectionMatrix() { return glm::inverse(projectionMatrix); }
	void getViewMatrix(float4* mat);
	void getProjectionMatrix(float4* mat);
	void getInverseViewMatrix(float4* mat, bool transpose);
	void getInverseProjectionMatrix(float4* mat);
	int2 getframeSize();
	float getfovVertical();		// get fov in radian
	float getapertureRadius();
	float getfocalDistance();
	float getaspectRatio();
};
