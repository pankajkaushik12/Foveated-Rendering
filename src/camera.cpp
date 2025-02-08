#include "camera.h"

//Camera camera;

void Camera::setcamPos(float3 from_){
    from = from_;
}

void Camera::setframeSize(int2 size) {
	frameSize = size;
}

void Camera::setfovVertical(float fov_) {
	fovVertical = fov_;
}

void Camera::setgazeVector(float3 lookat_) {
	gaze = lookat_;
}

void Camera::setupVector(float3 upVector_) {
	up = upVector_;
}

void Camera::setrightVector(float3 rightVector_) {
	right = rightVector_;
}

void Camera::setViewMatrix(glm::mat4 viewMatrix_) {
	viewMatrix = viewMatrix_;
}

void Camera::setProjectionMatrix(glm::mat4 projectionMatrix_) {
	projectionMatrix = projectionMatrix_;
	inverseProjectionMatrix = glm::inverse(projectionMatrix);
}

void Camera::setInverseProjectionMatrix(glm::mat4 inverseProjectionMatrix_) {
	inverseProjectionMatrix = inverseProjectionMatrix_;
}

void Camera::setapertureRadius(float apertureRadius_) {
	apertureRadius = apertureRadius_;
}

void Camera::setfocalDistance(float focalDistance_) {
	focalDistance = focalDistance_;
}

void Camera::setaspectRatio(float aspectRatio_) {
	aspectRatio = aspectRatio_;
}

float3 Camera::getcamPos(){
    return from;
}

int2 Camera::getframeSize() {
	return frameSize;
}

float Camera::getfovVertical() {
	return fovVertical * M_PI / 180.f;
}

float3 Camera::getgazeVector() {
	return gaze;
}

float3 Camera::getupVector() {
	return up;
}

float3 Camera::getrightVector() {
	return right;
}

void Camera::getViewMatrix(float4* viewMatrix_) {
	viewMatrix_[0] = make_float4(viewMatrix[0].x, viewMatrix[1].x, viewMatrix[2].x, viewMatrix[3].x);
	viewMatrix_[1] = make_float4(viewMatrix[0].y, viewMatrix[1].y, viewMatrix[2].y, viewMatrix[3].y);
	viewMatrix_[2] = make_float4(viewMatrix[0].z, viewMatrix[1].z, viewMatrix[2].z, viewMatrix[3].z);
	viewMatrix_[3] = make_float4(viewMatrix[0].w, viewMatrix[1].w, viewMatrix[2].w, viewMatrix[3].w);
}

void Camera::getProjectionMatrix(float4* projectionMatrix_) {
	projectionMatrix_[0] = make_float4(projectionMatrix[0].x, projectionMatrix[1].x, projectionMatrix[2].x, projectionMatrix[3].x);
	projectionMatrix_[1] = make_float4(projectionMatrix[0].y, projectionMatrix[1].y, projectionMatrix[2].y, projectionMatrix[3].y);
	projectionMatrix_[2] = make_float4(projectionMatrix[0].z, projectionMatrix[1].z, projectionMatrix[2].z, projectionMatrix[3].z);
	projectionMatrix_[3] = make_float4(projectionMatrix[0].w, projectionMatrix[1].w, projectionMatrix[2].w, projectionMatrix[3].w);
}

void Camera::getInverseViewMatrix(float4* viewMatrix_, bool transpose) {
	glm::mat4 temp = glm::inverse(viewMatrix);
	temp = transpose ? glm::transpose(temp) : temp;			// We need viewMatrix_ to be a row-majored matrix
	viewMatrix_[0] = make_float4(temp[0].x, temp[1].x, temp[2].x, temp[3].x);
	viewMatrix_[1] = make_float4(temp[0].y, temp[1].y, temp[2].y, temp[3].y);
	viewMatrix_[2] = make_float4(temp[0].z, temp[1].z, temp[2].z, temp[3].z);
	viewMatrix_[3] = make_float4(temp[0].w, temp[1].w, temp[2].w, temp[3].w);
}

void Camera::getInverseProjectionMatrix(float4* projectionMatrix_) {
	glm::mat4 temp = inverseProjectionMatrix;
	projectionMatrix_[0] = make_float4(temp[0].x, temp[1].x, temp[2].x, temp[3].x);
	projectionMatrix_[1] = make_float4(temp[0].y, temp[1].y, temp[2].y, temp[3].y);
	projectionMatrix_[2] = make_float4(temp[0].z, temp[1].z, temp[2].z, temp[3].z);
	projectionMatrix_[3] = make_float4(temp[0].w, temp[1].w, temp[2].w, temp[3].w);
}

float Camera::getapertureRadius() {
	return apertureRadius;
}

float Camera::getfocalDistance() {
	return focalDistance;
}

float Camera::getaspectRatio() {
	return aspectRatio;
}
