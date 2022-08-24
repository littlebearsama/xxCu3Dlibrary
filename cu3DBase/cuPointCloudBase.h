#include"glmpointcloud.h"
#include <vector>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
//�������cuda����
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//ɾ���㣨��������ΪNANֵ��,flagsΪ0����ΪNAN
__global__ void cuGetSubPts(int N, glm::vec3* pts_in_dev, glm::vec3* pts_out_dev, int* flags, int flag_value);

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
};


// Multiplies a glm::mat4 matrix and a vec4.
__host__ __device__ static
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
	return glm::vec3(m * v);
}

// �����������㣬����������ε���С��Χ��
__host__ __device__ static
AABB getAABBForTriangle(const glm::vec3 tri[3]) {
	AABB aabb;
	aabb.min = glm::vec3(
		__min(__min(tri[0].x, tri[1].x), tri[2].x),
		__min(__min(tri[0].y, tri[1].y), tri[2].y),
		__min(__min(tri[0].z, tri[1].z), tri[2].z));
	aabb.max = glm::vec3(
		__max(__max(tri[0].x, tri[1].x), tri[2].x),
		__max(__max(tri[0].y, tri[1].y), tri[2].y),
		__max(__max(tri[0].z, tri[1].z), tri[2].z));
	return aabb;
}

//�����з��ŵ����������
__host__ __device__ static
float calculateSignedArea(const glm::vec3 tri[3]) {
	return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

//�����������긨������
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const glm::vec3 tri[3]) {
	glm::vec3 baryTri[3];
	baryTri[0] = glm::vec3(a, 0);
	baryTri[1] = glm::vec3(b, 0);
	baryTri[2] = glm::vec3(c, 0);
	return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

//Calculate barycentric coordinates. 
//������������
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 tri[3], glm::vec2 point) {
	float beta = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
	float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
	float alpha = 1.0 - beta - gamma;
	return glm::vec3(alpha, beta, gamma);
}

// Check if a barycentric coordinate is within the boundaries of a triangle. 
// �ж�һ�����Ƿ��ڵ�λ�����巶Χ��
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
	return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
		barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
		barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

//For a given barycentric coordinate, compute the corresponding z position��(i.e. depth) on the triangle.
//����һ����������Ӧ���������ϵ�͸��ͶӰ��Zֵ�����ֵ����
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
	return -(barycentricCoord.x * tri[0].z
		+ barycentricCoord.y * tri[1].z
		+ barycentricCoord.z * tri[2].z);
}

//����һ����������Ӧ���������ϵ�͸��ͶӰ��Zֵ�����ֵ����
__host__ __device__ static
float computeOneOverZ(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
	return 1.0f / (
		barycentricCoord.x / tri[0].z
		+ barycentricCoord.y / tri[1].z
		+ barycentricCoord.z / tri[2].z);
}

__host__ __device__ static
glm::vec3 correctCoordPerspective(const float z, glm::vec3 barycentricCoord, const glm::vec3 tri[3], const glm::vec3 coord[3]) {
	return z * glm::vec3(
		coord[0] * barycentricCoord.x / tri[0].z +
		coord[1] * barycentricCoord.y / tri[1].z +
		coord[2] * barycentricCoord.z / tri[2].z);
}

