#include "cuGLFWbaseFunction.h"
#include <cuda_runtime.h>
#include "../cu3DBase/checkCUDAError.h"
#include <device_launch_parameters.h>//�������ʶ��threadId���ڲ�����������
#include <string>


// ���ƶ��� glm::vec3 *pos ----> float *vbo
__global__ void kernCopyPositionsToVBO(int N, int offset, glm::vec3 *pos, float *vbo, float c_scale) {
	//��ǰ�̵߳����ݵ�����Ԫ�ص�����
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < N) {
		vbo[4 * (index + offset) + 0] = pos[index].x * c_scale;
		vbo[4 * (index + offset) + 1] = pos[index].y * c_scale;
		vbo[4 * (index + offset) + 2] = pos[index].z * c_scale;
		vbo[4 * (index + offset) + 3] = 1.0f;
	}
}

// ������ɫ  glm::vec3 color ----> float *vbo
__global__ void kernCopyColorsToVBO(int N, int offset, glm::vec3 color, float *vbo) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < N) {
		vbo[4 * (index + offset) + 0] = color.x;
		vbo[4 * (index + offset) + 1] = color.y;
		vbo[4 * (index + offset) + 2] = color.z;
		vbo[4 * (index + offset) + 3] = 1.0f;
	}
}

glm::vec3 *dev_pos_fixed = nullptr;
glm::vec3 *dev_pos_rotated = nullptr;

void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities, int numObjects_fixed, int numObjects_rotated, int blockSize, float c_scale) {
	//������Ҫblock�ĸ�������֪blockSize=128
	dim3 fullBlocksPerGrid_fixed((numObjects_fixed + blockSize - 1) / blockSize);  //���ڴ���fixed���Ƶĺ˺�����BLOCK�ĸ���
	dim3 fullBlocksPerGrid_rotated((numObjects_rotated + blockSize - 1) / blockSize);//���ڴ���rotated���Ƶĺ˺�����BLOCK�ĸ���
	//1.
	kernCopyPositionsToVBO << <fullBlocksPerGrid_fixed, blockSize >> >(numObjects_fixed, 0, dev_pos_fixed, vbodptr_positions, c_scale);
	checkCUDAError("copyPositionsFixed failed!");
	//2.��ɫ
	kernCopyColorsToVBO << <fullBlocksPerGrid_fixed, blockSize >> >(numObjects_fixed, 0, glm::vec3(0.0f, 1.0f, 0.0f), vbodptr_velocities);
	checkCUDAError("copyColorsFixed failed!");
	//3.
	kernCopyPositionsToVBO << < fullBlocksPerGrid_rotated, blockSize >> >(numObjects_rotated, numObjects_fixed, dev_pos_rotated, vbodptr_positions, c_scale);
	checkCUDAError("copyPositionsRotated failed!");
	//4.��ɫ
	kernCopyColorsToVBO << < fullBlocksPerGrid_rotated, blockSize >> >(numObjects_rotated, numObjects_fixed, glm::vec3(0.0f, 0.0f, 1.0f), vbodptr_velocities);
	checkCUDAError("copyColorsRotated failed!");

	cudaDeviceSynchronize();//�ȴ����к˺���ִ����
}


void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities, std::vector<glm::vec3*>pointsVec_dev, std::vector< glm::vec3>colors, std::vector<int>sizes, int blockSize, float c_scale)
{
	size_t numOffset = 0;
	for (size_t i = 0; i < sizes.size(); i++)
	{
		int num = sizes[i];
		glm::vec3 color = colors[i];
		//������Ҫblock�ĸ�������֪blockSize=128
		dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);  //�˺�����BLOCK�ĸ���
		kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(num, numOffset, pointsVec_dev[i], vbodptr_positions, c_scale);//���Ƶ���
		checkCUDAError("copyPositions failed!");
		kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> >(num, numOffset, color, vbodptr_velocities);//������ɫ
		checkCUDAError("copyColorsFixed failed!");
		cudaDeviceSynchronize();//�ȴ����к˺���ִ����
		numOffset += num;
	}
}

float *vbodPositionsPtr = nullptr;
float *vbodVelocitiesPtr = nullptr;

void copyPointsToVBO(std::vector<glm::vec3*>pointsVec_dev, std::vector< glm::vec3>colors, std::vector<int>sizes, int blockSize, float c_scale)
{
	size_t numOffset = 0;
	for (size_t i = 0; i < sizes.size(); i++)
	{
		int num = sizes[i];
		glm::vec3 color = colors[i];
		//������Ҫblock�ĸ�������֪blockSize=128
		dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);  //�˺�����BLOCK�ĸ���
		kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(num, numOffset, pointsVec_dev[i], vbodPositionsPtr, c_scale);//���Ƶ���
		checkCUDAError("copyPositions failed!");
		kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> >(num, numOffset, color, vbodVelocitiesPtr);//������ɫ
		checkCUDAError("copyColorsFixed failed!");
		cudaDeviceSynchronize();//�ȴ����к˺���ִ����
		numOffset += num;
	}
}

void setPointsCUDATOVBO(float *vbodptr_positions, float *vbodptr_velocities)
{
	vbodPositionsPtr = vbodptr_positions;
	vbodVelocitiesPtr = vbodptr_velocities;
}