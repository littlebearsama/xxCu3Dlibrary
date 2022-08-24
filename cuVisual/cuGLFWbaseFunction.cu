#include "cuGLFWbaseFunction.h"
#include <cuda_runtime.h>
#include "../cu3DBase/checkCUDAError.h"
#include <device_launch_parameters.h>//解决不能识别threadId等内部变量的问题
#include <string>


// 复制顶点 glm::vec3 *pos ----> float *vbo
__global__ void kernCopyPositionsToVBO(int N, int offset, glm::vec3 *pos, float *vbo, float c_scale) {
	//当前线程的数据的数组元素的索引
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index < N) {
		vbo[4 * (index + offset) + 0] = pos[index].x * c_scale;
		vbo[4 * (index + offset) + 1] = pos[index].y * c_scale;
		vbo[4 * (index + offset) + 2] = pos[index].z * c_scale;
		vbo[4 * (index + offset) + 3] = 1.0f;
	}
}

// 复制颜色  glm::vec3 color ----> float *vbo
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
	//计算需要block的个数，已知blockSize=128
	dim3 fullBlocksPerGrid_fixed((numObjects_fixed + blockSize - 1) / blockSize);  //用于处理fixed点云的核函数的BLOCK的个数
	dim3 fullBlocksPerGrid_rotated((numObjects_rotated + blockSize - 1) / blockSize);//用于处理rotated点云的核函数的BLOCK的个数
	//1.
	kernCopyPositionsToVBO << <fullBlocksPerGrid_fixed, blockSize >> >(numObjects_fixed, 0, dev_pos_fixed, vbodptr_positions, c_scale);
	checkCUDAError("copyPositionsFixed failed!");
	//2.绿色
	kernCopyColorsToVBO << <fullBlocksPerGrid_fixed, blockSize >> >(numObjects_fixed, 0, glm::vec3(0.0f, 1.0f, 0.0f), vbodptr_velocities);
	checkCUDAError("copyColorsFixed failed!");
	//3.
	kernCopyPositionsToVBO << < fullBlocksPerGrid_rotated, blockSize >> >(numObjects_rotated, numObjects_fixed, dev_pos_rotated, vbodptr_positions, c_scale);
	checkCUDAError("copyPositionsRotated failed!");
	//4.蓝色
	kernCopyColorsToVBO << < fullBlocksPerGrid_rotated, blockSize >> >(numObjects_rotated, numObjects_fixed, glm::vec3(0.0f, 0.0f, 1.0f), vbodptr_velocities);
	checkCUDAError("copyColorsRotated failed!");

	cudaDeviceSynchronize();//等待所有核函数执行完
}


void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities, std::vector<glm::vec3*>pointsVec_dev, std::vector< glm::vec3>colors, std::vector<int>sizes, int blockSize, float c_scale)
{
	size_t numOffset = 0;
	for (size_t i = 0; i < sizes.size(); i++)
	{
		int num = sizes[i];
		glm::vec3 color = colors[i];
		//计算需要block的个数，已知blockSize=128
		dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);  //核函数的BLOCK的个数
		kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(num, numOffset, pointsVec_dev[i], vbodptr_positions, c_scale);//复制点云
		checkCUDAError("copyPositions failed!");
		kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> >(num, numOffset, color, vbodptr_velocities);//复制颜色
		checkCUDAError("copyColorsFixed failed!");
		cudaDeviceSynchronize();//等待所有核函数执行完
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
		//计算需要block的个数，已知blockSize=128
		dim3 fullBlocksPerGrid((num + blockSize - 1) / blockSize);  //核函数的BLOCK的个数
		kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(num, numOffset, pointsVec_dev[i], vbodPositionsPtr, c_scale);//复制点云
		checkCUDAError("copyPositions failed!");
		kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> >(num, numOffset, color, vbodVelocitiesPtr);//复制颜色
		checkCUDAError("copyColorsFixed failed!");
		cudaDeviceSynchronize();//等待所有核函数执行完
		numOffset += num;
	}
}

void setPointsCUDATOVBO(float *vbodptr_positions, float *vbodptr_velocities)
{
	vbodPositionsPtr = vbodptr_positions;
	vbodVelocitiesPtr = vbodptr_velocities;
}