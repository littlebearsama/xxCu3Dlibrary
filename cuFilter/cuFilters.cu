#include"cuFilters.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "cuda.h"
#include "../cu3DBase/cuPointCloudBase.h"
#include <iostream>
#include"../cuRegistration/cuRegistrationBase.h"
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>   
#include <thrust/execution_policy.h>
#include <device_launch_parameters.h>//�������ʶ��threadId���ڲ�����������


__global__ void getXYZarray(int N, glm::vec3* pts, float *Array,int axis)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N)
	{
		if (axis == 1)
		{
			Array[index] = pts[index].x;
		}
		else if (axis == 2)
		{
			Array[index] = pts[index].y;
		}
		else if (axis == 3)
		{
			Array[index] = pts[index].z;
		}
	}
}
//��ֵΪNAN��flags��Ϊ1��������Ϊ0
__global__ void getNANpts(int N, float*Z, int* flags)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if (index < N)
	{
		flags[index] = isnan(Z[index]) ? 1 : 0;
	}
}
__global__ void setNANzeros(int N, float*Z_noNAN, float*Z, int* flags,int flag_vaule)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < N)
	{
		Z_noNAN[index] = (flags[index] == flag_vaule) ? 0 : Z[index];
	}
}
__global__ void getInlierZs(int N, float*Z, int*flags,int flag_value)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index < N)
	{
		Z[index] = (flags[index] == flag_value) ? Z[index] : NAN;
	}
}
//flags:0 ԭ����NANֵ�ĵ� 1:�ڵ� 2�����
//����������У���ߣ�����flags���У����������ֵ����Ϊ��ǰ��ѯ����inliers���ٵĺϸ������ĸ����������������ֵ����Ϊʲô����ʲô��Χ���Ǻϸ������㣩
__global__ void deleleOutliersDepthData3X3(
	const float* depth_dev,
	int width,
	int height,
	int* flags_dev,
	int neighborthresh,
	float depthThresh)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int N = width*height;
	int x = index%width;
	int y = index/width;
	int offset = x + y*width;
	if (index < N)
	{
		//ÿ���������ص�Zֵ
		float t, l, c, r, b, lt, rt, lb, rb;//���������£�����������������
		//������߽������ĵ���ΪNAN��
		c = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//�м�
		//�����ǰ����NAN��,��־λ����Ϊ0������
		if (isnan(c))
		{
			flags_dev[offset] = 0;
			return;
		}
		//���߽���ΪNAN
		offset = (x - 1) + (y - 1)*width;
		lt = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//����
		offset = (x)+(y - 1)*width;
		t = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//��
		offset = (x + 1) + (y - 1)*width;
		rt = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//����
		offset = (x - 1) + (y)*width;
		l = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//��
		offset = (x + 1) + (y)*width;
		r = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//��
		offset = (x - 1) + (y + 1)*width;
		lb = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//����
		offset = (x)+(y + 1)*width;
		b = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//��
		offset = (x + 1) + (y + 1)*width;
		rb = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//����

		//�ж�ÿ�����ڵ��Ƿ��Ǹ߶��Ǹ߶ȲΧ�ڵĵ�
		int countNAN = 0;
		if (!isnan(lt))
		{
			if (abs(lt - c) > depthThresh)
			{
				lt = NAN;
				countNAN++;
			}
		}
		else
			countNAN++;
		//��
		if (!isnan(t))
		{
			if (abs(t - c) > depthThresh)
			{ 
				t = NAN;
				countNAN++;
			}
		}
		else
			countNAN++;
		//����
		if (!isnan(rt))
		{
			if (abs(rt - c) > depthThresh)
			{
				rt = NAN;
			}
		}
		else
			countNAN++;
		//��
		if (!isnan(l))
		{
			if (abs(l - c) > depthThresh)
			{ 
				l = NAN;
				countNAN++;
			}
		}
		else
			countNAN++;
		//��
		if (!isnan(r))
		{
			if (abs(r - c) > depthThresh)
			{ 
				r = NAN;
				countNAN++;
			}
		}
		else
			countNAN++;
		//����
		if (!isnan(lb))
		{
			if (abs(lb - c) > depthThresh)
			{ 
				lb = NAN;
				countNAN++;
			}
		}
		else
			countNAN++;
		//��
		if (!isnan(b))
		{
			if (abs(b - c) > depthThresh)
			{ 
				b = NAN;
				countNAN++;
			}
		}
		else
			countNAN++;
		//
		if (!isnan(rb))
		{
			if (abs(rb - c) > depthThresh)
			{ 
				rb = NAN;
				countNAN++;
			}
		}
		else
			countNAN++;

		offset = x + y*width;
		//������
		if ((8 - countNAN) >= neighborthresh)
		{
			flags_dev[offset] = 1;
		}
		//�߶�ֵ�쳣�ĵ�
		else
		{
			flags_dev[offset] = 2;
		}

	}
}
//Winszie=5 neighborthresh=12����
__global__ void deleleOutliersDepthDataNXN(
	const float* depth_dev,
	int width,
	int height,
	int* flags_dev,
	int Winszie,
	int neighborthresh,
	float depthThresh)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int N = width*height;
	int x = index%width;
	int y = index / width;
	int offset = x + y*width;
	if (index < N)
	{
		//ÿ���������ص�Zֵ
		int numofneighbor = Winszie*Winszie;
		float *values = new float[numofneighbor];
		float center = (offset >= 0 && offset < N) ? depth_dev[offset] : NAN;//�м�Ĳ�ѯ��
		if (isnan(center))
		{
			flags_dev[offset] = 0;
			delete[] values;
			values = nullptr;
			return;
		}
		//������߽������ĵ���ΪNAN��
		for (int i = 0; i < numofneighbor; i++)
		{ 
			int curr_index = offset - numofneighbor / 2 + i;
			values[i] = (curr_index >= 0 && curr_index < N) ? depth_dev[curr_index] : NAN;
		}

		//�ж�ÿ�����ڵ��Ƿ��Ǹ߶��Ǹ߶ȲΧ�ڵĵ�
		int countNAN = 0;
		int countOutliers = 0;
		for (int i = 0; i < numofneighbor; i++)
		{
			float curr_value = values[i];
			if (!isnan(curr_value))
			{
				if (abs(curr_value - center) > depthThresh)
				{
					countOutliers++;
				}
			}
			else
			{
				countNAN++;
			}
		}

		//������(�����Լ�)
		if ((numofneighbor - (countNAN + countOutliers) )>= neighborthresh)
		{
			flags_dev[offset] = 1;
		}
		//���
		else
		{
			flags_dev[offset] = 2;
		}
		delete[] values;
		values = nullptr;
	}
}
//Aixs=1-->X Aixs=2-->Y Aixs=3-->Z
__global__ void passthroughFilter(glm::vec3* pts_dev, int N, int* flags_dev, int Aixs, float max, float min)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N)
	{
		
		if (Aixs==1)
		{
			auto value = pts_dev[index].x;
			if (!isnan(value))
			{
				flags_dev[index] = (value >= min&&value <= max) ? 1 : 2;//1Ϊ��dian��2Ϊ���
			}
			else//��Ч��
			{
				flags_dev[index] = 0;
			}
		}
		else if (Aixs == 2)
		{
			auto value = pts_dev[index].y;
			if (!isnan(value))
			{
				flags_dev[index] = (value >= min&&value <= max) ? 1 : 2;//1Ϊ��dian��2Ϊ���
			}
			else//��Ч��
			{
				flags_dev[index] = 0;
			}

		}
		else if (Aixs == 3)
		{
			auto value = pts_dev[index].z;
			if (!isnan(value))
			{
				flags_dev[index] = (value >= min&&value <= max) ? 1 : 2;//1Ϊ��dian��2Ϊ���
			}
			else//��Ч��
			{
				flags_dev[index] = 0;
			}

		}
	}

}

__global__ void passthroughZFilter(float* depth_dev, int N, int* flags_dev, float max, float min)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N)
	{
		float value = depth_dev[index];
		if (!isnan(value))
		{
			flags_dev[index] = (value >= min&&value <= max) ? 1 : 2;//1Ϊ��dian��2Ϊ���
		}
		else//��Ч��
		{
			flags_dev[index] = 0;
		}
	}

}

void cuFilter_passthroughZ(GLMPointCloud& cloudin_host, int N, GLMPointCloud& cloudout_host, GLMPointCloud& outliercloud_host,float Z_up, float Z_down)
{
	int blockSize = 128;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	glm::vec3* pts_dev = nullptr;
	glm::vec3* outliers_dev = nullptr;
	float* z_dev = nullptr;

	std::vector<glm::vec3>* ptspointer_host = cloudin_host.getPtspointer();
	cudaMalloc((void**)&pts_dev, sizeof(glm::vec3)*N);
	cudaMalloc((void**)&outliers_dev, sizeof(glm::vec3)*N);

	cudaMemcpy(pts_dev, ptspointer_host->data(), sizeof(glm::vec3)*N, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&z_dev, sizeof(float)*N);
	getXYZarray << <fullBlocksPerGrid, blockSize >> >(N, pts_dev, z_dev, 3);
	cudaDeviceSynchronize();
	//1.0 ֱͨ�˲�
	int* nanFlags_dev = nullptr;
	cudaMalloc((void**)&nanFlags_dev, N*sizeof(int));
	getNANpts << <fullBlocksPerGrid, blockSize >> >(N, z_dev, nanFlags_dev);
	cudaDeviceSynchronize();

	thrust::device_ptr<int> thrust_nanflags(nanFlags_dev);
	int num_of_nan = thrust::count(thrust_nanflags, thrust_nanflags + N, 1);  //ͳ���ж��ٸ�NAN��
	cudaDeviceSynchronize();

	float* Z_noNAN_dev = nullptr;
	cudaMalloc((void**)&Z_noNAN_dev, N*sizeof(float));
	setNANzeros << <fullBlocksPerGrid, blockSize >> >(N, Z_noNAN_dev, z_dev, nanFlags_dev, 1);
	cudaDeviceSynchronize();
	thrust::device_ptr<float> thrust_AverageZ(z_dev);
	float AverageZ = thrust::reduce(thrust_AverageZ, thrust_AverageZ + N, (float)0.0f, thrust::plus <float >());
	cudaDeviceSynchronize();

	AverageZ = AverageZ / (N - num_of_nan);
	float Z_max = Z_up - AverageZ;
	float Z_min = Z_down - AverageZ;
	int *inlier_flags = nullptr;
	cudaMalloc((void**)&inlier_flags, N*sizeof(int));
	passthroughZFilter << <fullBlocksPerGrid, blockSize >> > (z_dev, N, inlier_flags, Z_max, Z_min);//�ڵ�Ϊ1�����Ϊ2
	cudaDeviceSynchronize();

	cuGetSubPts << <fullBlocksPerGrid, blockSize >> >(N, pts_dev, outliers_dev, inlier_flags, 2);
	cuGetSubPts << <fullBlocksPerGrid, blockSize >> >(N, pts_dev, pts_dev, inlier_flags, 1);

	//����Host�ڴ�
	glm::vec3* pts_host = new glm::vec3[N];
	cudaMemcpy(pts_host, pts_dev, N*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cloudout_host.SetPointCloud(N, pts_host);
	glm::vec3* outliers_host = new glm::vec3[N];
	cudaMemcpy(outliers_host, outliers_dev, N*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	outliercloud_host.SetPointCloud(N, outliers_host);

	//�ͷ��ڴ�
	cudaFree(outliers_dev);
	outliers_dev = nullptr;
	cudaFree(pts_dev);
	pts_dev = nullptr;
	cudaFree(z_dev);
	z_dev = nullptr;
	cudaFree(inlier_flags);
	inlier_flags = nullptr;
	cudaFree(nanFlags_dev);
	nanFlags_dev = nullptr;
	cudaFree(Z_noNAN_dev);
	Z_noNAN_dev = nullptr;

}
//1.0
void cuFilter_DeleleOutliers(GLMPointCloud& cloudin_host, int width, int height, GLMPointCloud& cloudout_host, GLMPointCloud& outliercloud_host, int neighborWin, int NeighborThresh, float depthThresh)
{
	int blockSize = 128;
	int N = width*height;
	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	glm::vec3* pts_dev = nullptr;
	glm::vec3* outliers_dev = nullptr;
	float* z_dev = nullptr;

	std::vector<glm::vec3>* ptspointer_host = cloudin_host.getPtspointer();
	cudaMalloc((void**)&pts_dev, sizeof(glm::vec3)*N);
	cudaMalloc((void**)&outliers_dev, sizeof(glm::vec3)*N);

	//cudaMemcpy(pts_dev, &(*ptspointer_host)[0], sizeof(glm::vec3)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(pts_dev, ptspointer_host->data(), sizeof(glm::vec3)*N, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&z_dev, sizeof(float)*N);
	getXYZarray << <fullBlocksPerGrid, blockSize >> >(N, pts_dev, z_dev, 3);
	cudaDeviceSynchronize();

	//2.0 ���������޳����
	int* flags_dev;
	cudaMalloc((void**)&flags_dev, N*sizeof(int));//Ϊ������������ڴ�
	//deleleOutliersDepthData3X3 << <fullBlocksPerGrid, blockSize >> >(z_dev, width, height, flags_dev, NeighborThresh, depthThresh);
	deleleOutliersDepthDataNXN << <fullBlocksPerGrid, blockSize >> >(z_dev, width, height, flags_dev, neighborWin,NeighborThresh, depthThresh);
	cuGetSubPts << <fullBlocksPerGrid, blockSize >> >(N, pts_dev, outliers_dev, flags_dev, 2);
	cuGetSubPts << <fullBlocksPerGrid, blockSize >> >(N, pts_dev, pts_dev, flags_dev, 1);
 
	//����Host�ڴ�
	glm::vec3* pts_host = new glm::vec3[N];
	cudaMemcpy(pts_host, pts_dev, N*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	cloudout_host.SetPointCloud(N, pts_host);
	glm::vec3* outliers_host = new glm::vec3[N];
	cudaMemcpy(outliers_host, outliers_dev, N*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	outliercloud_host.SetPointCloud(N, outliers_host);


	//�ͷ��ڴ�
	cudaFree(outliers_dev);
	outliers_dev = nullptr;
	cudaFree(pts_dev);
	pts_dev = nullptr;
	cudaFree(z_dev);
	z_dev = nullptr;
	cudaFree(flags_dev);
	flags_dev = nullptr;

}