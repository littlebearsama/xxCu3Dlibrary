#include "cuICP.h"
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>//�������ʶ��threadId���ڲ�����������
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include "../cu3DBase/checkCUDAError.h"
#include <time.h>
#include "../cu3DBase/utilityCore.hpp"
#include "cuRegistrationBase.h"//��������Ļ�������
#include "../cu3DBase/glmpointcloud.h"//������

#include "../cuVisual/cuGLFWbaseFunction.h"//���ӻ�����

//���ӻ�
//#include"cuGLFWbaseFunction.h"

#define CUICP_DEBUG 0
#define CUICP_VISUAL 0

#pragma region CUICP

//TODO��
//�������Ż��Ĳ��֣�ѡȡ���ٵĽ���KDtree
bool cuICP(std::vector<glm::vec3>& cloudFixed, 
	std::vector<glm::vec3>& cloudMoved, 
	std::vector<glm::vec3>& cloudOut, 
	float neighbourDistanceThreshold, 
	float convergencePrecision, 
	int maxTimes, 
	float& AverageDis, 
	int &times, 
	glm::mat4& transfromationMat)
{
	//1.����GPU�ڴ�
	float scene_scale = 1.0f;
	int blockSize = 128;
	float DistanceThreshold = neighbourDistanceThreshold;
	// ��׼��ʼ��:
	//1. �����ڴ�
	int numObjects_fixed = 0;
	int numObjects_rotated = 0;
	glm::vec3*dev_pos_fixed = nullptr;
	glm::vec3*dev_pos_rotated = nullptr;
	glm::vec3*dev_pos_corr = nullptr;
	glm::vec3*dev_pos_rotated_centered = nullptr;
	glm::vec3*dev_pos_A = nullptr;
	glm::vec3*dev_pos_B = nullptr;
	glm::mat3*dev_w = nullptr;
	float *corr_distance = nullptr;
	int  *flags = nullptr;
	Node *dev_kd = nullptr;

	numObjects_fixed = (int)cloudFixed.size();
	numObjects_rotated = (int)cloudMoved.size();
	//2. ����Դ���ƺͲο����Ƶ�ȫ���ڴ�
	dim3 fullBlocksPerGrid_fixed((numObjects_fixed + blockSize - 1) / blockSize);
	dim3 fullBlocksPerGrid_rotated((numObjects_rotated + blockSize - 1) / blockSize);

	//�����ڴ�
	cudaMalloc((void**)&dev_pos_fixed, numObjects_fixed * sizeof(glm::vec3));           //reference scan��������Ϊglm::vec3��
	cudaMalloc((void**)&dev_pos_rotated, numObjects_rotated * sizeof(glm::vec3));       //source scan��������Ϊglm::vec3��     
	cudaMalloc((void**)&dev_pos_corr, numObjects_rotated * sizeof(glm::vec3));          //��Ӧ��㼯��������Ϊglm::vec3��
	cudaMalloc((void**)&dev_pos_rotated_centered, numObjects_rotated * sizeof(glm::vec3));//�Ƶ����ĺ��source scan
	cudaMalloc((void**)&dev_w, numObjects_rotated * sizeof(glm::mat3));                 //����3X3����,ÿһ���Ӧ���3X3����
	cudaMalloc((void**)&flags, numObjects_rotated * sizeof(int));
	cudaMalloc((void**)&corr_distance, numObjects_rotated * sizeof(float));
	//���븨���ڴ�
	cudaMalloc((void**)&dev_pos_A, numObjects_rotated * sizeof(glm::vec3));//�������ڵĶ�Ӧ��
	cudaMalloc((void**)&dev_pos_B, numObjects_rotated * sizeof(glm::vec3));//�������ڵĶ�Ӧ��
	cudaMemcpy(dev_pos_fixed, &cloudFixed[0], numObjects_fixed * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pos_rotated, &cloudMoved[0], numObjects_rotated * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	if (CUICP_DEBUG)
	{
#include <iostream>
		std::cout << "����������Ƶ�Xֵ��" << std::endl;
		for (int i = 0; i < 100; i++)
		{
			std::cout << cloudFixed[i].x << " ";
		}
		std::cout << "GPU�еĵ��Ƶ�Xֵ��" << std::endl;
		glm::vec3 *host_fixed = new glm::vec3[numObjects_fixed];
		cudaMemcpy(host_fixed, dev_pos_fixed, numObjects_fixed*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
		for (int i = 0; i < 100; i++)
		{
			std::cout << host_fixed[i].x << " ";
		}
		delete[] host_fixed;
		host_fixed = nullptr;
	}
	

	//Ϊkdtree����ȫ���ڴ�
	cudaMalloc((void**)&dev_kd, numObjects_fixed * sizeof(Node));
	//���Զ�����๹�캯������kdtree
	clock_t timeStart = clock();
	KDTree kdtree(cloudFixed, scene_scale);//ʹ��pt����ʼ��kdtree
	std::vector<Node> tree = kdtree.getTree();
	clock_t timeEnd = clock();
	//std::cout << std::endl << std::endl << "����KDtreeʱ��Ϊ��" << timeEnd - timeStart << "ms" << std::endl << std::endl;
	//Ȼ���Ƶ�ȫ���ڴ���
	cudaMemcpy(dev_kd, &tree[0], numObjects_fixed * sizeof(Node), cudaMemcpyHostToDevice);

	//��������
	if (abs(scene_scale - 1.0f) > 1e-6){
		scalingVec3Array << < fullBlocksPerGrid_fixed, blockSize >> > (numObjects_fixed, dev_pos_fixed, scene_scale);
		scalingVec3Array << < fullBlocksPerGrid_rotated, blockSize >> > (numObjects_rotated, dev_pos_rotated, scene_scale);
	}
	//��rotated���Ƹ���Ϊ���ݼ���block�ĸ���
	dim3 fullBlocksPerGrid((numObjects_rotated + blockSize - 1) / blockSize);//����block�ĸ���

	int timesCount = 0;
	float curr_AverageDis = 0;
	float last_AverageDis = 0;
	glm::mat4 FinalTransformation = glm::mat4(1.0f);
	//��ʼѭ��
	while (1)
	{
		if (maxTimes <= timesCount)
			break;

		//���ҽ���
		getNearestNeighborKDTree << <fullBlocksPerGrid, blockSize >> > (numObjects_rotated, dev_pos_rotated,
			dev_kd, dev_pos_corr, corr_distance);
		checkCUDAError("Find nearest Neighbor");
		//set_one2 << <fullBlocksPerGrid, blockSize >> >(numObjects_rotated, flags);//ȫ����Ϊ1���ҳ������ж�Ӧ�㶼�ǽ��ڵ�
		isNeighbourhood_setflags << < fullBlocksPerGrid, blockSize >> > (numObjects_rotated, flags, corr_distance, DistanceThreshold);
		//ͳ��flags����1�ĸ������õ��ڵ������
		thrust::device_ptr<int> thrust_flags(flags);
		int inliers_count = thrust::reduce(thrust_flags, thrust_flags + numObjects_rotated);
		//�����Ƕ�Ӧ��ĵ���Ϊ��0��0��0�����Լ���Э�������û��Ӱ�죻
		getInlierPointset << <fullBlocksPerGrid, blockSize >> >(numObjects_rotated, flags, dev_pos_rotated, dev_pos_A);
		getInlierPointset << <fullBlocksPerGrid, blockSize >> >(numObjects_rotated, flags, dev_pos_corr, dev_pos_B);
		if (0)//�����Ӧ���
		{
			if (timesCount==0)//ֻ���һ�ξͺ���
			{
				glm::vec3 *host_pos_A = new glm::vec3[numObjects_rotated];
				glm::vec3 *host_pos_B = new glm::vec3[numObjects_rotated];
				cudaMemcpy(host_pos_A, dev_pos_A, numObjects_rotated*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
				cudaMemcpy(host_pos_B, dev_pos_B, numObjects_rotated*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
				GLMPointCloud pcA, pcB;
				pcA.SetPointCloud(numObjects_rotated, host_pos_A);
				pcB.SetPointCloud(numObjects_rotated, host_pos_B);
				pcA.savePointcloud_without000("host_pos_A.txt");
				pcB.savePointcloud_without000("host_pos_B.txt");
				delete host_pos_A; host_pos_A = nullptr;
				delete host_pos_B; host_pos_B = nullptr;
			}
		}
		//�����ڵ��AverageDis
		thrust::device_ptr<float> thrust_AverageDis(corr_distance);
		curr_AverageDis = thrust::reduce(thrust_AverageDis, thrust_AverageDis + numObjects_rotated, 0.0);
		curr_AverageDis = curr_AverageDis/ numObjects_rotated;
		//std::cout << "curr_AverageDis:" << curr_AverageDis << std::endl;//debug�����ƽ��ǰ���
		if (abs(curr_AverageDis - last_AverageDis) <= convergencePrecision)
			break;
		last_AverageDis = curr_AverageDis;
		timesCount++;
		//std::cout << "��ǰ����������" << timesCount << std::endl;
		//ƽ��
		thrust::device_ptr<glm::vec3> thrust_pos_A(dev_pos_A);
		thrust::device_ptr<glm::vec3> thrust_pos_B(dev_pos_B);
		glm::vec3 pos_A_mean = thrust::reduce(thrust_pos_A, thrust_pos_A + numObjects_rotated,
			glm::vec3(0.f, 0.f, 0.f));//����ͷ��β���ֵ
		glm::vec3 pos_B_mean = thrust::reduce(thrust_pos_B, thrust_pos_B + numObjects_rotated,
			glm::vec3(0.f, 0.f, 0.f));//����ͷ��β���ֵ
		pos_A_mean /= inliers_count;
		pos_B_mean /= inliers_count;
		//����ȥ��������
		glm::mat4 translation_matrix = constructTranslationMatrix2(-pos_A_mean);
		translatePts << <fullBlocksPerGrid, blockSize >> > (inliers_count, dev_pos_A, dev_pos_A, translation_matrix);//�ƶ������Ĵ�
		translation_matrix = constructTranslationMatrix2(-pos_B_mean);
		translatePts << <fullBlocksPerGrid, blockSize >> > (inliers_count, dev_pos_B, dev_pos_B, translation_matrix);
		checkCUDAError("Translating Pts");
		//����w
		calW << < fullBlocksPerGrid, blockSize >> > (inliers_count, dev_pos_A, dev_pos_B, dev_w);
		thrust::device_ptr<glm::mat3> thrust_w(dev_w);
		//��3X3����������й�Լ�õ�������W��3X3����
		glm::mat3 W = thrust::reduce(thrust_w, thrust_w + inliers_count, glm::mat3(0.f));
		checkCUDAError("Calculated W");
		//4.��W����SVD�ֽ⣬����UV
		glm::mat3 S, U, V;
		svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
			U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
			S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
			V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);
		//5.�õ�R��t
		glm::mat3 R = glm::transpose(U) * V;
		glm::vec3 t = pos_B_mean - R * pos_A_mean;
		//R = glm::transpose(U) * V;
		//t = pos_B_mean - R * pos_A_mean;

		glm::mat4 T = glm::translate(glm::mat4(), t);
		//glm::mat4 transformation = T * glm::mat4(R) * scale_matrix;
		//6. R��t�ϳɱ任����T
		glm::mat4 transformation = T * glm::mat4(R);
		FinalTransformation = transformation*FinalTransformation;
		//���ƶ��ĵ��ƽ��б任dev_pos_rotated
		transformPoints << < fullBlocksPerGrid, blockSize >> > (numObjects_rotated, dev_pos_rotated, dev_pos_rotated, transformation);
		checkCUDAError("Transforming Pts");
		if (CUICP_VISUAL)
		{
			std::vector<glm::vec3*>pointsVec_dev;
			pointsVec_dev.push_back(dev_pos_fixed);
			pointsVec_dev.push_back(dev_pos_rotated);
			std::vector< glm::vec3>colors;
			colors.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
			colors.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
			std::vector<int>sizes;
			sizes.push_back(numObjects_fixed);
			sizes.push_back(numObjects_rotated);
			int blockSize=128;
			float c_scale=1.0f;
			copyPointsToVBO(pointsVec_dev, colors, sizes, blockSize, c_scale);//���Ƶ�VBO����
		}
	}
	AverageDis = curr_AverageDis;
	times = timesCount;
	transfromationMat = FinalTransformation;
	//�ͷ��ڴ�
	cudaFree(dev_pos_rotated);
	cudaFree(dev_pos_fixed);
	cudaFree(dev_pos_rotated_centered);
	cudaFree(dev_pos_corr);
	cudaFree(dev_w);
	cudaFree(dev_kd);
	cudaFree(flags);
	cudaFree(corr_distance);
	cudaFree(dev_pos_A);
	cudaFree(dev_pos_B);

	dev_pos_fixed = NULL;
	dev_pos_rotated = NULL;
	dev_pos_corr = NULL;
	dev_pos_rotated_centered = NULL;
	dev_w = NULL;
	flags = NULL;
	corr_distance = NULL;
	dev_pos_A = NULL;
	dev_pos_B = NULL;
	checkCUDAError("registration Free");

	return true;
}

#pragma endregion


#pragma region CPUICP
template <typename T>
T calculate_vector_mean(std::vector<T> input){
	T sum;
	for (auto &element : input){
		sum += element;
	}
	return sum /= input.size();
};

//��ʽת����3X1ƽ��+3X3��ת-->4X4�任
glm::mat4 constructTransformationMatrix(const glm::vec3 &translation, const glm::vec3& rotation, const glm::vec3& scale) {
	glm::mat4 translation_matrix = glm::translate(glm::mat4(), translation);
	glm::mat4 rotation_matrix = glm::rotate(glm::mat4(), rotation.x, glm::vec3(1, 0, 0));
	rotation_matrix *= glm::rotate(glm::mat4(), rotation.y, glm::vec3(0, 1, 0));
	rotation_matrix *= glm::rotate(glm::mat4(), rotation.z, glm::vec3(0, 0, 1));
	glm::mat4 scale_matrix = glm::scale(glm::mat4(), scale);
	return translation_matrix* rotation_matrix * scale_matrix;
}

//��CPU�Ͻ��е���׼
std::vector<glm::vec3> registration_init_cpu(std::vector<glm::vec3> &input){
	glm::mat4 transformation = constructTransformationMatrix(glm::vec3(1.0f, 0.0f, 0.0f),
		glm::vec3(0.4f, 0.4f, -0.2f), glm::vec3(1.0f, 1.0f, 1.0f));
	std::vector<glm::vec3>result(input.size(), glm::vec3(0.f, 0.f, 0.f));
	for (int i = 0; i < input.size(); ++i){
		result[i] = glm::vec3(transformation * glm::vec4(input[i], 1.0f));
	}
	return result;
}

//��CPU�Ͻ��е���׼
// skeleton code for cpu_step; no display, just for performance comparison
void registration_cpu(std::vector<glm::vec3>& target, std::vector<glm::vec3>& source){
	int numPts = source.size();

	std::vector<glm::vec3> corr(numPts, glm::vec3(0.f, 0.f, 0.f));
	//���
	for (int k = 0; k < numPts; k++){
		auto best_dist = glm::distance(source[k], target[0]);
		int i = 0;
		for (int j = 1; j < numPts; j++){
			auto d = glm::distance(source[k], target[j]);
			if (d < best_dist){
				best_dist = d;
				i = j;
			}
		}
		corr[k] = target[i];
	}
	//


	glm::vec3 mean_corr = calculate_vector_mean(corr);
	glm::vec3 mean_source = calculate_vector_mean(source);

	std::vector<glm::vec3> source_centered = source;

	for (int i = 0; i < numPts; i++){
		source_centered[i] = source[i] - mean_source;
		corr[i] -= mean_corr;
	}

	// calculate w
	std::vector<glm::mat3> w(numPts, glm::mat3(glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 0.f)));
	for (int i = 0; i < numPts; i++){
		w[i] = glm::outerProduct(source_centered[i], corr[i]);//�������
	}

	glm::mat3 W = calculate_vector_mean(w);
	W *= numPts;

	glm::mat3 S, U, V;

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]);

	glm::mat3 R = glm::transpose(U) * V;
	glm::vec3 t = mean_corr - R * mean_source;
	glm::mat4 T = glm::translate(glm::mat4(), t);
	glm::mat4 transformation = T * glm::mat4(R);

	for (int i = 0; i < numPts; i++){
		source[i] = glm::vec3(transformation * glm::vec4(source[i], 1.0f));
	}

}

#pragma endregion