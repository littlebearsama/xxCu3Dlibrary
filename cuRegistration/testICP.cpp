#include "testICP.h"
#include "cuICP.h"
#include <cuda_runtime.h>
#include <iostream>
#include "../cu3DBase/utilityCore.hpp"
#include <string>
#include "../cu3DBase/glmpointcloud.h"
#include <time.h>

void cuICPSample(const std::string& cloudfixed_filename, const std::string& cloudmove_filename)
{
	std::vector<glm::vec3> cloudOut;
	float MaxneighborDis = 1;//�������������ΪFLT_MAX������OWL:0.1
	float convergencePrecision = 1e-6;
	int maxTimes = 1000;
	float AverageDis;
	int times;
	glm::mat4 transfromationMat;

	clock_t timeStart = clock();
	clock_t timeEnd = clock();
	timeStart = clock();
	//���ص���
	std::string ext = utilityCore::getFilePathExtension(cloudfixed_filename);
	//����һ�����Ʊ���
	GLMPointCloud* pointcloud;
	GLMPointCloud* pointcloud2;
	int N, N2;
#define FREQ 1 // sample 1 pt from every FREQ pts in original
#define SEP ' '
	if (ext.compare("txt") == 0) {
		pointcloud = new GLMPointCloud(cloudfixed_filename, FREQ, SEP);
		N = pointcloud->getNumPoints();
		timeEnd = clock();
		std::cout << "���ص���ʱ��Ϊ��" << timeEnd - timeStart << "ms" << std::endl;
		std::cout << "������Ƶĵ���Ϊ��" << N << std::endl;

	}
	else {
		printf("Non Supported pc1 Format\n");
		return ;
	}



	timeStart = clock();
	std::string ext2 = utilityCore::getFilePathExtension(cloudmove_filename);
	if (ext2.compare("txt") == 0) {
		pointcloud2 = new GLMPointCloud(cloudmove_filename, FREQ, SEP);
		N2 = pointcloud2->getNumPoints();
		timeEnd = clock();
		std::cout << "���ص���ʱ��Ϊ��" << timeEnd - timeStart << "ms" << std::endl;
		std::cout << "������Ƶĵ���Ϊ��" << N2 << std::endl;
	}
	else {
		printf("Non Supported pc1 Format\n");
		return ;
	}
	std::vector<glm::vec3> cloudFixed = pointcloud->getPoints();
	std::vector<glm::vec3> cloudMove = pointcloud2->getPoints();
	//��ʱ��ʼ���Լ�д�ģ�
	cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cuICP(cloudFixed, cloudMove, cloudOut, MaxneighborDis, convergencePrecision, maxTimes, AverageDis, times, transfromationMat);
	//��ʱ�������Լ�д�ģ�
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float   elapsedTime;//ms
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//�����������Ҫ��ʱ���
	std::cout << "����������׼ʱ��Ϊ��" << elapsedTime << "ms" << std::endl;
	std::cout << "��������Ϊ��" << times << "��" << std::endl;
	std::cout << "��Ӧ��ƽ�����Ϊ��" << AverageDis << "mm" << std::endl;
	std::cout << "��׼�ı任����Ϊ��"<< std::endl;
	utilityCore::printMat4(transfromationMat);
	return;
}
