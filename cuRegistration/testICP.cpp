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
	float MaxneighborDis = 1;//就找最近点设置为FLT_MAX，数据OWL:0.1
	float convergencePrecision = 1e-6;
	int maxTimes = 1000;
	float AverageDis;
	int times;
	glm::mat4 transfromationMat;

	clock_t timeStart = clock();
	clock_t timeEnd = clock();
	timeStart = clock();
	//加载点云
	std::string ext = utilityCore::getFilePathExtension(cloudfixed_filename);
	//定义一个点云变量
	GLMPointCloud* pointcloud;
	GLMPointCloud* pointcloud2;
	int N, N2;
#define FREQ 1 // sample 1 pt from every FREQ pts in original
#define SEP ' '
	if (ext.compare("txt") == 0) {
		pointcloud = new GLMPointCloud(cloudfixed_filename, FREQ, SEP);
		N = pointcloud->getNumPoints();
		timeEnd = clock();
		std::cout << "加载点云时间为：" << timeEnd - timeStart << "ms" << std::endl;
		std::cout << "输入点云的点数为：" << N << std::endl;

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
		std::cout << "加载点云时间为：" << timeEnd - timeStart << "ms" << std::endl;
		std::cout << "输入点云的点数为：" << N2 << std::endl;
	}
	else {
		printf("Non Supported pc1 Format\n");
		return ;
	}
	std::vector<glm::vec3> cloudFixed = pointcloud->getPoints();
	std::vector<glm::vec3> cloudMove = pointcloud2->getPoints();
	//计时开始（自己写的）
	cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cuICP(cloudFixed, cloudMove, cloudOut, MaxneighborDis, convergencePrecision, maxTimes, AverageDis, times, transfromationMat);
	//计时结束（自己写的）
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float   elapsedTime;//ms
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//输出到窗口是要花时间的
	std::cout << "两个点云配准时间为：" << elapsedTime << "ms" << std::endl;
	std::cout << "迭代次数为：" << times << "次" << std::endl;
	std::cout << "对应点平均点距为：" << AverageDis << "mm" << std::endl;
	std::cout << "配准的变换矩阵为："<< std::endl;
	utilityCore::printMat4(transfromationMat);
	return;
}
