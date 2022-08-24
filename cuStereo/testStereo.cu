#include "testStereo.h"
#include "cuAdcensus.h"
#include "dataCompare.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/highgui.hpp>
//用于存储vector
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
// opencv library
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
void compareAdcensus()
{
	std::string path_left = "left.png";
	std::string path_right = "right.png";
	std::cout << "Image Loading..." << std::endl;
	cv::Mat img_left = cv::imread(path_left, cv::IMREAD_COLOR);
	cv::Mat img_right = cv::imread(path_right, cv::IMREAD_COLOR);
	if (img_left.data == nullptr || img_right.data == nullptr) {
		std::cout << "读取影像失败！" << std::endl;
		return;
	}
	if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
		std::cout << "左右影像尺寸不一致！" << std::endl;
		return;
	}
	//···············································································//
	const int width = static_cast<int>(img_left.cols);
	const int height = static_cast<int>(img_right.rows);

	// 左右影像的彩色数据
	uint8_t* left_rgbs_host = new uint8_t[width * height * 3];
	uint8_t* right_rgbs_host = new uint8_t[width * height * 3];
	uint64_t* censusLeft_host = new uint64_t[width * height];
	uint64_t* censusRight_host = new uint64_t[width * height];
	float * cost_init_host = new float[Disp_Range*width*height];
	float * cost_aggr_host = new float[Disp_Range*width*height];
	float * cost_temp1_host = new float[width*height];
	float * cost_temp2_host = new float[width*height];
	std::vector<float* > vec_cost_tmp;
	vec_cost_tmp.push_back(cost_temp1_host);
	vec_cost_tmp.push_back(cost_temp2_host);
	int* supportPixelcountVf_host = new int[width * height];
	int* supportPixelcountHf_host = new int[width * height];
	std::vector<int* > vec_sup_count;
	vec_sup_count.push_back(supportPixelcountHf_host);
	vec_sup_count.push_back(supportPixelcountVf_host);

	//cpu
	cv::Mat garyImageL_cpuResult = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat garyImageR_cpuResult = cv::Mat::zeros(height, width, CV_8UC1);
	float* disparity = new float[width * height];
	CrossArm* arms_host = new CrossArm[width * height];

	//申请设备内存
	uint8_t *left_rgbs_device;
	uint8_t *right_rgbs_device;
	uint8_t *left_gray_device;
	uint8_t *right_gray_device;
	uint64_t *left_census_device;
	uint64_t *right_census_device;
	float *cost_init_device;
	float *cost_aggr_device;
	float* cost_temp1_device;
	float* cost_temp2_device;
	CrossArm* arms_device;
	int* supportPixelcountVf_device;
	int* supportPixelcountHf_device;
	cudaMalloc((void**)&left_rgbs_device, width*height * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&right_rgbs_device, width*height * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&left_gray_device, width*height * sizeof(uint8_t));
	cudaMalloc((void**)&right_gray_device, width*height * sizeof(uint8_t));
	cudaMalloc((void**)&left_census_device, width*height * sizeof(uint64_t));
	cudaMalloc((void**)&right_census_device, width*height * sizeof(uint64_t));
	cudaMalloc((void**)&cost_init_device, Disp_Range*width*height * sizeof(float));
	cudaMalloc((void**)&cost_aggr_device, Disp_Range*width*height * sizeof(float));
	cudaMalloc((void**)&cost_temp1_device, width*height * sizeof(float));
	cudaMalloc((void**)&cost_temp2_device, width*height * sizeof(float));

	cudaMalloc((void**)&arms_device, width*height * sizeof(CrossArm));
	cudaMalloc((void**)&supportPixelcountVf_device, width*height * sizeof(int));
	cudaMalloc((void**)&supportPixelcountHf_device, width*height * sizeof(int));

	//赋值
#pragma omp for
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			left_rgbs_host[i * 3 * width + 3 * j] = img_left.at<cv::Vec3b>(i, j)[0];
			left_rgbs_host[i * 3 * width + 3 * j + 1] = img_left.at<cv::Vec3b>(i, j)[1];
			left_rgbs_host[i * 3 * width + 3 * j + 2] = img_left.at<cv::Vec3b>(i, j)[2];
			right_rgbs_host[i * 3 * width + 3 * j] = img_right.at<cv::Vec3b>(i, j)[0];
			right_rgbs_host[i * 3 * width + 3 * j + 1] = img_right.at<cv::Vec3b>(i, j)[1];
			right_rgbs_host[i * 3 * width + 3 * j + 2] = img_right.at<cv::Vec3b>(i, j)[2];
		}
	}
	//开始计算
	cudaMemcpy(left_rgbs_device, left_rgbs_host, width*height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(right_rgbs_device, right_rgbs_host, width*height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);


	int blockSize = 1;
	int imageSize = width*height;
	dim3 fullBlocksPerGrid((width*height + blockSize - 1) / blockSize);//需要开辟的线程模型

	//彩色图转灰度图
	rgb2gray << <fullBlocksPerGrid, blockSize >> > (left_rgbs_device, left_gray_device, imageSize);
	rgb2gray << <fullBlocksPerGrid, blockSize >> > (right_rgbs_device, right_gray_device, imageSize);
	rgb2gray_cpu(left_rgbs_host, garyImageL_cpuResult.data, imageSize);
	rgb2gray_cpu(right_rgbs_host, garyImageR_cpuResult.data, imageSize);
	//保存结果
	cv::Mat garyImageL_gpuResult = cv::Mat::zeros(height, width, CV_8UC1);
	cv::Mat garyImageR_gpuResult = cv::Mat::zeros(height, width, CV_8UC1);
	cudaMemcpy(garyImageL_gpuResult.data, left_gray_device, imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(garyImageR_gpuResult.data, right_gray_device, imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cv::imwrite("grayL_gpu.png", garyImageL_gpuResult);
	cv::imwrite("grayR_gpu.png", garyImageR_gpuResult);
	cv::imwrite("grayL_cpu.png", garyImageL_cpuResult);
	cv::imwrite("grayR_cpu.png", garyImageR_cpuResult);
	//对比结果
	//bool garyisSameFlag = compareDatas_gpucpu(left_gray_device, garyImageL_cpuResult.data, imageSize);
	//std::cout << "对比转灰度图结果L：" << garyisSameFlag << std::endl;
	//garyisSameFlag = compareDatas_gpucpu(right_gray_device, garyImageR_cpuResult.data, imageSize);
	//std::cout << "对比转灰度图结果R：" << garyisSameFlag << std::endl;

	//计算census特征
	cudaMemset(left_census_device, 0, imageSize * sizeof(uint64_t));
	cudaMemset(right_census_device, 0, imageSize * sizeof(uint64_t));
	memset(censusLeft_host, 0, imageSize * sizeof(uint64_t));
	memset(censusRight_host, 0, imageSize * sizeof(uint64_t));
	census9X7 << <fullBlocksPerGrid, blockSize >> > (left_gray_device, left_census_device, width, height, imageSize);
	census9X7 << <fullBlocksPerGrid, blockSize >> > (right_gray_device, right_census_device, width, height, imageSize);

	census9X7_cpu(garyImageL_gpuResult.data, censusLeft_host, width, height, imageSize);
	census9X7_cpu(garyImageR_gpuResult.data, censusRight_host, width, height, imageSize);

	bool censusisSameFlag = compareDatas_gpucpu(left_census_device, censusLeft_host, imageSize);
	std::cout << "对比census计算结果L：" << censusisSameFlag << std::endl;
	censusisSameFlag = compareDatas_gpucpu(right_census_device, censusRight_host, imageSize);
	std::cout << "对比census计算结果R：" << censusisSameFlag << std::endl;
	cv::Mat differnet1;
	compareImage_gpucpu(left_census_device, censusLeft_host, differnet1, width, height);
	cv::Mat differnet2;
	compareImage_gpucpu(right_census_device, censusRight_host, differnet2, width, height);
	//显示
	cv::Mat censusLeftImage = cv::Mat::zeros(height, width, CV_64F);
	cv::Mat censusRightImage = cv::Mat::zeros(height, width, CV_64F);
	cudaMemcpy(censusLeftImage.data, left_census_device, imageSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(censusRightImage.data, right_census_device, imageSize * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	memcpy(censusLeftImage.data, censusLeft_host, imageSize * sizeof(uint64_t));
	memcpy(censusRightImage.data, censusRight_host, imageSize * sizeof(uint64_t));

	//计算初始代价
	computeInitCost << <fullBlocksPerGrid, blockSize >> > (
		left_rgbs_device, right_rgbs_device,
		left_census_device, right_census_device,
		cost_init_device,
		width, imageSize);
	computeInitCost_cpu(
		left_rgbs_host, right_rgbs_host,
		censusLeft_host, censusRight_host,
		cost_init_host,
		width, imageSize
	);
	bool initCostIsSameFlag = compareDatas_gpucpu(cost_init_device, cost_init_host, Disp_Range*imageSize);
	std::cout << "初始代价是否一样:" << initCostIsSameFlag << std::endl;

	//计算交叉臂
	buildCorssArms << <fullBlocksPerGrid, blockSize >> > (left_rgbs_device, arms_device, height, width, imageSize);
	buildCorssArms_cpu(left_rgbs_host, arms_host, height, width, imageSize);
	//
	bool isSame = true;
	CrossArm* actual_host = new CrossArm[imageSize];
	cudaMemcpy(actual_host, arms_device, imageSize * sizeof(CrossArm), cudaMemcpyDeviceToHost);
	bool armsIsSameFlag = true;
	int count = 0;
	for (size_t i = 0; i < imageSize; i++)
	{
		if (fabs(actual_host[i].left - arms_host[i].left) > 0.001 ||
			fabs(actual_host[i].right - arms_host[i].right) > 0.001 ||
			fabs(actual_host[i].top - arms_host[i].top) > 0.001 ||
			fabs(actual_host[i].bottom - arms_host[i].bottom) > 0.001 ||
			fabs(actual_host[i].Height - arms_host[i].Height) > 0.001 ||
			fabs(actual_host[i].Width - arms_host[i].Width) > 0.001
			) {
			count++;
			std::cerr << "(" << i << "): ";
			std::cerr << static_cast<int>(actual_host[i].left) << " ";
			std::cerr << static_cast<int>(actual_host[i].right) << " ";
			std::cerr << static_cast<int>(actual_host[i].top) << " ";
			std::cerr << static_cast<int>(actual_host[i].bottom) << " ";
			std::cerr << static_cast<int>(actual_host[i].Height) << " ";
			std::cerr << static_cast<int>(actual_host[i].Width) << " ";
			std::cerr << " / ";		 
			std::cerr << static_cast<int>(arms_host[i].left) << " ";
			std::cerr << static_cast<int>(arms_host[i].right) << " ";
			std::cerr << static_cast<int>(arms_host[i].top) << " ";
			std::cerr << static_cast<int>(arms_host[i].bottom) << " ";
			std::cerr << static_cast<int>(arms_host[i].Height) << " ";
			std::cerr << static_cast<int>(arms_host[i].Width) << " ";
			std::cerr << std::endl;
			initCostIsSameFlag = false;
		}
	}
	std::cout << "不同元素计数个数为：" << count << std::endl;
	delete[] actual_host;
	std::cout << "交叉臂是否一样:" << initCostIsSameFlag << std::endl;
	//计算每个像素横竖的支持像素数量个数
	bool verticaldirection = true;
	getSupportPixelCount << <fullBlocksPerGrid, blockSize >> >(
		arms_device,
		supportPixelcountVf_device,
		verticaldirection,
		height, width, imageSize);
	verticaldirection = false;
	getSupportPixelCount << <fullBlocksPerGrid, blockSize >> >(
		arms_device,
		supportPixelcountHf_device,
		verticaldirection,
		height, width, imageSize);
	verticaldirection = true;
	getSupportPixelCount_cpu(
		arms_host,
		supportPixelcountVf_host,
		verticaldirection,
		height, width, imageSize);
	verticaldirection = false;
	getSupportPixelCount_cpu(
		arms_host,
		supportPixelcountHf_host,
		verticaldirection,
		height, width, imageSize);
	bool isVfSameFlag = compareDatas_gpucpu(supportPixelcountVf_device, supportPixelcountVf_host, imageSize);
	std::cout << "VF 计数是一样的:" << isVfSameFlag << std::endl;
	bool isHfSameFlag = compareDatas_gpucpu(supportPixelcountHf_device, supportPixelcountHf_host, imageSize);
	std::cout << "HF 计数是一样的:" << isHfSameFlag << std::endl;

	//复制初始代价m_cost_init_device给用于聚合的变量m_cost_aggr_device
	cudaMemcpy(cost_aggr_device, cost_init_device, width*height * Disp_Range * sizeof(float), cudaMemcpyDeviceToDevice);
	// 先将聚合代价初始化为初始代价
	memcpy(cost_aggr_host, cost_init_host, width*height*Disp_Range * sizeof(float));

	////比较复制之后
	//bool copyIsSameFlag = compareDatas_gpucpu(cost_aggr_device, cost_aggr_host, imageSize*Disp_Range);
	//std::cout << "比较复制之后代价值是否相同：" << copyIsSameFlag << std::endl;

	////比较取视差为d的代价函数
	//int current_d = 0;
	//getCostOfOneDisparity << <fullBlocksPerGrid, blockSize >> >
	//	(cost_temp1_device, cost_aggr_device, current_d, imageSize);
	//for (int y = 0; y < height; y++) {
	//	for (int x = 0; x < width; x++) {
	//		vec_cost_tmp[0][y * width + x]
	//			= cost_aggr_host[y * width * Disp_Range + x * Disp_Range + current_d];
	//	}
	//}
	////比较比较两端取同一位置同一视差的代价值是否相同
	//bool getOneDisparityIsSameFlag = compareDatas_gpucpu(cost_temp1_device, vec_cost_tmp[0], imageSize);
	//std::cout << "比较两端取同一位置同一视差的代价值是否相同：" << getOneDisparityIsSameFlag << std::endl;
	////比较聚合第一个方向得到的聚合带价值cost_temp2_device，vec_cost_tmp[1]
	//AggregateInVerticalDirection << <fullBlocksPerGrid, blockSize >> >
	//	(arms_device , cost_temp1_device, cost_temp2_device, height, width, imageSize);
	//for (int y = 0; y < height; y++) {
	//	for (int x = 0; x < width; x++) {
	//		// 获取arm数值
	//		auto& arm = arms_host[y*width + x];
	//		// 聚合
	//		float cost = 0.0f;
	//		for (int t = -arm.top; t <= arm.bottom; t++) {
	//			cost += vec_cost_tmp[0][(y + t) * width + x];
	//		}
	//		vec_cost_tmp[1][y*width + x] = cost;
	//	}
	//}
	//bool vericalAggrIsSameFlag = compareDatas_gpucpu(cost_temp2_device, vec_cost_tmp[1], imageSize);
	//std::cout << "比较聚合第一个方向得到的聚合带价值cost_temp2_device，vec_cost_tmp[1]：" << vericalAggrIsSameFlag << std::endl;
	//AggregateInHorizontalDirection << <fullBlocksPerGrid, blockSize >> >
	//	(arms_device, cost_temp2_device, cost_temp1_device, imageSize);
	//for (int y = 0; y < height; y++) {
	//	for (int x = 0; x < width; x++) {
	//		// 获取arm数值
	//		auto& arm = arms_host[y*width + x];
	//		// 聚合
	//		float cost = 0.0f;
	//		for (int t = -arm.left; t <= arm.right; t++) {
	//			cost += vec_cost_tmp[1][y*width + x + t];
	//		}
	//		vec_cost_tmp[0][y*width + x] = cost;
	//	}
	//}
	//bool horizontalAggrIsSameFlag = compareDatas_gpucpu(cost_temp1_device, vec_cost_tmp[0], imageSize);
	//std::cout << "比较聚合第一个方向得到的聚合带价值cost_temp1_device，vec_cost_tmp[0]：" << horizontalAggrIsSameFlag << std::endl;
	//
	////计算视差的聚合代价
	//getAggregatedCost << <fullBlocksPerGrid, blockSize >> >
	//	(cost_temp1_device, cost_aggr_device, supportPixelcountVf_device, 0, imageSize);
	//for (int y = 0; y < height; y++) {
	//	for (int x = 0; x < width; x++) {
	//		cost_aggr_host[y*width*Disp_Range + x*Disp_Range + 0] = vec_cost_tmp[0][y*width + x] / vec_sup_count[1][y*width + x];
	//	}
	//}
	//
	//bool AggrCostsIsSameFlag = compareDatas_gpucpu(cost_aggr_device, cost_aggr_host, imageSize*Disp_Range);
	//std::cout << "比较cost_aggr_device, cost_aggr_host：" << AggrCostsIsSameFlag << std::endl;


	//多迭代聚合
	bool horizontal_first = false;

	for (int k = 0; k < 4; k++) {
		//通过流来并行？？？，不同视差的代价指分别聚合
		for (int d = Min_Disparity; d < Disp_Range; d++) {
			aggregateInArms(fullBlocksPerGrid, blockSize, arms_device,
				cost_temp1_device, cost_temp2_device, cost_aggr_device,
				supportPixelcountHf_device, supportPixelcountVf_device,
				d, horizontal_first, height, width, imageSize);
		}
		// 下一次迭代，调换顺序
		horizontal_first = !horizontal_first;
	}


	//horizontal_first = false;
	//// 多迭代聚合
	//for (int k = 0; k < 4; k++) {
	//	for (int d = Min_Disparity; d < Disp_Range; d++) {
	//		aggregateInArms_cpu(
	//			d,
	//			cost_aggr_host,
	//			horizontal_first, width, height,
	//			arms_host,
	//			vec_cost_tmp,
	//			vec_sup_count);
	//	}
	//	// 下一次迭代，调换顺序
	//	horizontal_first = !horizontal_first;
	//}


	float* tempCost1 = new float[width*height * Disp_Range];
	cudaMemcpy(tempCost1, cost_aggr_device, width*height * Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);
	cv::Mat costAggredevice = cv::Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			costAggredevice.at<float>(i, j) = tempCost1[256 * (i*width + j) + 30];
		}
	}


	//复制出来进行下一步的验证
	cudaMemcpy(cost_aggr_host, cost_aggr_device, width*height * Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);

	bool aggrCostIsSameFlag = compareDatas_gpucpu(cost_aggr_device, cost_aggr_host, Disp_Range*width*height);
	std::cout << "代价聚合结果一致：" << aggrCostIsSameFlag << std::endl;

	//扫描线优化
	float so_p1 = So_p1;
	float so_p2 = So_p2;
	float so_tso = So_tso;
	dim3 fullBlocksPerGrid2((height + blockSize - 1) / blockSize);
	//gpu1
	scanlineOptimizeLeftRight << <fullBlocksPerGrid2, blockSize >> >
		(left_rgbs_device, right_rgbs_device,
			cost_aggr_device, cost_init_device, 
			true, so_p1, so_p2, so_tso, height, width);

	float* tempCost3 = new float[width*height * Disp_Range];
	cudaMemcpy(tempCost3, cost_init_device, width*height * Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);
	cv::Mat LOLR1cost30device = cv::Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			LOLR1cost30device.at<float>(i, j) = tempCost3[256 * (i*width + j) + 30];
		}
	}
	//cpu1
	scanlineOptimizeLeftRight_cpu(
		left_rgbs_host, right_rgbs_host, 
		cost_aggr_host, cost_init_host, 
		true, so_p1, so_p2, so_tso, height, width);
	cv::Mat LOLR1cost30host = cv::Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			LOLR1cost30host.at<float>(i, j) = cost_init_host[256 * (i*width + j) + 30];
		}
	}

	//比较扫描线优化后的代价值
	bool costaggrAfterLOLR_Flag = compareDatas_gpucpu(cost_init_device, cost_init_host, width*height * Disp_Range);
	std::cout << "第一次左右扫描线优化后的数值比较：" << costaggrAfterLOLR_Flag << std::endl;

	//gpu2
	scanlineOptimizeLeftRight << <fullBlocksPerGrid2, blockSize >> >
		(left_rgbs_device, right_rgbs_device,
			cost_init_device, cost_aggr_device, false,
			so_p1, so_p2, so_tso, height, width);

	cv::Mat LOLR2cost30device = cv::Mat::zeros(height, width, CV_32FC1);
	cudaMemcpy(tempCost3, cost_aggr_device, width*height * Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			LOLR2cost30device.at<float>(i, j) = tempCost3[256 * (i*width + j) + 30];
		}
	}

	//cpu2
	scanlineOptimizeLeftRight_cpu(
		left_rgbs_host, right_rgbs_host, 
		cost_init_host, cost_aggr_host, 
		false, so_p1, so_p2, so_tso, height, width);
	cv::Mat LOLR2cost30host = cv::Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			LOLR2cost30host.at<float>(i, j) = cost_aggr_host[256 * (i*width + j) + 30];
		}
	}
	costaggrAfterLOLR_Flag = compareDatas_gpucpu(cost_aggr_device, cost_aggr_host, width*height * Disp_Range);
	std::cout << "第二次左右扫描线优化后的数值比较：" << costaggrAfterLOLR_Flag << std::endl;

	//上下
	dim3 fullBlocksPerGrid3((width + blockSize - 1) / blockSize);
	//gpu3
	scanlineOptimizeUpDown << <fullBlocksPerGrid3, blockSize >> >
		(left_rgbs_device, right_rgbs_device,
			cost_aggr_device, cost_init_device, 
			true, so_p1, so_p2, so_tso, height, width);

	cv::Mat LOUD1cost30device = cv::Mat::zeros(height, width, CV_32FC1);
	cudaMemcpy(tempCost3, cost_init_device, width*height * Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			LOUD1cost30device.at<float>(i, j) = tempCost3[256 * (i*width + j) + 30];
		}
	}

	//cpu3
	scanlineOptimizeUpDown_cpu(
		left_rgbs_host, right_rgbs_host, 
		cost_aggr_host, cost_init_host, 
		true, so_p1, so_p2, so_tso, height, width);
	cv::Mat LOUD1cost30host = cv::Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			LOUD1cost30host.at<float>(i, j) = cost_init_host[256 * (i*width + j) + 30];
		}
	}
	costaggrAfterLOLR_Flag = compareDatas_gpucpu(cost_init_device, cost_init_host, width*height * Disp_Range);
	std::cout << "第一次上下扫描线优化后的数值比较：" << costaggrAfterLOLR_Flag << std::endl;

	//gpu4
	scanlineOptimizeUpDown << <fullBlocksPerGrid3, blockSize >> >
		(left_rgbs_device, right_rgbs_device,
			cost_init_device, cost_aggr_device, 
			false,so_p1, so_p2, so_tso, height, width);

	cv::Mat LOUD2cost30device = cv::Mat::zeros(height, width, CV_32FC1);
	cudaMemcpy(tempCost3, cost_aggr_device, width*height * Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			LOUD2cost30device.at<float>(i, j) = tempCost3[256 * (i*width + j) + 30];
		}
	}
	//cpu4
	scanlineOptimizeUpDown_cpu(
		left_rgbs_host, right_rgbs_host, 
		cost_init_host, cost_aggr_host, 
		false, so_p1, so_p2, so_tso, height, width);
	cv::Mat LOUD2cost30host = cv::Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			LOUD2cost30host.at<float>(i, j) = cost_aggr_host[256 * (i*width + j) + 30];
		}
	}
	costaggrAfterLOLR_Flag = compareDatas_gpucpu(cost_aggr_device, cost_aggr_host, width*height * Disp_Range);
	std::cout << "第一次上下扫描线优化后的数值比较：" << costaggrAfterLOLR_Flag << std::endl;
	

	//从代价中得到视差值
	//计算主右视差
	//computeDisparity << <fullBlocksPerGrid, blockSize >> >
	//	(m_cost_aggr_device, m_disparity_left_device, m_height, m_width);
	//computeDisparityRight << <fullBlocksPerGrid, blockSize >> >
	//	(m_cost_aggr_device, m_disparity_right_device, m_height, m_width);

	//getdisparity(disparity, m_disparity_left_device);

	if (tempCost1 != nullptr)
	{
		cudaFree(tempCost1);
		tempCost1 = nullptr;
	}


	//释放内存
	if (left_rgbs_host != nullptr)
	{
		delete[] left_rgbs_host;
		left_rgbs_host = nullptr;
	}
		
	if (right_rgbs_host != nullptr)
	{
		delete[] right_rgbs_host;
		right_rgbs_host = nullptr;
	}

	if (disparity != nullptr)
	{
		delete[] disparity;
		disparity = nullptr;
	}
		
	if (censusLeft_host!=nullptr)
	{
		delete[] censusLeft_host;
		censusLeft_host = nullptr;
	}
	if (censusRight_host != nullptr)
	{
		delete[] censusRight_host;
		censusRight_host = nullptr;
	}
	if (cost_init_host != nullptr) 
	{
		delete[] cost_init_host;
		cost_init_host = nullptr;
	}
	if (cost_aggr_host != nullptr)
	{
		delete[] cost_aggr_host;
		cost_aggr_host = nullptr;
	}
	if (arms_host != nullptr)
	{
		delete[] arms_host;
		arms_host = nullptr;
	}

	if (left_rgbs_device != nullptr)
	{
		cudaFree(left_rgbs_device);
		left_rgbs_device = nullptr;
	}
	if (right_rgbs_device != nullptr)
	{
		cudaFree(right_rgbs_device);
		right_rgbs_device = nullptr;
	}
	if (left_gray_device != nullptr)
	{
		cudaFree(left_gray_device);
		left_gray_device = nullptr;
	}
	if (right_gray_device != nullptr)
	{
		cudaFree(right_gray_device);
		right_gray_device = nullptr;
	}
	if (left_census_device != nullptr)
	{
		cudaFree(left_census_device);
		left_census_device = nullptr;
	}
	if (right_census_device != nullptr)
	{
		cudaFree(right_census_device);
		right_census_device = nullptr;
	}
	if (cost_init_device != nullptr)
	{
		cudaFree(cost_init_device);
		cost_init_device = nullptr;
	}
	if (cost_aggr_device != nullptr)
	{
		cudaFree(cost_aggr_device);
		cost_aggr_device = nullptr;
	}
	if (arms_device != nullptr)
	{
		cudaFree(arms_device);
		arms_device = nullptr;
	}
}

void testCuAdcensus()
{
	std::string path_left = "left.png";
	std::string path_right = "right.png";
	std::cout << "Image Loading..." << std::endl;
	cv::Mat img_left = cv::imread(path_left, cv::IMREAD_COLOR);
	cv::Mat img_right = cv::imread(path_right, cv::IMREAD_COLOR);
	if (img_left.data == nullptr || img_right.data == nullptr) {
		std::cout << "读取影像失败！" << std::endl;
		return;
	}
	if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
		std::cout << "左右影像尺寸不一致！" << std::endl;
		return;
	}
	//···············································································//
	const int width = static_cast<int>(img_left.cols);
	const int height = static_cast<int>(img_right.rows);

	// 左右影像的彩色数据
	uint8_t* bytes_left = new uint8_t[width * height * 3];
	uint8_t* bytes_right = new uint8_t[width * height * 3];
	float* disparity = new float[width * height];

#pragma omp for
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bytes_left[i * 3 * width + 3 * j] = img_left.at<cv::Vec3b>(i, j)[0];
			bytes_left[i * 3 * width + 3 * j + 1] = img_left.at<cv::Vec3b>(i, j)[1];
			bytes_left[i * 3 * width + 3 * j + 2] = img_left.at<cv::Vec3b>(i, j)[2];
			bytes_right[i * 3 * width + 3 * j] = img_right.at<cv::Vec3b>(i, j)[0];
			bytes_right[i * 3 * width + 3 * j + 1] = img_right.at<cv::Vec3b>(i, j)[1];
			bytes_right[i * 3 * width + 3 * j + 2] = img_right.at<cv::Vec3b>(i, j)[2];
		}
	}

	cuAdcensus cuAdcensusObj(width, height);
	cuAdcensusObj.setIterationTimes(4);
	cuAdcensusObj.compute(bytes_left, bytes_right, disparity);

	//释放内存
	if (bytes_left != nullptr)
	{
		delete[] bytes_left;
		bytes_left = nullptr;
	}
		
	if (bytes_right != nullptr)
	{
		delete[] bytes_right;
		bytes_right = nullptr;
	}

	if (disparity != nullptr)
	{
		delete[] disparity;
		disparity = nullptr;
	}
}

