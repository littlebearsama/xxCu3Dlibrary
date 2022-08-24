#include "cuAdcensus.h"
//cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>//解决不能识别threadId等内部变量的问题

#define DEBUGPrintCost
#ifdef DEBUGPrintCost
#include <opencv2/opencv.hpp>
#include <iostream>
#endif // DEBUGPrint
//#define DEBUGPrintGrayImage
#define DEBUGPrintInitCost
//#define DEBUGPrintArms
//#define DEBUGPrintSupportArea
#define DEBUGPrintCopyInitCost
#define DEBUGPrintCensus


cuAdcensus::cuAdcensus(int width, int height)
{
	m_IterationTimes = 4;
	m_memoryAllocFlag = false;
	//host
	m_left_rgbs_host = nullptr;
	m_right_rgbs_host = nullptr;
	m_disparity_host = nullptr;
	//device
	m_left_rgbs_device = nullptr;
	m_right_rgbs_device = nullptr;
	m_left_grays_device = nullptr;
	m_right_grays_device = nullptr;
	m_left_census_device = nullptr;
	m_right_census_device = nullptr;

	m_arms_device = nullptr;
	m_supportPixelcountVf_device = nullptr;
	m_supportPixelCountHf_device = nullptr;
	m_cost_init_device = nullptr;
	//for (int i = 0; i < Disp_Range; i++)
	//{
	//	m_cost_per_disp_device[i] = nullptr;
	//}
	m_cost_temp1_device = nullptr;
	m_cost_temp2_device = nullptr;
	m_cost_aggr_device = nullptr;

	m_disparity_left_device = nullptr;
	m_disparity_right_device = nullptr;
	m_disparity_device = nullptr;
	memoryAllocForObject(width, height);
}

cuAdcensus::~cuAdcensus()
{
	m_memoryAllocFlag = false;
	destoryAllmemories();
}


bool cuAdcensus::memoryAllocForObject(int width, int height)
{
	m_width = width;
	m_height = height;
	//host
	m_left_rgbs_host = new uint8_t[width*height * 3];
	m_right_rgbs_host = new uint8_t[width*height * 3];
	m_disparity_host = new float[width*height];
	//device
	cudaMalloc((void**)&m_left_rgbs_device, width*height * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&m_right_rgbs_device, width*height * 3 * sizeof(uint8_t));
	cudaMalloc((void**)&m_left_grays_device, width*height * sizeof(uint8_t));
	cudaMalloc((void**)&m_right_grays_device, width*height * sizeof(uint8_t));
	cudaMalloc((void**)&m_left_census_device, width*height * sizeof(uint64_t));
	cudaMalloc((void**)&m_right_census_device, width*height * sizeof(uint64_t));

	cudaMalloc((void**)&m_arms_device, width*height * sizeof(CrossArm));
	cudaMalloc((void**)&m_supportPixelcountVf_device, width*height * sizeof(int));
	cudaMalloc((void**)&m_supportPixelCountHf_device, width*height * sizeof(int));

	cudaMalloc((void**)&m_cost_init_device, Disp_Range*width*height * sizeof(float));
	//for (int i = 0; i < Disp_Range; i++)
	//{
	//	cudaMalloc((void**)&m_cost_per_disp_device[i], width*height * sizeof(float));
	//}

	cudaMalloc((void**)&m_cost_temp1_device, width*height * sizeof(float));
	cudaMalloc((void**)&m_cost_temp2_device, width*height * sizeof(float));
	cudaMalloc((void**)&m_cost_aggr_device, Disp_Range*width*height * sizeof(float));

	cudaMalloc((void**)&m_disparity_left_device, width*height * sizeof(float));
	cudaMalloc((void**)&m_disparity_right_device, width*height * sizeof(float));
	cudaMalloc((void**)&m_disparity_device, width*height * sizeof(float));

	m_memoryAllocFlag = true;
	return true;
}

void cuAdcensus::setIterationTimes(int times)
{
	m_IterationTimes = times;
}

void cuAdcensus::setScanlineOptimizeParameter(float so_p1, float so_p2, float so_tso)
{
	m_so_p1 = so_p1;
	m_so_p2 = so_p2;
	m_so_tso = so_tso;
}

void cuAdcensus::compute(uint8_t* left_rgbs_host_ptr, uint8_t* right_rgbs_host_ptr, float* disparity)
{
	cudaMemcpy(m_left_rgbs_device, left_rgbs_host_ptr, m_width*m_height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(m_right_rgbs_device, right_rgbs_host_ptr, m_width*m_height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
#ifdef DEBUGPrintRGBImage
	cv::Mat rgbImageL = cv::Mat::zeros(m_height, m_width, CV_8UC3);
	cudaMemcpy(rgbImageL.data, m_left_rgbs_device, 3 * m_width*m_height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cv::Mat rgbImageR = cv::Mat::zeros(m_height, m_width, CV_8UC3);
	cudaMemcpy(rgbImageR.data, m_right_rgbs_device, 3 * m_width*m_height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
#endif // DEBUGPrintRGBImage

	int blockSize = 128;
	int imageSize = m_width*m_height;
	dim3 fullBlocksPerGrid((m_width*m_height + blockSize - 1) / blockSize);//需要开辟的线程模型
	//彩色图转灰度图
	rgb2gray<<<fullBlocksPerGrid, blockSize >>>(m_left_rgbs_device, m_left_grays_device, imageSize);
	rgb2gray<<<fullBlocksPerGrid, blockSize >>>(m_right_rgbs_device, m_right_grays_device, imageSize);
#ifdef DEBUGPrintGrayImage
	cv::Mat garyImageL = cv::Mat::zeros(m_height, m_width, CV_8UC1);
	cudaMemcpy(garyImageL.data, m_left_grays_device, m_width*m_height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cv::Mat garyImageR = cv::Mat::zeros(m_height, m_width, CV_8UC1);
	cudaMemcpy(garyImageR.data, m_right_grays_device, m_width*m_height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cv::imwrite("garyImageL.png", garyImageL);
	cv::imwrite("garyImageR.png", garyImageR);
#endif // DEBUGPrintGrayImage

	//计算census特征
	census9X7 <<<fullBlocksPerGrid, blockSize >>> (m_left_grays_device, m_left_census_device, m_width, m_height, imageSize);
	census9X7 <<<fullBlocksPerGrid, blockSize >>> (m_right_grays_device, m_right_census_device, m_width, m_height, imageSize);
#ifdef DEBUGPrintCensus
	cv::Mat censusL64 = cv::Mat::zeros(m_height, m_width, CV_64FC1);
	cv::Mat censusR64 = cv::Mat::zeros(m_height, m_width, CV_64FC1);
	cudaMemcpy(censusL64.data, m_left_census_device, m_width*m_height * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(censusR64.data, m_right_census_device, m_width*m_height * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cv::imwrite("censusL64.bmp", censusL64);
	cv::imwrite("censusR64.bmp", censusR64);
#endif // DEBUGPrintCensus

	//计算初始代价
 	computeInitCost << <fullBlocksPerGrid, blockSize >> >(
		m_left_rgbs_device, m_right_rgbs_device,
		m_left_census_device, m_right_census_device,
		m_cost_init_device,
		m_width, imageSize);
#ifdef DEBUGPrintInitCost
	//打印d=52初始代价
	float* cost_host = new float[m_width*m_height*Disp_Range];
	cudaMemcpy(cost_host, m_cost_init_device, m_width*m_height *Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);
	cv::Mat init_cost_d52 = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	for (int i = 0; i < m_height; i++)
	{
		for (int j = 0; j < m_width; j++)
		{
			init_cost_d52.at<float>(i, j) = cost_host[Disp_Range*(i*m_width + j) + 52];
		}
	}
	cv::imwrite("init_cost_d52.tiff", init_cost_d52);
	delete[] cost_host;
#endif//DEBUGPrintInitCost

	//计算交叉臂
	buildCorssArms << <fullBlocksPerGrid, blockSize >> >(m_left_rgbs_device, m_arms_device, m_height, m_width, imageSize);
#ifdef DEBUGPrintArms
	CrossArm* armsTemp = new CrossArm[m_width*m_height];
	cudaMemcpy(armsTemp, m_arms_device, m_width*m_height * sizeof(CrossArm), cudaMemcpyDeviceToHost);
	for (int i = 0; i < m_height; i=i+80)
	{
		for (int j = 0; j < m_width; j=j+80)
		{
			int index = i*m_width+j;
			std::cout << (int)armsTemp[index].left << " " << (int)armsTemp[index].right << " " << (int)armsTemp[index].top << " " << (int)armsTemp[index].bottom << ";";
		}
		std::cout << std::endl;
	}
	delete[] armsTemp;
#endif // DEBUGPrintArms

	//计算每个像素横竖的支持像素数量个数
	bool verticaldirection = true;
	getSupportPixelCount << <fullBlocksPerGrid, blockSize >> >(
		m_arms_device,
		m_supportPixelcountVf_device,
		verticaldirection,
		m_height, m_width, imageSize);
	verticaldirection = false;
	getSupportPixelCount << <fullBlocksPerGrid, blockSize >> >(
		m_arms_device,	
		m_supportPixelCountHf_device,
		verticaldirection,
		m_height, m_width, imageSize);
#ifdef DEBUGPrintSupportArea
	int* supportAreaCountVf = new int[m_width*m_height];
	int* supportAreaCountHf = new int[m_width*m_height];
	cudaMemcpy(supportAreaCountVf, m_supportPixelcountVf_device, m_width*m_height * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(supportAreaCountHf, m_supportPixelCountHf_device, m_width*m_height * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < m_height; i = i + 80)
	{
		for (int j = 0; j < m_width; j = j + 80)
		{
			int index = i*m_width + j;
			std::cout << (int)supportAreaCountVf[index] << " " << (int)supportAreaCountHf[index] << ";";
		}
		std::cout << std::endl;
	}
	delete[] supportAreaCountVf;
	delete[] supportAreaCountHf;
#endif //DEBUGPrintSupportArea

	//复制初始代价m_cost_init_device给用于聚合的变量m_cost_aggr_device
	cudaMemcpy(m_cost_aggr_device, m_cost_init_device, m_width*m_height * Disp_Range * sizeof(float), cudaMemcpyDeviceToDevice);

#ifdef DEBUGPrintCopyInitCost
	//打印d=52初始代价
	float* copy_host = new float[m_width*m_height*Disp_Range];
	cudaMemcpy(copy_host, m_cost_aggr_device, m_width*m_height *Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);
	cv::Mat init_copy_d = cv::Mat::zeros(m_height, m_width, CV_32FC1);
	for (int i = 0; i < m_height; i++)
	{
		for (int j = 0; j < m_width; j++)
		{
			init_copy_d.at<float>(i, j) = copy_host[Disp_Range*(i*m_width + j) + 52];
		}
	}
	cv::imwrite("init_cost_d52_cpy.tiff", init_copy_d);
	delete[] copy_host;
#endif // DEBUGPrintCopyInitCost

	std::cout << "blockSize:" << blockSize << std::endl;
	std::cout << "m_height:" << m_height << std::endl;
	std::cout << "m_width:" << m_width << std::endl;
	std::cout << "imageSize:" << imageSize << std::endl;
	std::cout << "imageSize:" << imageSize << std::endl;
	std::cout << "imageSize:" << imageSize << std::endl;
	//多迭代聚合
	bool horizontal_first = false;
	for (int k = 0; k < m_IterationTimes; k++) {
		//通过流来并行？？？，不同视差的代价指分别聚合
		for (int d = Min_Disparity; d < Max_Disparity; d++) {
			aggregateInArms(fullBlocksPerGrid, blockSize, m_arms_device, 
				m_cost_temp1_device, m_cost_temp2_device, m_cost_aggr_device, 
				m_supportPixelCountHf_device, m_supportPixelcountVf_device, 
				d, horizontal_first, m_height, m_width, imageSize);
//#ifdef DEBUGPrintCost
//			float* cost_host = new float[m_width*m_height*Disp_Range];
//			cudaMemcpy(cost_host, m_cost_aggr_device, m_width*m_height *Disp_Range * sizeof(float), cudaMemcpyDeviceToHost);
//			cv::Mat costAggre = cv::Mat::zeros(m_height, m_width, CV_32FC1);
//			for (int i = 0; i < m_height; i++)
//			{
//				for (int j = 0; j < m_width; j++)
//				{
//					costAggre.at<float>(i, j) = cost_host[Disp_Range*(i*m_width + j) + d];
//				}
//			}
//			delete[] cost_host;
//#endif
		}
		// 下一次迭代，调换顺序
		horizontal_first = !horizontal_first;
	}

	//扫描线优化
	dim3 fullBlocksPerGrid2((m_height + blockSize - 1) / blockSize);
	scanlineOptimizeLeftRight << <fullBlocksPerGrid2, blockSize >> > 
		(m_left_rgbs_device, m_right_rgbs_device, 
			m_cost_aggr_device, m_cost_init_device, true, 
			m_so_p1, m_so_p2, m_so_tso, m_height, m_width);
	scanlineOptimizeLeftRight << <fullBlocksPerGrid2, blockSize >> >
		(m_left_rgbs_device, m_right_rgbs_device,
			m_cost_init_device, m_cost_aggr_device, false,
			m_so_p1, m_so_p2, m_so_tso, m_height, m_width);

	dim3 fullBlocksPerGrid3((m_width + blockSize - 1) / blockSize);
	scanlineOptimizeUpDown << <fullBlocksPerGrid3, blockSize >> >
		(m_left_rgbs_device, m_right_rgbs_device,
			m_cost_aggr_device, m_cost_init_device, true,
			m_so_p1, m_so_p2, m_so_tso, m_height, m_width);
	scanlineOptimizeUpDown << <fullBlocksPerGrid3, blockSize >> >
		(m_left_rgbs_device, m_right_rgbs_device,
			m_cost_init_device, m_cost_aggr_device, false,
			m_so_p1, m_so_p2, m_so_tso, m_height, m_width);

	//从代价中得到视差值
	//计算主右视差
	computeDisparity << <fullBlocksPerGrid, blockSize >> >
		(m_cost_aggr_device, m_disparity_left_device, m_height, m_width);
	computeDisparityRight << <fullBlocksPerGrid, blockSize >> >
		(m_cost_aggr_device, m_disparity_right_device, m_height, m_width);

	getdisparity(disparity, m_disparity_left_device);
	//多步骤视差优化
	//OutlierDetection();


	//IterativeRegionVoting();

	//适当插值
	//ProperInterpolation();

	//DepthDiscontinuityAdjustment();

	// median filter

}

void cuAdcensus::destoryAllmemories()
{
	//host
	if (m_left_rgbs_host != nullptr) 
	{
		delete[] m_left_rgbs_host;
		m_left_rgbs_host = nullptr;
	}
	if (m_right_rgbs_host != nullptr)
	{
		delete[] m_right_rgbs_host;
		m_right_rgbs_host = nullptr;
	}	
	if (m_disparity_host != nullptr)
	{
		delete[] m_disparity_host;
		m_disparity_host = nullptr;
	}

	//device
	if (m_left_rgbs_device != nullptr) 
	{
		cudaFree(m_left_rgbs_device);
		m_left_rgbs_device = nullptr;
	}
	if (m_right_rgbs_device != nullptr)
	{
		cudaFree(m_right_rgbs_device);
		m_right_rgbs_device = nullptr;
	}
	if (m_left_grays_device != nullptr)
	{
		cudaFree(m_left_grays_device);
		m_left_grays_device = nullptr;
	}
	if (m_right_grays_device != nullptr)
	{
		cudaFree(m_right_grays_device);
		m_right_grays_device = nullptr;
	}
	if (m_left_census_device != nullptr)
	{
		cudaFree(m_left_census_device);
		m_left_census_device = nullptr;
	}
	if (m_right_census_device != nullptr)
	{
		cudaFree(m_right_census_device);
		m_right_census_device = nullptr;
	}
	if (m_arms_device != nullptr)
	{
		cudaFree(m_arms_device);
		m_arms_device = nullptr;
	}
	if (m_supportPixelCountHf_device != nullptr)
	{
		cudaFree(m_supportPixelCountHf_device);
		m_supportPixelCountHf_device = nullptr;
	}
	if (m_supportPixelcountVf_device != nullptr)
	{
		cudaFree(m_supportPixelcountVf_device);
		m_supportPixelcountVf_device = nullptr;
	}

	//for (int i = 0; i < Disp_Range; i++)
	//{
	//	if (m_cost_per_disp_device[i]!=nullptr)
	//	{
	//		cudaFree(m_cost_per_disp_device[i]);
	//		m_cost_per_disp_device[i] = nullptr;
	//	}
	//	
	//}
	if (m_cost_init_device != nullptr) 
	{
		cudaFree(m_cost_init_device);
		m_cost_init_device = nullptr;
	}
	if (m_cost_aggr_device != nullptr) 
	{
		cudaFree(m_cost_aggr_device);
		m_cost_aggr_device = nullptr;
	}
	if (m_disparity_left_device!=nullptr)
	{
		cudaFree(m_disparity_left_device);
		m_disparity_left_device = nullptr;

	}
	if (m_disparity_right_device != nullptr)
	{
		cudaFree(m_disparity_right_device);
		m_disparity_right_device = nullptr;

	}
	if (m_disparity_device != nullptr)
	{
		cudaFree(m_disparity_device);
		m_disparity_device = nullptr;
	}
	if (m_cost_temp1_device != nullptr) 
	{
		cudaFree(m_cost_temp1_device);
		m_cost_temp1_device = nullptr;
	}
	if (m_cost_temp2_device != nullptr)
	{
		cudaFree(m_cost_temp2_device);
		m_cost_temp2_device = nullptr;
	}
}

void cuAdcensus::getdisparity(float* disparity_host)
{
	cudaMemcpy(m_disparity_device, disparity_host, m_width*m_height * sizeof(float), cudaMemcpyDeviceToHost);
}

void cuAdcensus::getdisparity(float* disparity_host, float* disparity_device)
{
	cudaMemcpy(disparity_device, disparity_host, m_width*m_height * sizeof(float), cudaMemcpyDeviceToHost);
}
