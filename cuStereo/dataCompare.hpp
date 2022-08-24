#pragma once
#include <iostream>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <math.h>
template <typename T>
inline bool compareDatas_cpucpu(
	const T *actual,
	const T *expect,
	size_t datasize)
{
	bool accept = true;
	int count = 0;
	for (size_t i = 0; i < datasize; i++)
	{
		if (fabs(actual[i] -expect[i])>0.001) {
			count++;
			//std::cerr << "(" << i << "): ";
			//std::cerr << static_cast<double>(actual[i]) << " / ";
			//std::cerr << static_cast<double>(expect[i]) << std::endl;
			accept = false;
		}
	}
	std::cout << "不同元素计数个数为：" << count << std::endl;
	return accept;
}

template <typename T>
inline bool compareDatas_gpucpu(
	const T *actual_device,
	const T *expect_host,
	size_t datasize)
{
	bool accept = true;
	T* actual_host = new T[datasize];
	cudaMemcpy(actual_host, actual_device, datasize * sizeof(T), cudaMemcpyDeviceToHost);
	accept = compareDatas_cpucpu(actual_host, expect_host, datasize);
	delete[] actual_host;
	return accept;
}

template <typename T>
inline bool compareDatas_gpugpu(
	const T *actual_device,
	const T *expect_device,
	size_t datasize)
{
	bool accept = true;
	T* actual_host = new T[datasize];
	T* expect_host = new T[datasize];
	cudaMemcpy(actual_host, actual_device, datasize * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(expect_host, expect_device, datasize * sizeof(T), cudaMemcpyDeviceToHost);
	accept = compareDatas_cpucpu(actual_host, expect_host, datasize);
	delete[] actual_host;
	delete[] expect_host;
	return accept;
}

template <typename T>
inline void compareImage_gpucpu(
	const T *image_device,
	const T *image_host,
	cv::Mat& differnet,
	int width,
	int height)
{
	T* actual_host = new T[width* height];
	cudaMemcpy(actual_host, image_device, width* height * sizeof(T), cudaMemcpyDeviceToHost);
	
	differnet = cv::Mat::zeros(height, width, CV_8UC1);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int currnetId = y*width + x;
			differnet.at<uchar>(y, x) = (actual_host[currnetId] != image_host[currnetId]) ? 255 : 0;
		}
	}
	delete[] actual_host;
}