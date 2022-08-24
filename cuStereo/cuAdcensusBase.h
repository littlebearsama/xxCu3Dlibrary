#pragma once
#include <cstdint> //uint8_t
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
//用于存储vector
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include"adcensusBase.h"

__global__ void rgb2gray(const uint8_t* rgbs, uint8_t* gray, int imagesize);
__global__ void census9X7(const uint8_t *garyImage, uint64_t *censusImage, int width, int height, int imagesize);
__global__ void computeInitCost(const uint8_t* left_rgbs, const uint8_t* right_rgbs,
	const uint64_t *left_censusImage, const uint64_t *right_censusImage,
	float* cost_init,
	int width, int imagesize);

//每个像素构建十字交叉臂
__global__ void buildCorssArms(const uint8_t* left_rgbs, CrossArm* arms, int height, int width, int imagesize);
//计算像素的支持区像素数量(横竖计算出来的大小是不同的)
__global__ void getSupportPixelCount(CrossArm* arms,
	int* vec_supportPixel_count,
	bool verticaldirection,
	int height, int width, int imagesize);
//代价聚合

void aggregateInArms(
	dim3 gridsize,
	int blockSize,
	CrossArm* cross_arms,
	float* costs_temp1,//第一个数组是代价数组 第二个数组是临时数组
	float* costs_temp2,//第一个数组是代价数组 第二个数组是临时数组
	float* costs_aggr,
	int* sup_countHf,
	int* sup_countVf,
	int d,
	bool horizontalDirectFirst,
	int height, int width, int imagesize);

__global__ void scanlineOptimizeLeftRight(uint8_t *img_left_, uint8_t *img_right_,
	float* cost_so_src, float* cost_so_dst, bool is_forward,
	float p1, float p2, float tso,
	int height, int width);

__global__ void scanlineOptimizeUpDown(uint8_t *img_left_, uint8_t *img_right_,
	float* cost_so_src, float* cost_so_dst, bool is_forward,
	float p1, float p2, float tso,
	int height, int width);

//从代价中得到视差
__global__ void computeDisparity(float* costVec, float* disparity, int height, int width);
__global__ void computeDisparityRight(float* costVec, float* disparity, int height, int width);

//找到误匹配mismatchFlags以及遮挡区域occlusionFlags
__global__ void outlierDetection(float* disparityLeft, float* disparityRight, int* mismatchFlags, int* occlusionFlags, float lrCheckThreshold, int height, int width);
//基于交叉十字臂区域的投票
__global__ void regionVotingOneLoop(float* dispLeft, CrossArm* cross_arms, int* errorPts, int irvCountThresh, int irvDispThresh, int height, int width);
__global__ void getCostOfOneDisparity(
	float* costOfOneDisparity,
	float* cost,
	int d,
	int imagesize);
//在竖直方向聚合一次
__global__ void AggregateInVerticalDirection
(
	CrossArm* cross_arms,
	float* costs_input,
	float* costs_output,
	int height, int width, int imagesize
);
__global__ void AggregateInHorizontalDirection
(
	CrossArm* cross_arms,
	float* costs_input,
	float* costs_output,
	int imagesize
);
__global__ void AggregateInVerticalDirection
(
	CrossArm* cross_arms,
	float* costs_input,
	float* costs_output,
	int height, int width, int imagesize
);
__global__ void getAggregatedCost
(
	float* costs,
	float* costs_aggr,
	int* sup_counts,
	int d,
	int imagesize
);
