#include <stdio.h>
#include <assert.h>
#include <math_constants.h>

#include "math_constants.h" //CUDART_INF
#include <math.h>
#include <cmath>
#include <cstdint> //uint8_t

//相关声明的头文件
#include "cuAdcensusBase.h"

//#define DEBUGPrint
#ifdef DEBUGPrint
#include <opencv2/opencv.hpp>
#include <iostream>
#endif // DEBUGPrint

//author: xiaoxiong
//sicription:
//1.modify form cpu code of adcensu(https://github.com/ethan-li-coding/AD-Census)
//2.GPU code ,CUDA
//date: 2022.05.21

//#ifndef FLT_MAX
//#define FLT_MAX 3.402823466e+38F 
//#endif // !FLT_MAX

//xiaoxiong 2022.05.20
//彩色转灰度图
//imagesize: 数组大小
__global__ void rgb2gray(const uint8_t* rgbs, uint8_t* gray, int imagesize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < imagesize) {
		float b = rgbs[id * 3];
		float g = rgbs[id * 3 + 1];
		float r = rgbs[id * 3 + 2];
		gray[id] = uint8_t(r* 0.299 + g * 0.587 + b * 0.114);
	}
}

//xiaoxiong 2022.05.20
//census变换（计算每个像素的census特征）
__global__ void census9X7(const uint8_t *garyImage, uint64_t *censusImage,int width,int height, int imagesize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// 中心像素值
	if (id < imagesize) 
	{
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col
		//跳过边界
		if (x < 3 || x >= width - 3 || y < 4 || y >= height - 4) 
		{
			return;
		}
		uint64_t census_val = 0u;
		uint8_t gray_center = garyImage[id];
		//计算特征
		for (int r = -4; r <= 4; r++)
		{
			for (int c = -3; c <= 3; c++)
			{
				census_val <<= 1;//左移一位，后面补0
				uint8_t gray = garyImage[(y + r) * width + x + c];
				if (gray < gray_center) {
					census_val += 1;
				}
			}
		}
		// 中心像素的census值
		censusImage[id] = census_val;
	}

}

__global__ void census3X3(const uint8_t *garyImage, uint8_t *censusImage, int width, int height, int imagesize)
{
	const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	// 中心像素值
	if (id < imagesize)
	{
		const int y = id / width;//当前像素row
		const int x = id%width;//当前像素col
							   //跳过边界
		if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
		{
			return;
		}

	}
}

//计算两个census特征之间的hamming距离
//被计算初始代价调用
__inline__ __device__ uint8_t  Hamming64Distance(const uint64_t& x, const uint64_t& y)
{
	uint64_t dist = 0, val = x ^ y;

	// Count the number of set bits
	while (val) {
		++dist;
		val &= val - 1;//效果就是消除了二进制里面的最后一个1
	}
	return static_cast<uint8_t>(dist);
}

//xiaoxiong 2022.05.20
//计算ADCensus初始代价
//cost_init 大小为Disp_Range*imagesize
__global__ void computeInitCost(const uint8_t* left_rgbs, const uint8_t* right_rgbs, 
	const uint64_t *left_censusImage, const uint64_t *right_censusImage,
	float* cost_init,
	int width, int imagesize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize) 
	{
		
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col
		uint8_t bl = left_rgbs[3 * id];
		uint8_t gl = left_rgbs[3 * id + 1];
		uint8_t rl = left_rgbs[3 * id + 2];
		uint64_t census_val_l = left_censusImage[id];
		for (int d = Min_Disparity; d < Max_Disparity; d++)
		{
			int currentIndex = Disp_Range*id + (d - Min_Disparity);
			int xr = x - d;
			if (xr < 0 || xr >= width) {
				cost_init[currentIndex] = 1.0f;
				continue;
			}
			//计算ad代价
			uint8_t br = right_rgbs[3 * (y * width + xr)];
			uint8_t gr = right_rgbs[3 * (y * width + xr) + 1];
			uint8_t rr = right_rgbs[3 * (y * width + xr) + 2];
			uint64_t census_val_r = right_censusImage[y * width + xr];
			float cost_ad = (abs(bl - br) + abs(gl - gr) + abs(rl - rr)) / 3.0f;
			//计算census代价
			float cost_census = static_cast<float>(Hamming64Distance(census_val_l, census_val_r));
			cost_init[currentIndex] = 1 - exp(-cost_ad / Lambda_ad) + 1 - exp(-cost_census / Lambda_census);
		}
	}
}

__inline__ __device__ int ColorDist(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2, uint8_t g2, uint8_t b2) {
	return __max(abs(r1 - r2), __max(abs(g1 - g2), abs(b1 - b2)));
}

//输入像素位置以及rgb
__inline__ __device__ void findHorizontalArm(int x, int y, const uint8_t* left_rgbs, int& left, int& right, int height, int width)
{
	int colorIndex = 3 * (y*width + x);
	uint8_t r0 = left_rgbs[colorIndex];
	uint8_t g0 = left_rgbs[colorIndex + 1];
	uint8_t b0 = left_rgbs[colorIndex + 2];
	left = right = 0;
	//计算左右臂,先左臂后右臂
	int dir = -1;
	//k = 0 ->左
	//k = 1 ->右
	for (int k = 0; k <= 1; k++) {
		// 延伸臂直到条件不满足
		// 臂长不得超过cross_L1
		int currentindex = colorIndex + dir * 3;
		uint8_t color_lastr = r0;
		uint8_t color_lastg = g0;
		uint8_t color_lastb = b0;
		int xn_current = x + dir;//当前像素x值
		int length = __min(Cross_L1, MAX_ARM_LENGTH);
		for (int n = 0; n < length; n++)
		{
			// 边界处理
			if (k == 0) {
				if (xn_current < 0) {
					break;
				}
			}
			else {
				if (xn_current == width) {
					break;
				}
			}

			// 获取颜色值
			uint8_t color_currentr = left_rgbs[currentindex];
			uint8_t	color_currentg = left_rgbs[currentindex + 1];
			uint8_t	color_currentb = left_rgbs[currentindex + 2];

			// 颜色距离1（臂上像素color_current和中间像素的颜色距离color0）
			int color_dist1 = ColorDist(color_currentr, color_currentg, color_currentb, r0, g0, b0);
			if (color_dist1 >= Cross_t1) {
				break;
			}

			// 颜色距离2（臂上像素color0和前一个像素color_last的颜色距离）
			if (n > 0) {
				int color_dist2 = ColorDist(color_currentr, color_currentg, color_currentb, color_lastr, color_lastg, color_lastb);
				if (color_dist2 >= Cross_t1) {
					break;
				}
			}

			// 臂长大于L2后，颜色距离阈值减小为t2
			if (n + 1 > Cross_L2) {
				if (color_dist1 >= Cross_t2) {
					break;
				}
			}

			if (k == 0) {
				left++;
			}
			else {
				right++;
			}

			color_lastr = color_currentr;
			color_lastg = color_currentg;
			color_lastb = color_currentb;
			xn_current += dir;
			currentindex += dir * 3;
		}
		dir = -dir;
	}
	
}

//输入像素位置以及rgb
__inline__ __device__ void findVerticalArm(int x, int y, const uint8_t* left_rgbs, int& top, int& bottom, int height, int width)
{
	int colorIndex = y*width * 3 + x * 3;
	uint8_t r0 = left_rgbs[colorIndex];
	uint8_t g0 = left_rgbs[colorIndex + 1];
	uint8_t b0 = left_rgbs[colorIndex + 2];
	top = bottom = 0;
	//计算上下臂,先上臂后下臂
	int dir = -1;
	//k = 0 ->上
	//k = 1 ->下
	for (int k = 0; k < 2; k++) {
		// 延伸臂直到条件不满足
		// 臂长不得超过cross_L1
		int currentindex = colorIndex + dir * 3 * width;
		uint8_t color_lastr = r0;
		uint8_t color_lastg = g0;
		uint8_t color_lastb = b0;
		int yn_current = y + dir;//当前像素y值
		int length = __min(Cross_L1, MAX_ARM_LENGTH);
		for (int n = 0; n < length; n++)
		{
			// 边界处理
			if (k == 0) {
				if (yn_current < 0) {
					break;
				}
			}
			else {
				if (yn_current == height) {
					break;
				}
			}

			// 获取颜色值
			uint8_t color_currentr = left_rgbs[currentindex];
			uint8_t	color_currentg = left_rgbs[currentindex + 1];
			uint8_t	color_currentb = left_rgbs[currentindex + 2];

			// 颜色距离1（臂上像素color_current和中间像素的颜色距离color0）
			int color_dist1 = ColorDist(color_currentr, color_currentg, color_currentb, r0, g0, b0);
			if (color_dist1 >= Cross_t1) {
				break;
			}

			// 颜色距离2（臂上像素color0和前一个像素color_last的颜色距离）
			if (n > 0) {
				int color_dist2 = ColorDist(color_currentr, color_currentg, color_currentb, color_lastr, color_lastg, color_lastb);
				if (color_dist2 >= Cross_t1) {
					break;
				}
			}

			// 臂长大于L2后，颜色距离阈值减小为t2
			if (n + 1 > Cross_L2) {
				if (color_dist1 >= Cross_t2) {
					break;
				}
			}

			if (k == 0) {
				top++;
			}
			else {
				bottom++;
			}
			color_lastr = color_currentr;
			color_lastg = color_currentg;
			color_lastb = color_currentb;
			yn_current += dir;
			currentindex += width* dir * 3;
		}
		dir = -dir;
	}
}

//xiaoxiong 2022.05.20
//每个像素构建十字交叉臂
__global__ void buildCorssArms(const uint8_t* left_rgbs, CrossArm* arms,int height, int width, int imagesize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize)
	{
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col
		CrossArm& currentArm = arms[id];
		//是否横竖分开步骤计算比较好 这样子不容易索引冲突
		findHorizontalArm(x, y, left_rgbs, currentArm.left, currentArm.right, height, width);
		currentArm.Width = currentArm.left + currentArm.right + 1;
		findVerticalArm(x, y, left_rgbs, currentArm.top, currentArm.bottom, height, width);
		currentArm.Height = currentArm.top + currentArm.bottom + 1;

	}
}
//xiaoxiong 2022.05.21
//计算像素的支持区像素数量(横竖计算出来的大小是不同的)
__global__ void getSupportPixelCount(CrossArm* arms,  int* vec_supportPixel_count,bool verticaldirection, int height, int width, int imagesize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize)
	{
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col
		CrossArm& arm = arms[y*width + x];
		// vertical
		int count = 0;
		if (verticaldirection)
		{
			for (int t = -arm.top; t <= arm.bottom; t++) {
				int currentPixelArmlength = arms[(y + t)*width + x].Width;
				count += currentPixelArmlength;
			}
		}
		else
		{
			for (int t = -arm.left; t <= arm.right; t++) {
				int currentPixelArmlength = arms[y*width + x + t].Height;
				count += currentPixelArmlength;
			}
		}
		vec_supportPixel_count[id] = count;
	}
}

//xiaoxiong 2022.05.21
//计算像素的支持区像素数量（Old）
__global__ void subPixelCount(CrossArm* arms, int* vec_sup_count_tmp, int** vec_sup_count, int height, int width, int imagesize)
{
	// 注意：两种不同的聚合方向，像素的支持区像素是不同的，需要分开计算
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize)
	{
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col
		bool horizontal_first = true;
		// n=0 : horizontal_first; n=1 : vertical_first
		int direct = horizontal_first ? 0 : 1;
		for (int n = 0; n < 2; n++) 
		{
			for (int k = 0; k < 2; k++) 
			{
				// k=0 : pass1; k=1 : pass2
				CrossArm& arm = arms[y*width + x];
				int count = 0;
				if (horizontal_first) 
				{
					if (k == 0) {
						// horizontal
						//for (int t = -arm.left; t <= arm.right; t++) {
						//	count++;
						//}
						count = count + arm.right + arm.left + 1;
					}
					else {
						// vertical
						for (int t = -arm.top; t <= arm.bottom; t++) {
							count += vec_sup_count_tmp[(y + t)*width + x];
						}
					}
				}
				else {
					if (k == 0) {
						// vertical
						for (int t = -arm.top; t <= arm.bottom; t++) {
							count++;
						}
					}
					else {
						// horizontal
						for (int t = -arm.left; t <= arm.right; t++) {
							count += vec_sup_count_tmp[y*width + x + t];
						}
					}
				}
				if (k == 0) {
					vec_sup_count_tmp[y*width + x] = count;
				}
				else {
					vec_sup_count[direct][y*width + x] = count;
				}
			}
					
		}
		horizontal_first = !horizontal_first;
	}
}

//得到所有像素某一视差的代价值
//可以使用cudaMemcpy代替？？？
__global__ void getCostOfOneDisparity(
	float* costOfOneDisparity,
	float* cost, 
	int d,
	int imagesize)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize)
	{
		//赋值
		costOfOneDisparity[id] = cost[Disp_Range*id + d];
	}
}

//代价聚合（错误写法）
__global__ void getCostForCurrentDisparityInArms
(
	CrossArm* cross_arms,
	float* costs_temp1,//第一个数组是代价数组 第二个数组是临时数组
	float* costs_temp2,//第一个数组是代价数组 第二个数组是临时数组
	float* costs_aggr,
	int* vec_sup_countHf,
	int* vec_sup_countVf,
	int d,
	bool horizontalDirectFirst,
	int height, int width, int imagesize
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize)
	{
		//const int ct_id = horizontalDirectFirst ? 0 : 1;
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col
		int disp = d - Min_Disparity;
		CrossArm current_arm = cross_arms[id];
		//uint16_t current_supCount_H = vec_sup_count[0][id];
		//uint16_t current_supCount_V = vec_sup_count[1][id];
		// 聚合
		float cost = 0.0f;
		//两个方向
		if (horizontalDirectFirst)
		{
			// horizontal
			for (int t = -current_arm.left; t <= current_arm.right; t++) 
			{
				cost += costs_temp1[id + t];
			}
			costs_temp2[id] = cost;//存到第二个数组
			// vertical
			for (int t = -current_arm.top; t <= current_arm.bottom; t++) 
			{
				cost += costs_temp2[(y + t)*width + x];
			}
			costs_aggr[Disp_Range*id + disp] = cost / float(vec_sup_countHf[id]);
		}
		else
		{
			// vertical
			for (int t = -current_arm.top; t <= current_arm.bottom; t++) 
			{
				cost += costs_temp1[(y + t)*width + x];
			}
			costs_temp2[id] = cost;//存到第二个数组
			// horizontal
			for (int t = -current_arm.left; t <= current_arm.right; t++) 
			{
				cost += costs_temp2[id + t];
			}
			costs_aggr[Disp_Range*id + disp] = cost / float(vec_sup_countVf[id]);
		}

	}

}

//代价聚合第二种写法1
__global__ void AggregateInHorizontalDirection
(
	CrossArm* cross_arms,
	float* costs_input,
	float* costs_output,
	int imagesize
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize)
	{
		CrossArm current_arm = cross_arms[id];
		// 聚合
		float cost = 0.0f;
		// horizontal
		for (int t = -current_arm.left; t <= current_arm.right; t++)
		{
			cost += costs_input[id + t];
		}
		costs_output[id] = cost;//输出
	}
}
//代价聚合第二种写法2
__global__ void AggregateInVerticalDirection
(
	CrossArm* cross_arms,
	float* costs_input,
	float* costs_output,
	int height, int width, int imagesize
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize)
	{
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col
		CrossArm current_arm = cross_arms[id];
		// 聚合
		float cost = 0.0f;
		//vertical
		for (int t = -current_arm.top; t <= current_arm.bottom; t++)
		{
			cost += costs_input[(y + t)*width + x];
		}
		costs_output[id] = cost;//输出
	}
}


__global__ void getAggregatedCost
(
	float* costs,
	float* costs_aggr,
	int* sup_counts,
	int d,
	int imagesize
)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < imagesize)
	{
		costs_aggr[Disp_Range*id + d] = costs[id] / float(sup_counts[id]);
	}
}

void aggregateInArms(
	dim3 gridsize,
	int blockSize,
	CrossArm* cross_arms,
	float* costs_temp1,
	float* costs_temp2,
	float* costs_aggr,
	int* sup_countHf,
	int* sup_countVf,
	int d,
	bool horizontalDirectFirst,
	int height, int width, int imagesize)
{
	int disp = d - Min_Disparity;
	getCostOfOneDisparity << <gridsize, blockSize >> > 
		(costs_temp1, costs_aggr, disp, imagesize);

	if (horizontalDirectFirst)
	{
		AggregateInHorizontalDirection << <gridsize, blockSize >> >
			(cross_arms, costs_temp1, costs_temp2, imagesize);
		AggregateInVerticalDirection << <gridsize, blockSize >> >
			(cross_arms, costs_temp2, costs_temp1, height, width, imagesize);
		getAggregatedCost << <gridsize, blockSize >> >
			(costs_temp1, costs_aggr, sup_countHf, disp, imagesize);
	}
	else
	{
		AggregateInVerticalDirection << <gridsize, blockSize >> >
			(cross_arms, costs_temp1, costs_temp2, height, width, imagesize);
		AggregateInHorizontalDirection << <gridsize, blockSize >> >
			(cross_arms, costs_temp2, costs_temp1, imagesize);
		getAggregatedCost << <gridsize, blockSize >> >
			(costs_temp1, costs_aggr, sup_countVf, disp, imagesize);

	}
}

//扫描线优化
__global__ void scanlineOptimizeLeftRight(uint8_t *img_left_, uint8_t *img_right_, 
	float* cost_so_src, float* cost_so_dst, bool is_forward,
	float p1,float p2,float tso, 
	int height,int width)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < height)
	{
		//float flt_max = 3.402823466e+38F;
		float flt_max = 1000000.0f;
		int y = id;
		int direction = is_forward ? 1 : -1;
		//扫描开始的位置
		float* cost_src_row = (is_forward) ? (cost_so_src + y * width * Disp_Range) 
			                                : (cost_so_src + y * width * Disp_Range + (width - 1) * Disp_Range);
		float* cost_dst_row = (is_forward) ? (cost_so_dst + y * width * Disp_Range) 
			                                : (cost_so_dst + y * width * Disp_Range + (width - 1) * Disp_Range);
		uint8_t* img_row_l = (is_forward) ? (img_left_ + y * width * 3) 
			                            : (img_left_ + y * width * 3 + 3 * (width - 1));
		uint8_t* img_row_r = img_right_ + y * width * 3;
		int x = (is_forward) ? 0 : width - 1;
		//第一个像素的rgb
		uint8_t color_l_r = img_row_l[0];
		uint8_t color_l_g = img_row_l[1];
		uint8_t color_l_b = img_row_l[2];
		uint8_t color_l_last_r = color_l_r;
		uint8_t color_l_last_g = color_l_g;
		uint8_t color_l_last_b = color_l_b;
		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		float cost_last_path[Disp_Range + 2];
		//cudaMemcpy(cost_aggr_row, cost_init_row, Disp_Range * sizeof(float), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(cost_last_path + 1, cost_aggr_row, Disp_Range * sizeof(float), cudaMemcpyDeviceToDevice);
		// 初始化：第一个像素的聚合代价值等于初始代价值
		for (int i = 0; i < Disp_Range; i++)
		{
			cost_last_path[i + 1] = cost_dst_row[i] = cost_src_row[i];
		}
		cost_last_path[0] = flt_max;
		cost_last_path[Disp_Range-1] = flt_max;
		
		cost_src_row += direction * Disp_Range;
		cost_dst_row += direction * Disp_Range;
		//指向下一个像素
		img_row_l += direction * 3;
		//指向下一个像素
		x += direction;	

		// 上个像素的最小代价值
		float mincost_last_path = flt_max;
		//赋值为第一个像素的最小代价
		for (auto cost : cost_last_path) {
			mincost_last_path = __min(mincost_last_path, cost);
		}
		// 自方向上第2个像素开始按顺序聚合
		for (int j = 0; j < width - 1; j++) {
			color_l_r = img_row_l[0];
			color_l_g = img_row_l[1];
			color_l_b = img_row_l[2];
			//前后像素的距离
			uint8_t d1 = ColorDist(color_l_r, color_l_g, color_l_b, color_l_last_r, color_l_last_g, color_l_last_b);
			uint8_t d2 = d1;
			float min_cost = flt_max;
			for (int d = 0; d < Disp_Range; d++) {
				int xr = x - d - Min_Disparity;//右图中对应像素的位置
				//计算右图上在上一个像素
				if (xr > 0 && xr < width - 1) {
					uint8_t color_r_r = img_row_r[3 * xr];
					uint8_t color_r_g = img_row_r[3 * xr + 1];
					uint8_t color_r_b = img_row_r[3 * xr + 2];
					uint8_t color_last_r_r = img_row_r[3 * (xr - direction)];
					uint8_t color_last_r_g = img_row_r[3 * (xr - direction) + 1];
					uint8_t color_last_r_b = img_row_r[3 * (xr - direction) + 2];
					//右图中对应点的两个像素之间的距离
					d2 = ColorDist(color_r_r, color_r_g, color_r_b, color_last_r_r, color_last_r_g, color_last_r_b);
				}

				// 计算P1和P2
				float P1(0.0f), P2(0.0f);
				if (d1 < tso && d2 < tso) {
					P1 = p1; P2 = p2;
				}
				else if (d1 < tso && d2 >= tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 < tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 >= tso) {
					P1 = p1 / 10; P2 = p2 / 10;
				}

				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				float  cost = cost_src_row[d];
				float l1 = cost_last_path[d + 1];
				float l2 = cost_last_path[d] + P1;
				float l3 = cost_last_path[d + 2] + P1;
				float l4 = mincost_last_path + P2;

				float cost_s = cost + static_cast<float>(__min(__min(l1, l2), __min(l3, l4)));
				cost_s /= 2;

				cost_dst_row[d] = cost_s;//优化出来的代价值
				min_cost = __min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			for (int i = 0; i < Disp_Range; i++)
			{
				cost_last_path[i + 1] = cost_dst_row[i];
			}
			//cudaMemcpy(cost_last_path, cost_aggr_row, Disp_Range * sizeof(float), cudaMemcpyDeviceToDevice);

			// 下一个像素
			cost_src_row += direction * Disp_Range;
			cost_dst_row += direction * Disp_Range;
			img_row_l += direction * 3;
			x += direction;

			// 像素值重新赋值
			color_l_last_r = color_l_r;
			color_l_last_g = color_l_g;
			color_l_last_b = color_l_b;
		}


	}
}

__global__ void scanlineOptimizeUpDown(uint8_t *img_left_, uint8_t *img_right_,
	float* cost_so_src, float* cost_so_dst, bool is_forward,
	float p1, float p2, float tso,
	int height, int width)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < width)
	{
		const float flt_max = 3.402823466e+38F;
		int x = id;
		int direction = is_forward ? 1 : -1;
		//扫描开始的位置
		float* cost_src_col = (is_forward) ? (cost_so_src + x * Disp_Range)
			: (cost_so_src + (height - 1) * width * Disp_Range + x * Disp_Range);
		float* cost_dst_col = (is_forward) ? (cost_so_dst + x * Disp_Range)
			: (cost_so_dst + (height - 1) * width * Disp_Range + x * Disp_Range);
		uint8_t* img_col = (is_forward) ? (img_left_ + 3 * x) : (img_left_ + (height - 1) * width * 3 + 3 * x);
		int y = (is_forward) ? 0 : height - 1;

		uint8_t color_r = img_col[0];
		uint8_t color_g = img_col[1];
		uint8_t color_b = img_col[2];
		uint8_t color_last_r = color_r;
		uint8_t color_last_g = color_g;
		uint8_t color_last_b = color_b;

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		float cost_last_path[Disp_Range + 2];
		for (int i = 0; i < Disp_Range; i++)
		{
			cost_last_path[i + 1] = cost_dst_col[i] = cost_src_col[i];
		}
		cost_last_path[0] = flt_max;
		cost_last_path[Disp_Range - 1] = flt_max;

		cost_src_col += direction * Disp_Range;
		cost_dst_col += direction * Disp_Range;
		img_col += direction * width * 3;
		y += direction;

		// 上个像素的最小代价值
		float mincost_last_path = flt_max;
		for (auto cost : cost_last_path) {
			mincost_last_path = __min(mincost_last_path, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (int j = 0; j < height - 1; j++) {
			color_r = img_col[0];
			color_g = img_col[1];
			color_b = img_col[2];
			uint8_t d1 = ColorDist(color_r, color_g, color_b, color_last_r, color_last_g, color_last_b);
			uint8_t d2 = d1;
			float min_cost = flt_max;
			for (int d = 0; d < Disp_Range; d++) {
				const int xr = x - d - Min_Disparity;
				//计算右图上在上一个像素
				if (xr > 0 && xr < width - 1) {
					const uint8_t color_r_r = img_right_[y * width * 3 + 3 * xr];
					const uint8_t color_r_g = img_right_[y * width * 3 + 3 * xr + 1];
					const uint8_t color_r_b = img_right_[y * width * 3 + 3 * xr + 2];

					const uint8_t color_last_r_r = img_right_[(y - direction) * width * 3 + 3 * xr];
					const uint8_t color_last_r_g = img_right_[(y - direction) * width * 3 + 3 * xr + 1];
					const uint8_t color_last_r_b = img_right_[(y - direction) * width * 3 + 3 * xr + 2];
					d2 = ColorDist(color_r_r, color_r_g, color_r_b, color_last_r_r, color_last_r_g, color_last_r_b);
				}

				// 计算P1和P2
				float P1(0.0f), P2(0.0f);
				if (d1 < tso && d2 < tso) {
					P1 = p1; P2 = p2;
				}
				else if (d1 < tso && d2 >= tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 < tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 >= tso) {
					P1 = p1 / 10; P2 = p2 / 10;
				}

				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				float  cost = cost_src_col[d];
				float l1 = cost_last_path[d + 1];
				float l2 = cost_last_path[d] + P1;
				float l3 = cost_last_path[d + 2] + P1;
				float l4 = mincost_last_path + P2;

				float cost_s = cost + static_cast<float>(__min(__min(l1, l2), __min(l3, l4)));
				cost_s /= 2;

				cost_dst_col[d] = cost_s;
				min_cost = __min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			for (int i = 0; i < Disp_Range; i++)
			{
				cost_last_path[i + 1] = cost_dst_col[i];
			}
			//cudaMemcpy(cost_last_path, cost_aggr_row, Disp_Range * sizeof(float), cudaMemcpyDeviceToDevice);

			// 下一个像素
			cost_src_col += direction * Disp_Range;
			cost_dst_col += direction * Disp_Range;
			img_col += direction * 3;
			x += direction;

			// 像素值重新赋值
			color_last_r = color_r;
			color_last_g = color_g;
			color_last_b = color_b;
		}
	}
}

//从代价中得到视差
__global__ void computeDisparity(float* costVec,float* disparity, int height, int width)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int imageSize = width*height;
	if (id < width)
	{
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col

		const float flt_max = 3.402823466e+38F;
		float min_cost = flt_max;
		int best_disparity = 0;

		float cost_local[Disp_Range];
		//遍历当前视差，得到最小代价以及
		for (int d = Min_Disparity; d < Max_Disparity; d++)
		{
			const int d_idx = d - Min_Disparity;
			float cost = cost_local[d_idx] = costVec[y * width * Disp_Range + x * Disp_Range + d_idx];
			if (min_cost > cost) {
				min_cost = cost;
				best_disparity = d;
			}
		}
		// ---子像素拟合
		if (best_disparity == Min_Disparity || best_disparity == Max_Disparity - 1) {
			disparity[y * width + x] = INFINITY;
			return;
		}
		// 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
		const int idx_1 = best_disparity - 1 - Min_Disparity;
		const int idx_2 = best_disparity + 1 - Min_Disparity;
		const float cost_1 = cost_local[idx_1];
		const float cost_2 = cost_local[idx_2];
		// 解一元二次曲线极值
		const float denom = cost_1 + cost_2 - 2 * min_cost;
		if (denom != 0.0f) {
			disparity[y * width + x] = static_cast<float>(best_disparity) + (cost_1 - cost_2) / (denom * 2.0f);
		}
		else {
			disparity[y * width + x] = static_cast<float>(best_disparity);
		}
	}
	
}

//从代价中得到视差
__global__ void computeDisparityRight(float* costVec, float* disparity, int height, int width)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int imageSize = width*height;
	if (id < width)
	{
		int y = id / width;//当前像素row
		int x = id%width;//当前像素col

		const float flt_max = 3.402823466e+38F;
		float min_cost = flt_max;
		int best_disparity = 0;

		float cost_local[Disp_Range];
		//遍历当前视差，得到最小代价以及
		for (int d = Min_Disparity; d < Max_Disparity; d++)
		{
			const int d_idx = d - Min_Disparity;
			const int col_left = y + d;
			if (col_left >= 0 && col_left < width) {
				float cost = cost_local[d_idx] =
					costVec[y * width * Disp_Range + col_left * Disp_Range + d_idx];
				if (min_cost > cost) {
					min_cost = cost;
					best_disparity = d;
				}
			}
			else
			{
				cost_local[d_idx] = flt_max;
			}
		}
		// ---子像素拟合
		if (best_disparity == Min_Disparity || best_disparity == Max_Disparity - 1) {
			disparity[y * width + x] = INFINITY;
			return;
		}
		// 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
		const int idx_1 = best_disparity - 1 - Min_Disparity;
		const int idx_2 = best_disparity + 1 - Min_Disparity;
		const float cost_1 = cost_local[idx_1];
		const float cost_2 = cost_local[idx_2];
		// 解一元二次曲线极值
		const float denom = cost_1 + cost_2 - 2 * min_cost;
		if (denom != 0.0f) {
			disparity[y * width + x] = 
				static_cast<float>(best_disparity) + (cost_1 - cost_2) / (denom * 2.0f);
		}
		else {
			disparity[y * width + x] = 
				static_cast<float>(best_disparity);
		}
	}

}

//找到误匹配mismatches以及遮挡区域occlusions
__global__ void outlierDetection(float* disparityLeft, float* disparityRight, int* mismatchFlags, int* occlusionFlags, float lrCheckThreshold, int height, int width)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int imageSize = width*height;
	if (id < imageSize) 
	{
		int y = id/width;//当前像素row
		int x = id%width;//当前像素col

		//引用
		float& disp_left = disparityLeft[y * width + x];
		if (disp_left == INFINITY) {
			mismatchFlags[id] = 1;
			return;
		}
		// 根据视差值找到右影像上对应的同名像素
		long col_right = lround(x - disp_left);//(转换成长整型)当前像素在右视图的同名像素的col值
		if (col_right >= 0 && col_right < width) 
		{
			// 右影像上同名像素的视差值
			float disp_right = disparityRight[y * width + col_right];
			// 判断两个视差值是否一致（差值在阈值内）
			if (abs(disp_left - disp_right) > lrCheckThreshold)
			{
				// 区分遮挡区和误匹配区
				// 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl
				// if(disp_rl > disp) 
				//		pixel in occlusions
				// else 
				//		pixel in mismatches
				int col_rl = lround(col_right + disp_right);
				if (col_rl > 0 && col_rl < width) {
					float disp_l = disparityLeft[y * width + col_rl];
					if (disp_l > disp_left) {
						occlusionFlags[id] = 1;
					}
					else {
						mismatchFlags[id] = 1;
					}
				}
				else {
					mismatchFlags[id] = 1;
				}
				// 让视差值无效
				disp_left = INFINITY;
			}
		}
		else
		{
			// 通过视差值在右影像上找不到同名像素（超出影像范围）
			disp_left = INFINITY;
			mismatchFlags[id] = 1;
		}

	}

}

//
__global__ void regionVotingOneLoop(float* dispLeft, CrossArm* cross_arms,int* errorPts, int irvCountThresh,int irvDispThresh,int height, int width)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int imageSize = width*height;
	if (id < imageSize) 
	{
		//如果是正常的像素点，不用再去估计视差值
		if (errorPts[id] == 0) 
		{
			return;
		}
		int x = id&width;
		int y = id / width;
		float currentdisp = dispLeft[y * width + x];
		if (currentdisp != INFINITY) {
			return;
		}
		int histogram[Disp_Range] = { 0 };

		//计算支持区的视差直方图
		CrossArm currentarm = cross_arms[y * width + x];
		// 遍历支持区像素视差，统计直方图
		// 先竖后横
		for (int t = -currentarm.top; t <= currentarm.bottom; t++)
		{
			const int yt = y + t;
			CrossArm arm2 = cross_arms[yt * width + x];
			for (int s = -arm2.left; s < arm2.right; s++)
			{
				float d = dispLeft[yt * width + x + s];
				if (d != INFINITY) {
					long di = lround(d);
					histogram[di - Min_Disparity]++;//统计视差
				}
			}
		}
		// 计算直方图峰值对应的视差
		int best_disp = 0, count = 0;//最常出视差，所有被统计的像素个数
		int max_ht = 0;
		for (int d = 0; d < Disp_Range; d++) {
			int h = histogram[d];
			if (max_ht < h) {
				max_ht = h;
				best_disp = d;
			}
			count += h;
		}
		//将当前像素视差设置为估计的似乎查之
		if (max_ht > 0) {
			if (count > irvCountThresh && max_ht * 1.0f / count > irvDispThresh) {
				dispLeft[y * width + x] = best_disp + Min_Disparity;
				errorPts[id] = 0;
			}
		}
	}
}
#pragma  region legecy_code

#define TB 128
#define DISP_MAX 256



void checkCudaError() {
	//cudaError_t status = cudaPeekAtLastError();
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}
	else
	{
		printf("cuda error: %i\n", err);
		getchar();
	}
}

#define COLOR_DIFF(x, i, j) \
	__max(abs(x[(i)]               - x[(j)]), \
    __max(abs(x[(i) +   dim2*dim3] - x[(j) +   dim2*dim3]), \
	    abs(x[(i) + 2*dim2*dim3] - x[(j) + 2*dim2*dim3])))

//求像素绝对差
__global__ void ad(float *x0, float *x1, float *output, int size, int size3, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		int d = id;
		int x = d % size3;
		int xy = d % size23;
		d /= size23;

		float dist = 0;
		if (x - d < 0) {
			//dist = CUDART_INF;
			dist = CUDART_INF_F;
		}
		else {
			for (int i = 0; i < 3; i++) {
				int ind = i * size23 + xy;
				dist += fabsf(x0[ind] - x1[ind - d]);
			}
		}
		output[id] = dist / 3;
	}
}

//census变换（计算每个像素的census特征）
__global__ void census9X7(float *x0, float *x1, float *output, int size, int size2, int size3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < size) {
		int d = id;
		int x = d % size3;//中间像素x值
		d /= size3;
		int y = d % size2;//中间像素y值
		d /= size2;

		float dist;
		if (x - d < 0) {
			dist = CUDART_INF_F;
			//dist = CUDART_INF;
		}
		else {
			dist = 0;
			for (int i = 0; i < 3; i++) {
				int ind_p = (i * size2 + y) * size3 + x;
				for (int yy = y - 3; yy <= y + 3; yy++) {
					for (int xx = x - 4; xx <= x + 4; xx++) {
						if (0 <= xx - d && xx < size3 && 0 <= yy && yy < size2) {
							int ind_q = (i * size2 + yy) * size3 + xx;
							if ((x0[ind_q] < x0[ind_p]) != (x1[ind_q - d] < x1[ind_p - d])) {
								dist++;
							}
						}
						else {
							dist++;
						}
					}
				}
			}
		}
		output[id] = dist / 3;
	}
}

__global__ void spatial_argmin(float *input, float *output, int size, int size1, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dim23 = id % size23;
		int dim0 = id / size23;

		int argmin = 0;
		float min = 2e38;
		for (int i = 0; i < size1; i++) {
			float val = input[(dim0 * size1 + i) * size23 + dim23];
			if (val < min) {
				min = val;
				argmin = i;
			}
		}
		output[id] = argmin + 1;
	}
}

__global__ void median3(float *img, float *out, int size, int height, int width)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		const int x = id % width;
		const int y = id / width;

		float w[9] = {
			y == 0 || x == 0 ? 0 : img[id - width - 1],
			y == 0 ? 0 : img[id - width],
			y == 0 || x == width - 1 ? 0 : img[id - width + 1],
			x == 0 ? 0 : img[id - 1],
			img[id],
			x == width - 1 ? 0 : img[id + 1],
			y == height - 1 || x == 0 ? 0 : img[id + width - 1],
			y == height - 1 ? 0 : img[id + width],
			y == height - 1 || x == width - 1 ? 0 : img[id + width + 1]
		};

		for (int i = 0; i < 5; i++) {
			float tmp = w[i];
			int idx = i;
			for (int j = i + 1; j < 9; j++) {
				if (w[j] < tmp) {
					idx = j;
					tmp = w[j];
				}
			}
			w[idx] = w[i];
			w[i] = tmp;
		}

		out[id] = w[4];
	}
}

__global__ void cross(float *x0, float *out, int size, int dim2, int dim3, int L1, int L2, float tau1, float tau2)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int dir = id;
		int x = dir % dim3;
		dir /= dim3;
		int y = dir % dim2;
		dir /= dim2;

		int dx = 0;
		int dy = 0;
		if (dir == 0) {
			dx = -1;
		}
		else if (dir == 1) {
			dx = 1;
		}
		else if (dir == 2) {
			dy = -1;
		}
		else if (dir == 3) {
			dy = 1;
		}
		else {
			assert(0);
		}

		int xx, yy, ind1, ind2, ind3, dist;
		ind1 = y * dim3 + x;
		for (xx = x + dx, yy = y + dy;; xx += dx, yy += dy) {
			if (xx < 0 || xx >= dim3 || yy < 0 || yy >= dim2) break;

			dist = __max(abs(xx - x), abs(yy - y));
			if (dist == 1) continue;

			ind2 = yy * dim3 + xx;
			ind3 = (yy - dy) * dim3 + (xx - dx);

			/* rule 1 */
			if (COLOR_DIFF(x0, ind1, ind2) >= tau1) break;
			if (COLOR_DIFF(x0, ind2, ind3) >= tau1) break;

			/* rule 2 */
			if (dist >= L1) break;

			/* rule 3 */
			if (dist >= L2) {
				if (COLOR_DIFF(x0, ind1, ind2) >= tau2) break;
			}
		}
		out[id] = dir <= 1 ? xx : yy;
	}
}

__global__ void cbca(float *x0c, float *x1c, float *vol, float *out, int size, int dim2, int dim3, int direction)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = id;
		int x = d % dim3;
		d /= dim3;
		int y = d % dim2;
		d /= dim2;

		if (x - d < 0) {
			out[id] = vol[id];
		}
		else {
			float sum = 0;
			int cnt = 0;

			assert(0 <= direction && direction < 2);
			if (direction == 0) {
				int xx_s = __max(x0c[(0 * dim2 + y) * dim3 + x], x1c[(0 * dim2 + y) * dim3 + x - d] + d);
				int xx_t = __min(x0c[(1 * dim2 + y) * dim3 + x], x1c[(1 * dim2 + y) * dim3 + x - d] + d);
				for (int xx = xx_s + 1; xx < xx_t; xx++) {
					int yy_s = __max(x0c[(2 * dim2 + y) * dim3 + xx], x1c[(2 * dim2 + y) * dim3 + xx - d]);
					int yy_t = __min(x0c[(3 * dim2 + y) * dim3 + xx], x1c[(3 * dim2 + y) * dim3 + xx - d]);
					for (int yy = yy_s + 1; yy < yy_t; yy++) {
						sum += vol[(d * dim2 + yy) * dim3 + xx];
						cnt++;
					}
				}
			}
			else {
				int yy_s = __max(x0c[(2 * dim2 + y) * dim3 + x], x1c[(2 * dim2 + y) * dim3 + x - d]);
				int yy_t = __min(x0c[(3 * dim2 + y) * dim3 + x], x1c[(3 * dim2 + y) * dim3 + x - d]);
				for (int yy = yy_s + 1; yy < yy_t; yy++) {
					int xx_s = __max(x0c[(0 * dim2 + yy) * dim3 + x], x1c[(0 * dim2 + yy) * dim3 + x - d] + d);
					int xx_t = __min(x0c[(1 * dim2 + yy) * dim3 + x], x1c[(1 * dim2 + yy) * dim3 + x - d] + d);
					for (int xx = xx_s + 1; xx < xx_t; xx++) {
						sum += vol[(d * dim2 + yy) * dim3 + xx];
						cnt++;
					}
				}
			}

			assert(cnt > 0);
			out[id] = sum / cnt;
		}
	}
}

__global__ void sgm(float *x0, float *x1, float *vol, float *out, int dim1, int dim2, int dim3, float pi1, float pi2, float tau_so, int direction)
{
	int x, y, dx, dy;

	dx = dy = 0;
	assert(0 <= direction && direction < 8);
	if (direction <= 1) {
		y = blockIdx.x * blockDim.x + threadIdx.x;
		if (y >= dim2) {
			return;
		}
		if (direction == 0) {
			x = 0;
			dx = 1;
		}
		else if (direction == 1) {
			x = dim3 - 1;
			dx = -1;
		}
	}
	else if (direction <= 3) {
		x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= dim3) {
			return;
		}
		if (direction == 2) {
			y = 0;
			dy = 1;
		}
		else if (direction == 3) {
			y = dim2 - 1;
			dy = -1;
		}
	}
	else if (direction <= 7) {
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= dim2 + dim3 - 1) {
			return;
		}
		if (direction == 4) {
			if (id < dim2) {
				x = 0;
				y = id;
			}
			else {
				x = id - dim2 + 1;
				y = 0;
			}
			dx = 1;
			dy = 1;
		}
		else if (direction == 5) {
			if (id < dim2) {
				x = dim3 - 1;
				y = id;
			}
			else {
				x = id - dim2 + 1;
				y = dim2 - 1;
			}
			dx = -1;
			dy = -1;
		}
		else if (direction == 6) {
			if (id < dim2) {
				x = 0;
				y = id;
			}
			else {
				x = id - dim2 + 1;
				y = dim2 - 1;
			}
			dx = 1;
			dy = -1;
		}
		else if (direction == 7) {
			if (id < dim2) {
				x = dim3 - 1;
				y = id;
			}
			else {
				x = id - dim2 + 1;
				y = 0;
			}
			dx = -1;
			dy = 1;
		}
	}

	//float min_prev = CUDART_INF;
	float min_prev = CUDART_INF_F;
	for (; 0 <= y && y < dim2 && 0 <= x && x < dim3; x += dx, y += dy) {
		//float min_curr = CUDART_INF;
		float min_curr = CUDART_INF_F;
		for (int d = 0; d < dim1; d++) {
			int ind = (d * dim2 + y) * dim3 + x;
			if (x - d < 0 || y - dy < 0 || y - dy >= dim2 || x - d - dx < 0 || x - dx >= dim3) {
				out[ind] = vol[ind];
			}
			else {
				int ind2 = y * dim3 + x;

				float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * dim3 - dx);
				float D2 = COLOR_DIFF(x1, ind2 - d, ind2 - d - dy * dim3 - dx);
				float P1, P2;
				if (D1 < tau_so && D2 < tau_so) {
					P1 = pi1;
					P2 = pi2;
				}
				else if (D1 > tau_so && D2 > tau_so) {
					P1 = pi1 / 10;
					P2 = pi2 / 10;
				}
				else {
					P1 = pi1 / 4;
					P2 = pi2 / 4;
				}

				//assert(min_prev != CUDART_INF);
				assert(min_prev != CUDART_INF_F);
				float cost = __min(out[ind - dy * dim3 - dx], min_prev + P2);
				if (d > 0) {
					cost = __min(cost, out[ind - dim2 * dim3 - dy * dim3 - dx] + P1);
				}
				if (d < dim1 - 1) {
					cost = __min(cost, out[ind + dim2 * dim3 - dy * dim3 - dx] + P1);
				}
				out[ind] = vol[ind] + cost - min_prev;
			}
			if (out[ind] < min_curr) {
				min_curr = out[ind];
			}
		}
		min_prev = min_curr;
	}
}

__global__ void fliplr(float *in, float *out, int size, int dim3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		out[id + dim3 - 2 * x - 1] = in[id];
	}
}

__global__ void outlier_detection(float *d0, float *d1, float *outlier, int size, int dim3, int disp_max)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int d0i = d0[id];
		if (x - d0i < 0) {
			outlier[id] = 1;
		}
		else if (abs(d0[id] - d1[id - d0i]) < 1.1) {
			outlier[id] = 0; /* match */
		}
		else {
			outlier[id] = 1; /* occlusion */
			for (int d = 0; d < disp_max; d++) {
				if (x - d >= 0 && abs(d - d1[id - d]) < 1.1) {
					outlier[id] = 2; /* mismatch */
					break;
				}
			}
		}
	}
}

__global__ void iterative_region_voting(float *d0, float *x0c, float *x1c, float *outlier, float *d0_out, float *outlier_out, int size, int dim2, int dim3, float tau_s, float tau_h, int disp_max, int direction)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		d0_out[id] = d0[id];
		outlier_out[id] = outlier[id];

		if (outlier[id] == 0) return;

		assert(disp_max < DISP_MAX);
		int hist[DISP_MAX];
		for (int i = 0; i < disp_max; i++) {
			hist[i] = 0;
		}

		assert(0 <= direction && direction < 2);
		if (direction == 0) {
			int xx_s = x0c[(0 * dim2 + y) * dim3 + x];
			int xx_t = x0c[(1 * dim2 + y) * dim3 + x];
			for (int xx = xx_s + 1; xx < xx_t; xx++) {
				int yy_s = x0c[(2 * dim2 + y) * dim3 + xx];
				int yy_t = x0c[(3 * dim2 + y) * dim3 + xx];
				for (int yy = yy_s + 1; yy < yy_t; yy++) {
					if (outlier[yy * dim3 + xx] == 0) {
						hist[(int)d0[yy * dim3 + xx]]++;
					}
				}
			}
		}
		else {
			int yy_s = x0c[(2 * dim2 + y) * dim3 + x];
			int yy_t = x0c[(3 * dim2 + y) * dim3 + x];
			for (int yy = yy_s + 1; yy < yy_t; yy++) {
				int xx_s = x0c[(0 * dim2 + yy) * dim3 + x];
				int xx_t = x0c[(1 * dim2 + yy) * dim3 + x];
				for (int xx = xx_s + 1; xx < xx_t; xx++) {
					if (outlier[yy * dim3 + xx] == 0) {
						hist[(int)d0[yy * dim3 + xx]]++;
					}
				}
			}
		}

		int cnt = 0;
		int max_i = 0;
		for (int i = 0; i < disp_max; i++) {
			cnt += hist[i];
			if (hist[i] > hist[max_i]) {
				max_i = i;
			}
		}

		if (cnt > tau_s && (float)hist[max_i] / cnt > tau_h) {
			outlier_out[id] = 0;
			d0_out[id] = max_i;
		}
	}
}

__global__ void proper_interpolation(float *x0, float *d0, float *outlier, float *out, int size, int dim2, int dim3)
{
	const float dir[] = {
		0   ,  1,
		-0.5,  1,
		-1  ,  1,
		-1  ,  0.5,
		-1  ,  0,
		-1  , -0.5,
		-1  , -1,
		-0.5, -1,
		0   , -1,
		0.5 , -1,
		1   , -1,
		1   , -0.5,
		1   ,  0,
		1   ,  0.5,
		1   ,  1,
		0.5 ,  1
	};

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (outlier[id] == 0) {
			out[id] = d0[id];
			return;
		}

		int x = id % dim3;
		int y = id / dim3;
		//float min_d = CUDART_INF;
		float min_d = CUDART_INF_F;
		//float min_diff = CUDART_INF;
		float min_diff = CUDART_INF_F;
		for (int d = 0; d < 16; d++) {
			float dx = dir[2 * d];
			float dy = dir[2 * d + 1];
			float xx = x;
			float yy = y;
			int xx_i = round(xx);
			int yy_i = round(yy);
			while (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3 && outlier[yy_i * dim3 + xx_i] != 0) {
				xx += dx;
				yy += dy;
				xx_i = round(xx);
				yy_i = round(yy);
			}

			int ind = yy_i * dim3 + xx_i;
			if (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3) {
				assert(outlier[ind] == 0);
				if (outlier[id] == 1) {
					if (d0[ind] < min_d) {
						min_d = d0[ind];
					}
				}
				else if (outlier[id] == 2) {
					float diff = COLOR_DIFF(x0, id, ind);
					if (diff < min_diff) {
						min_diff = diff;
						min_d = d0[ind];
					}
				}
			}
		}
		//assert(min_d != CUDART_INF);
		assert(min_d != CUDART_INF_F);
		out[id] = min_d;
	}
}

__global__ void sobel(float *x, float *g1, float *g2, int size, int dim2, int dim3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int xx = id % dim3;
		int yy = id / dim3;

		if (1 <= yy && yy < dim2 - 1 && 1 <= xx && xx < dim3 - 1) {
			g1[id] = -x[id - dim3 - 1] + x[id - dim3 + 1] - 2 * x[id - 1] + 2 * x[id + 1] - x[id + dim3 - 1] + x[id + dim3 + 1];
			g2[id] = x[id - dim3 - 1] + 2 * x[id - dim3] + x[id - dim3 + 1] - x[id + dim3 - 1] - 2 * x[id + dim3] - x[id + dim3 + 1];
		}
		else {
			g1[id] = 0;
			g2[id] = 0;
		}
	}
}

__global__ void depth_discontinuity_adjustment(float *d0, float *vol, float *g1, float *g2, float *out, int size, int dim23, int dim3, float tau_e)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;

		if (x - (int)d0[id] < 0) {
			out[id] = d0[id];
			return;
		}

		int dp, dc, dn, edge;
		dc = d0[id];
		edge = 0;

		if (g1[id] < -tau_e) {
			dp = d0[id - 1];
			dn = d0[id + 1];
			edge = 1;
		}
		else if (abs(g2[id]) > tau_e) {
			dp = d0[id - dim3];
			dn = d0[id + dim3];
			edge = 1;
		}

		if (edge) {
			if (vol[dp * dim23 + id] < vol[dc * dim23 + id]) dc = dp;
			if (vol[dn * dim23 + id] < vol[dc * dim23 + id]) dc = dn;
		}

		out[id] = dc;
	}
}

__global__ void subpixel_enchancement(float *d0, float *c2, float *out, int size, int dim23, int disp_max) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = d0[id];
		out[id] = d;
		if (1 <= d && d < disp_max - 1) {
			float cn = c2[(d - 1) * dim23 + id];
			float cz = c2[d * dim23 + id];
			float cp = c2[(d + 1) * dim23 + id];
			float denom = 2 * (cp + cn - 2 * cz);
			if (denom > 1e-5) {
				out[id] = d - __min(1.0, __max(-1.0, (cp - cn) / denom));
			}
		}
	}
}

#pragma endregion