#pragma once
#include "cuAdcensusBase.h"
#ifdef  DLL_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)   
#endif

#include <string.h>

class cuAdcensus
{
public:
	cuAdcensus(int width,int height);
	~cuAdcensus();
	
	void setIterationTimes(int times);
	void setScanlineOptimizeParameter(float so_p1, float so_p2, float so_tso);//聚合
	void compute(
		uint8_t* left_rgbs_host, 
		uint8_t* right_rgbs_host, 
		float* disparity);
	
private:
	bool memoryAllocForObject(int width, int height);
	void destoryAllmemories();//销毁所有内存 
	void getdisparity(float* disparity_host);
	void getdisparity(float* disparity_host, float* disparity_device);


private:
	int m_width;
	int m_height;
	uint8_t* m_left_rgbs_host;
	uint8_t* m_right_rgbs_host;
	float* m_disparity_host;
	//
	uint8_t* m_left_rgbs_device;
	uint8_t* m_right_rgbs_device;
	uint8_t* m_left_grays_device;
	uint8_t* m_right_grays_device;
	uint64_t *m_left_census_device;
	uint64_t *m_right_census_device;

	CrossArm* m_arms_device;
	//(uint16_t*)m_vec_supportPixel_count[2];//扩展成Disp_Range个可以并行，每次迭代可以分成Disp_Range个流，因为每一层视差都要查询支持区域
	int* m_supportPixelCountHf_device;//扩展成Disp_Range个可以并行，每次迭代可以分成Disp_Range个流，因为每一层视差都要查询支持区域
	int* m_supportPixelcountVf_device;//扩展成Disp_Range个可以并行，每次迭代可以分成Disp_Range个流，因为每一层视差都要查询支持区域
	float* m_cost_init_device;
	//(float* m_cost_per_disp_device)[Disp_Range];//每个视差层的代价
	float* m_cost_temp1_device;//和上面一样扩展成Disp_Range个可以并行，每次迭代可以分成Disp_Range个流
	float* m_cost_temp2_device;//和上面一样扩展成Disp_Range个可以并行，每次迭代可以分成Disp_Range个流
	float* m_cost_aggr_device;

	float* m_disparity_left_device;
	float* m_disparity_right_device;

	float* m_disparity_device;

	bool m_memoryAllocFlag;
	int m_IterationTimes;

	//scanlineOptimize
	float m_so_p1;
	float m_so_p2;
	float m_so_tso;

};


