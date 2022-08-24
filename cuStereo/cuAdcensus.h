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
	void setScanlineOptimizeParameter(float so_p1, float so_p2, float so_tso);//�ۺ�
	void compute(
		uint8_t* left_rgbs_host, 
		uint8_t* right_rgbs_host, 
		float* disparity);
	
private:
	bool memoryAllocForObject(int width, int height);
	void destoryAllmemories();//���������ڴ� 
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
	//(uint16_t*)m_vec_supportPixel_count[2];//��չ��Disp_Range�����Բ��У�ÿ�ε������Էֳ�Disp_Range��������Ϊÿһ���ӲҪ��ѯ֧������
	int* m_supportPixelCountHf_device;//��չ��Disp_Range�����Բ��У�ÿ�ε������Էֳ�Disp_Range��������Ϊÿһ���ӲҪ��ѯ֧������
	int* m_supportPixelcountVf_device;//��չ��Disp_Range�����Բ��У�ÿ�ε������Էֳ�Disp_Range��������Ϊÿһ���ӲҪ��ѯ֧������
	float* m_cost_init_device;
	//(float* m_cost_per_disp_device)[Disp_Range];//ÿ���Ӳ��Ĵ���
	float* m_cost_temp1_device;//������һ����չ��Disp_Range�����Բ��У�ÿ�ε������Էֳ�Disp_Range����
	float* m_cost_temp2_device;//������һ����չ��Disp_Range�����Բ��У�ÿ�ε������Էֳ�Disp_Range����
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


