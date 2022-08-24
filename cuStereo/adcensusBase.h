#pragma once
#define Min_Disparity 0
#define Max_Disparity 256
#define Disp_Range 256 //Max_Disparity-Min_Disparity
#define Lambda_ad 10
#define Lambda_census 30

#define MAX_ARM_LENGTH 255 
#define Cross_L1 34
#define Cross_L2 17
#define Cross_t1 20
#define Cross_t2 6

#define IterationTimes 4
#define So_p1  1.0;
#define So_p2  3.0;
#define So_tso 15;
#define Irv_ts  20;
#define Irv_th  0.4;
// 一致性检查阈值
#define Lrcheck_thres = 1.0f;
#include <cstdint> //uint8_t
#include <vector>
/**
* \brief 交叉十字臂结构
* 为了限制内存占用，臂长类型设置为uint8，这意味着臂长最长不能超过255
*/
struct CrossArm {
	int left, right, top, bottom, Height, Width;
	CrossArm() : left(0), right(0), top(0), bottom(0) { }
};
struct pairInt
{
	int x, y;
	pairInt() :x(0), y(0) {}
};

struct ADColor {
	uint8_t r, g, b;
	ADColor() : r(0), g(0), b(0) {}
	ADColor(uint8_t _b, uint8_t _g, uint8_t _r) {
		r = _r; g = _g; b = _b;
	}
};
void rgb2gray_cpu(const uint8_t* rgbs, uint8_t* gray, int imagesize);
void census9X7_cpu(const uint8_t *garyImage, uint64_t *censusImage, int width, int height, int imagesize);
void census3X3_cpu(const uint8_t *garyImage, uint8_t *censusImage, int width, int height, int imagesize);
inline uint8_t  Hamming64Distance_cpu(const uint64_t& x, const uint64_t& y);
void computeInitCost_cpu(const uint8_t* left_rgbs, const uint8_t* right_rgbs,
	const uint64_t *left_censusImage, const uint64_t *right_censusImage,
	float* cost_init,
	int width, int imagesize);
inline int ColorDist_cpu(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2, uint8_t g2, uint8_t b2);
inline void findHorizontalArm_cpu(int x, int y, const uint8_t* left_rgbs, int& left, int& right, int height, int width);
inline void findVerticalArm_cpu(int x, int y, const uint8_t* left_rgbs, int& top, int& bottom, int height, int width);
void buildCorssArms_cpu(const uint8_t* left_rgbs, CrossArm* arms, int height, int width, int imagesize);
void getSupportPixelCount_cpu(CrossArm* arms, int* vec_supportPixel_count, bool verticaldirection, int height, int width, int imagesize);
void subPixelCount_cpu(CrossArm* arms, int* vec_sup_count_tmp, int** vec_sup_count, int height, int width, int imagesize);
void aggregateInArms_cpu(
	const int& disparity,
	float* cost_aggr,
	bool horizontal_first, int width, int height,
	const CrossArm* vec_cross_arms,
	const std::vector<float*>& vec_cost_tmp,
	const std::vector<int*>& vec_sup_count);

void scanlineOptimizeLeftRight_cpu(uint8_t *img_left_, uint8_t *img_right_,
	float* cost_so_src, float* cost_so_dst, bool is_forward,
	float So_p1_, float So_p2_, float So_tso_,
	int height_, int width_);
void scanlineOptimizeUpDown_cpu(uint8_t *img_left_, uint8_t *img_right_,
	float* cost_so_src, float* cost_so_dst, bool is_forward,
	float So_p1_, float So_p2_, float So_tso_,
	int height_, int width_);