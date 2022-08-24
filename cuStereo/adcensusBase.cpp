#include "adcensusBase.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>

inline int ColorDist_cpu(const ADColor& c1, const ADColor& c2) {
	return std::max(abs(c1.r - c2.r), std::max(abs(c1.g - c2.g), abs(c1.b - c2.b)));
}

void rgb2gray_cpu(const uint8_t* rgbs, uint8_t* gray, int imagesize)
{
	// 彩色转灰度
	for (size_t i = 0; i < imagesize; i++)
	{
		float b = rgbs[i * 3];
		float g = rgbs[i * 3 + 1];
		float r = rgbs[i * 3 + 2];
		gray[i] = uint8_t(r* 0.299 + g * 0.587 + b * 0.114);
		//if (i == 458) 
		//{
		//	std::cout << "r:" << r << " g:" << g << " b:" << b << std::endl;
		//	std::cout << "gray[i] = uint8_t(r* 0.299 + g * 0.587 + b * 0.114) = " << r* 0.299 + g * 0.587 + b * 0.114 << std::endl;
		//	std::cout << uint8_t(r* 0.299 + g * 0.587 + b * 0.114) << std::endl;
		//	std::cout << (int)rgbs[i * 3  -3] << " " << (int)rgbs[i * 3  -2] << " " << (int)rgbs[i * 3  -1] << std::endl;
		//	std::cout << (int)rgbs[i * 3 + 3] << " " << (int)rgbs[i * 3 + 4] << " " << (int)rgbs[i * 3 + 5] << std::endl;
		//	getchar();
		//}
	}
}

void census9X7_cpu(const uint8_t *garyImage, uint64_t *censusImage, int width, int height, int imagesize)
{
	for (size_t y= 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			if (x < 3 || x >= width - 3 || y < 4 || y >= height - 4)
			{
				continue;
			}
			int currentId = y * width + x;
			uint64_t census_val = 0u;
			uint8_t gray_center = garyImage[currentId];
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
			censusImage[currentId] = census_val;
		}
	}
}

void census3X3_cpu(const uint8_t *garyImage, uint8_t *censusImage, int width, int height, int imagesize)
{
	//todo
}

uint8_t Hamming64Distance_cpu(const uint64_t& x, const uint64_t& y)
{
	uint64_t dist = 0, val = x ^ y;

	// Count the number of set bits
	while (val) {
		++dist;
		val &= val - 1;//效果就是消除了二进制里面的最后一个1
	}
	return static_cast<uint8_t>(dist);
}

void computeInitCost_cpu(const uint8_t* left_rgbs, const uint8_t* right_rgbs, const uint64_t *left_censusImage, const uint64_t *right_censusImage, float* cost_init, int width, int imagesize)
{
	const int height = imagesize / width;
	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			const int id = y*width + x;
			const uint8_t bl = left_rgbs[3 * id];
			const uint8_t gl = left_rgbs[3 * id + 1];
			const uint8_t rl = left_rgbs[3 * id + 2];
			const uint64_t census_val_l = left_censusImage[id];
			for (int d = Min_Disparity; d < Max_Disparity; d++)
			{
				int currentIndex = Disp_Range*id + (d - Min_Disparity);
				const int xr = x - d;
				if (xr < 0 || xr >= width) {
					cost_init[currentIndex] = 1.0f;
					continue;
				}
				//计算ad代价
				const uint8_t br = right_rgbs[3 * (y * width + xr)];
				const uint8_t gr = right_rgbs[3 * (y * width + xr) + 1];
				const uint8_t rr = right_rgbs[3 * (y * width + xr) + 2];
				const uint64_t census_val_r = right_censusImage[y * width + xr];

				const float cost_ad = (abs(bl - br) + abs(gl - gr) + abs(rl - rr)) / 3.0f;
				//计算census代价
				const float cost_census = static_cast<float>(Hamming64Distance_cpu(census_val_l, census_val_r));
				cost_init[currentIndex] = 1 - exp(-cost_ad / Lambda_ad) + 1 - exp(-cost_census / Lambda_census);
			}
		}
	}
}

int ColorDist_cpu(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2, uint8_t g2, uint8_t b2)
{
	return __max(abs(r1 - r2), __max(abs(g1 - g2), abs(b1 - b2)));
}

void findHorizontalArm_cpu(int x, int y, const uint8_t* left_rgbs, int& left, int& right, int height, int width)
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
			const int color_dist1 = ColorDist_cpu(color_currentr, color_currentg, color_currentb, r0, g0, b0);
			if (color_dist1 >= Cross_t1) {
				break;
			}

			// 颜色距离2（臂上像素color0和前一个像素color_last的颜色距离）
			if (n > 0) {
				const int color_dist2 = ColorDist_cpu(color_currentr, color_currentg, color_currentb, color_lastr, color_lastg, color_lastb);
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

void findVerticalArm_cpu(int x, int y, const uint8_t* left_rgbs, int& top, int& bottom, int height, int width)
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
			const int color_dist1 = ColorDist_cpu(color_currentr, color_currentg, color_currentb, r0, g0, b0);
			if (color_dist1 >= Cross_t1) {
				break;
			}

			// 颜色距离2（臂上像素color0和前一个像素color_last的颜色距离）
			if (n > 0) {
				const int color_dist2 = ColorDist_cpu(color_currentr, color_currentg, color_currentb, color_lastr, color_lastg, color_lastb);
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

void buildCorssArms_cpu(const uint8_t* left_rgbs, CrossArm* arms, int height, int width, int imagesize)
{
	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			int id = y * width + x;
			CrossArm& currentArm = arms[id];
			//是否横竖分开步骤计算比较好 这样子不容易索引冲突
			findHorizontalArm_cpu(x, y, left_rgbs, currentArm.left, currentArm.right, height, width);
			currentArm.Width = currentArm.left + currentArm.right + 1;
			findVerticalArm_cpu(x, y, left_rgbs, currentArm.top, currentArm.bottom, height, width);
			currentArm.Height = currentArm.top + currentArm.bottom + 1;
		}
	}


}

void getSupportPixelCount_cpu(CrossArm* arms, int* vec_supportPixel_count, bool verticaldirection, int height, int width, int imagesize)
{
	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			int id = x + y*width;
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


}

void subPixelCount_cpu(CrossArm* arms, int* vec_sup_count_tmp, int** vec_sup_count, int height, int width, int imagesize)
{
	for (size_t y = 0; y < height; y++)
	{
		for (size_t x = 0; x < width; x++)
		{
			int id = y*width + x;
			bool horizontal_first = true;
			// n=0 : horizontal_first; n=1 : vertical_first
			const int direct = horizontal_first ? 0 : 1;
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

}

void aggregateInArms_cpu(
	const int& disparity, 
	float* cost_aggr, 
	bool horizontal_first,int width,int height, 
	const CrossArm* vec_cross_arms,
	const std::vector<float*>& vec_cost_tmp,
	const std::vector<int*>& vec_sup_count)
{
	// 此函数聚合所有像素当视差为disparity时的代价
	const auto disp = disparity - Min_Disparity;
	const int disp_range = Max_Disparity - Min_Disparity;
	if (disp_range <= 0) {
		return;
	}

	// 将disp层的代价存入临时数组vec_cost_tmp_[0]
	// 这样可以避免过多的访问更大的cost_aggr_,提高访问效率
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			vec_cost_tmp[0][y * width + x]
				= cost_aggr[y * width * disp_range + x * disp_range + disp];
		}
	}

	// 逐像素聚合
	const int ct_id = horizontal_first ? 0 : 1;
	for (int k = 0; k < 2; k++) {
		// k==0: pass1
		// k==1: pass2
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				// 获取arm数值
				auto& arm = vec_cross_arms[y*width + x];
				// 聚合
				float cost = 0.0f;
				if (horizontal_first) {
					if (k == 0) {
						// horizontal
						for (int t = -arm.left; t <= arm.right; t++) {
							cost += vec_cost_tmp[0][y * width + x + t];
						}
					}
					else {
						// vertical
						for (int t = -arm.top; t <= arm.bottom; t++) {
							cost += vec_cost_tmp[1][(y + t)*width + x];
						}
					}
				}
				else {
					if (k == 0) {
						// vertical
						for (int t = -arm.top; t <= arm.bottom; t++) {
							cost += vec_cost_tmp[0][(y + t) * width + x];
						}
					}
					else {
						// horizontal
						for (int t = -arm.left; t <= arm.right; t++) {
							cost += vec_cost_tmp[1][y*width + x + t];
						}
					}
				}
				if (k == 0) {
					vec_cost_tmp[1][y*width + x] = cost;
				}
				else {
					cost_aggr[y*width*disp_range + x*disp_range + disp] = cost / vec_sup_count[ct_id][y*width + x];
				}
			}
		}
	}

}

void scanlineOptimizeLeftRight_cpu(uint8_t *img_left_, uint8_t *img_right_,
	float* cost_so_src, float* cost_so_dst, bool is_forward,
	float So_p1_, float So_p2_, float So_tso_,
	int height_, int width_)
{
	const auto width = width_;
	const auto height = height_;
	const auto min_disparity = Min_Disparity;
	const auto max_disparity = Max_Disparity;
	const auto p1 = So_p1_;
	const auto p2 = So_p2_;
	const auto tso = So_tso_;

	//assert(width > 0 && height > 0 && max_disparity > min_disparity);

	// 视差范围
	const int disp_range = max_disparity - min_disparity;

	// 正向(左->右) ：is_forward = true ; direction = 1
	// 反向(右->左) ：is_forward = false; direction = -1;
	const int direction = is_forward ? 1 : -1;

	// 聚合
	for (int y = 0u; y < height; y++) {
		// 路径头为每一行的首(尾,dir=-1)列像素
		auto cost_src_row = (is_forward) ? (cost_so_src + y * width * disp_range) : (cost_so_src + y * width * disp_range + (width - 1) * disp_range);
		auto cost_dst_row = (is_forward) ? (cost_so_dst + y * width * disp_range) : (cost_so_dst + y * width * disp_range + (width - 1) * disp_range);
		auto img_row = (is_forward) ? (img_left_ + y * width * 3) : (img_left_ + y * width * 3 + 3 * (width - 1));
		const auto img_row_r = img_right_ + y * width * 3;
		int x = (is_forward) ? 0 : width - 1;

		// 路径上当前颜色值和上一个颜色值
		ADColor color(img_row[0], img_row[1], img_row[2]);
		ADColor color_last = color;

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<float> cost_last_path(disp_range + 2, FLT_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_dst_row, cost_src_row, disp_range * sizeof(float));
		memcpy(&cost_last_path[1], cost_dst_row, disp_range * sizeof(float));
		cost_src_row += direction * disp_range;
		cost_dst_row += direction * disp_range;
		img_row += direction * 3;
		x += direction;

		// 路径上上个像素的最小代价值
		float mincost_last_path = FLT_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (int j = 0; j < width - 1; j++) {
			color = ADColor(img_row[0], img_row[1], img_row[2]);
			const uint8_t d1 = ColorDist_cpu(color, color_last);
			uint8_t d2 = d1;
			float min_cost = FLT_MAX;
			for (int d = 0; d < disp_range; d++) {
				const int xr = x - d - min_disparity;
				if (xr > 0 && xr < width - 1) {
					const ADColor color_r = ADColor(img_row_r[3 * xr], img_row_r[3 * xr + 1], img_row_r[3 * xr + 2]);
					const ADColor color_last_r = ADColor(img_row_r[3 * (xr - direction)],
						img_row_r[3 * (xr - direction) + 1],
						img_row_r[3 * (xr - direction) + 2]);
					d2 = ColorDist_cpu(color_r, color_last_r);
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
				const float  cost = cost_src_row[d];
				const float l1 = cost_last_path[d + 1];
				const float l2 = cost_last_path[d] + P1;
				const float l3 = cost_last_path[d + 2] + P1;
				const float l4 = mincost_last_path + P2;

				float cost_s = cost + static_cast<float>(std::min(std::min(l1, l2), std::min(l3, l4)));
				cost_s /= 2;

				cost_dst_row[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_dst_row, disp_range * sizeof(float));

			// 下一个像素
			cost_src_row += direction * disp_range;
			cost_dst_row += direction * disp_range;
			img_row += direction * 3;
			x += direction;

			// 像素值重新赋值
			color_last = color;
		}
	}
}

void scanlineOptimizeUpDown_cpu(uint8_t *img_left_, uint8_t *img_right_,
	float* cost_so_src, float* cost_so_dst, bool is_forward,
	float So_p1_, float So_p2_, float So_tso_,
	int height_, int width_)
{
	const auto width = width_;
	const auto height = height_;
	const auto min_disparity = Min_Disparity;
	const auto max_disparity = Max_Disparity;
	const auto p1 = So_p1_;
	const auto p2 = So_p2_;
	const auto tso = So_tso_;

	//assert(width > 0 && height > 0 && max_disparity > min_disparity);

	// 视差范围
	const int disp_range = max_disparity - min_disparity;

	// 正向(上->下) ：is_forward = true ; direction = 1
	// 反向(下->上) ：is_forward = false; direction = -1;
	const int direction = is_forward ? 1 : -1;

	// 聚合
	for (int x = 0; x < width; x++) {
		// 路径头为每一列的首(尾,dir=-1)行像素
		auto cost_src_col = (is_forward) ? (cost_so_src + x * disp_range) : (cost_so_src + (height - 1) * width * disp_range + x * disp_range);
		auto cost_dst_col = (is_forward) ? (cost_so_dst + x * disp_range) : (cost_so_dst + (height - 1) * width * disp_range + x * disp_range);
		auto img_col = (is_forward) ? (img_left_ + 3 * x) : (img_left_ + (height - 1) * width * 3 + 3 * x);
		int y = (is_forward) ? 0 : height - 1;

		// 路径上当前灰度值和上一个灰度值
		ADColor color(img_col[0], img_col[1], img_col[2]);
		ADColor color_last = color;

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<float> cost_last_path(disp_range + 2, FLT_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_dst_col, cost_src_col, disp_range * sizeof(float));
		memcpy(&cost_last_path[1], cost_dst_col, disp_range * sizeof(float));
		cost_src_col += direction * width * disp_range;
		cost_dst_col += direction * width * disp_range;
		img_col += direction * width * 3;
		y += direction;

		// 路径上上个像素的最小代价值
		float mincost_last_path = FLT_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (int i = 0; i < height - 1; i++) {
			color = ADColor(img_col[0], img_col[1], img_col[2]);
			const uint8_t d1 = ColorDist_cpu(color, color_last);
			uint8_t d2 = d1;
			float min_cost = FLT_MAX;
			for (int d = 0; d < disp_range; d++) {
				const int xr = x - d - min_disparity;
				if (xr > 0 && xr < width - 1) {
					const ADColor color_r = ADColor(img_right_[y * width * 3 + 3 * xr], img_right_[y * width * 3 + 3 * xr + 1], img_right_[y * width * 3 + 3 * xr + 2]);
					const ADColor color_last_r = ADColor(img_right_[(y - direction) * width * 3 + 3 * xr],
						img_right_[(y - direction) * width * 3 + 3 * xr + 1],
						img_right_[(y - direction) * width * 3 + 3 * xr + 2]);
					d2 = ColorDist_cpu(color_r, color_last_r);
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
				const float  cost = cost_src_col[d];
				const float l1 = cost_last_path[d + 1];
				const float l2 = cost_last_path[d] + P1;
				const float l3 = cost_last_path[d + 2] + P1;
				const float l4 = mincost_last_path + P2;

				float cost_s = cost + static_cast<float>(std::min(std::min(l1, l2), std::min(l3, l4)));
				cost_s /= 2;

				cost_dst_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_dst_col, disp_range * sizeof(float));

			// 下一个像素
			cost_src_col += direction * width * disp_range;
			cost_dst_col += direction * width * disp_range;
			img_col += direction * width * 3;
			y += direction;

			// 像素值重新赋值
			color_last = color;
		}
	}
}

void computeDisparityRight_cpu(float* costVec, float* disparity, int height, int width)
{

}

void outlierDetection_cpu(float* disparityLeft, float* disparityRight, int* mismatchFlags, int* occlusionFlags, float lrCheckThreshold, int height, int width)
{

}

void regionVotingOneLoop_cpu(float* dispLeft, CrossArm* cross_arms, int* errorPts, int irvCountThresh, int irvDispThresh, int height, int width) {}