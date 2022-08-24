#pragma  once
#include <glm/glm.hpp>
#include <vector>

#ifdef  DLL_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)   
#endif

//输入参考点云 Host端
//输入待配准点云 Host端
//输出配准后的点云 Host端
//输入邻域半径
//输入迭代的收敛精度
//输入最大迭代次数
//输出对应点平均点距
//输出迭代次数
//输出变换矩阵
DLLEXPORT bool cuICP(
	std::vector<glm::vec3>& cloudFixed,
	std::vector<glm::vec3>& cloudMoved,
	std::vector<glm::vec3>& cloudOut,
	float neighbourDistanceThreshold,
	float convergencePrecision,
	int maxTimes,
	float& averageDis,
	int& times,
	glm::mat4& transfromationMat);
//
