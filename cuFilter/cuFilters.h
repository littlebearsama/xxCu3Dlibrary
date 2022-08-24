#pragma  once
#include <glm/glm.hpp>
#include <vector>
#include "../cu3DBase/glmpointcloud.h"

//对Z进行直通滤波
void cuFilter_passthroughZ(GLMPointCloud& cloudin_host, int N, GLMPointCloud& cloudout_host, GLMPointCloud& outliercloud_host, float Z_up, float Z_down);
//根据近邻搞丢滤波
void cuFilter_DeleleOutliers(GLMPointCloud& cloudin_host, int width, int height, GLMPointCloud& cloudout_host, GLMPointCloud& outliercloud_host, int neighborWin, int NeighborThresh, float depthThresh);




