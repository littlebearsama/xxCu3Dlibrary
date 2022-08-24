#pragma  once
#include <glm/glm.hpp>
#include <vector>
#include "../cu3DBase/glmpointcloud.h"

//��Z����ֱͨ�˲�
void cuFilter_passthroughZ(GLMPointCloud& cloudin_host, int N, GLMPointCloud& cloudout_host, GLMPointCloud& outliercloud_host, float Z_up, float Z_down);
//���ݽ��ڸ㶪�˲�
void cuFilter_DeleleOutliers(GLMPointCloud& cloudin_host, int width, int height, GLMPointCloud& cloudout_host, GLMPointCloud& outliercloud_host, int neighborWin, int NeighborThresh, float depthThresh);




