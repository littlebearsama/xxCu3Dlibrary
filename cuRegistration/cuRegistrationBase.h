#include <cuda_runtime.h>
#include <device_launch_parameters.h>//解决不能识别threadId等内部变量的问题
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../cu3DBase/kdtree.h"
#include "../cu3DBase/svd3.h"

//点集平移函数
__global__ void translatePts(int N, glm::vec3* pos_in, glm::vec3* pos_out, glm::mat4 translation);
//点集变换函数
__global__ void transformPoints(int N, glm::vec3 *pos_in, glm::vec3 *pos_out, glm::mat4 transformation);

//辅助函数，缩放点云
__global__ void scalingVec3Array(int N, glm::vec3 *pos, float scale);
//将Int数组的元素初始化
__global__ void setIntArray(int N, int* vec, int value);
//计算协方差矩阵
__global__ void calW(int N, glm::vec3* pos_rotated, glm::vec3* pos_cor, glm::mat3* w);
//kdtree方法得到最近邻对应点集
__global__ void getNearestNeighborKDTree(int N, const glm::vec3 *source, const Node *tree, glm::vec3 *corr, float* corr_distance);
//将邻域阈值内的点的flags置为1，否则置为0
__global__ void isNeighbourhood_setflags(int N, int*flag, float*distance, float threshold);
//取内点
__global__ void getInlierPointset(int N, int*flags, glm::vec3* input, glm::vec3* output);

//格式转换：3X1平移+3X3旋转-->4X4变换
glm::mat4 constructTransformationMatrix2(const glm::vec3 &translation, const glm::vec3& rotation, const glm::vec3& scale);
//格式转换：3X1平移-->4X4变换
glm::mat4 constructTranslationMatrix2(const glm::vec3 &translation);