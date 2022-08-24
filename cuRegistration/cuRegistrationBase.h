#include <cuda_runtime.h>
#include <device_launch_parameters.h>//�������ʶ��threadId���ڲ�����������
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../cu3DBase/kdtree.h"
#include "../cu3DBase/svd3.h"

//�㼯ƽ�ƺ���
__global__ void translatePts(int N, glm::vec3* pos_in, glm::vec3* pos_out, glm::mat4 translation);
//�㼯�任����
__global__ void transformPoints(int N, glm::vec3 *pos_in, glm::vec3 *pos_out, glm::mat4 transformation);

//�������������ŵ���
__global__ void scalingVec3Array(int N, glm::vec3 *pos, float scale);
//��Int�����Ԫ�س�ʼ��
__global__ void setIntArray(int N, int* vec, int value);
//����Э�������
__global__ void calW(int N, glm::vec3* pos_rotated, glm::vec3* pos_cor, glm::mat3* w);
//kdtree�����õ�����ڶ�Ӧ�㼯
__global__ void getNearestNeighborKDTree(int N, const glm::vec3 *source, const Node *tree, glm::vec3 *corr, float* corr_distance);
//��������ֵ�ڵĵ��flags��Ϊ1��������Ϊ0
__global__ void isNeighbourhood_setflags(int N, int*flag, float*distance, float threshold);
//ȡ�ڵ�
__global__ void getInlierPointset(int N, int*flags, glm::vec3* input, glm::vec3* output);

//��ʽת����3X1ƽ��+3X3��ת-->4X4�任
glm::mat4 constructTransformationMatrix2(const glm::vec3 &translation, const glm::vec3& rotation, const glm::vec3& scale);
//��ʽת����3X1ƽ��-->4X4�任
glm::mat4 constructTranslationMatrix2(const glm::vec3 &translation);