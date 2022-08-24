#include <vector>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

//__global__ void kernCopyPositionsToVBO(int N, int offset, glm::vec3 *pos, float *vbo, float c_scale);
//__global__ void kernCopyColorsToVBO(int N, int offset, glm::vec3 color, float *vbo);

//vbodptr_positions���������Դ�����Ҫ��ʾ��ĵ㼯
//vbodptr_velocities���������Դ�����Ҫ��ʾ�����ɫ�㼯
void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities, int numObjects_fixed, int numObjects_rotated, int blockSize, float c_scale);
//pointsVec_dev,�Դ�����ļ�������
void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities, std::vector<glm::vec3*>pointsVec_dev, std::vector< glm::vec3>colors, std::vector<int>sizes, int blockSize, float c_scale);
void copyPointsToVBO(std::vector<glm::vec3*>pointsVec_dev, std::vector< glm::vec3>colors, std::vector<int>sizes, int blockSize, float c_scale);
void setPointsCUDATOVBO(float *vbodptr_positions, float *vbodptr_velocities);



