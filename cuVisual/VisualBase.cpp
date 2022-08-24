#include "VisualBase.h"

void getAABBBox(const std::vector<glm::vec3>& cloud, glm::vec3& lefttop, glm::vec3& rightdown)
{
	if (cloud.size() < 2)
		return;
	float x_min = FLT_MAX;
	float x_max = -FLT_MAX;
	float y_min = FLT_MAX;
	float y_max = -FLT_MAX;
	float z_min = FLT_MAX;
	float z_max = -FLT_MAX;

	for (size_t i = 0; i < cloud.size(); i++)
	{
		x_min = cloud[i].x < x_min ? cloud[i].x : x_min;
		x_max = cloud[i].x > x_max ? cloud[i].x : x_max;
		y_min = cloud[i].y < y_min ? cloud[i].y : y_min;
		y_max = cloud[i].y > y_max ? cloud[i].y : y_max;
		z_min = cloud[i].z < z_min ? cloud[i].z : z_min;
		z_max = cloud[i].z > z_max ? cloud[i].z : z_max;
	}
	lefttop = glm::vec3(x_min, y_min, z_max);
	rightdown = glm::vec3(x_max, y_max, z_min);
}
