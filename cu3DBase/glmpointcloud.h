#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <fstream>
#include "utilityCore.hpp"

class GLMPointCloud{
public:
    GLMPointCloud(std::string filename, int freq, char sep);
	GLMPointCloud();
	~GLMPointCloud();

	void readPointCloudFile(std::string filename, int freq, char sep);
	void SetPointCloud(int N, glm::vec3*pts);
	std::vector<glm::vec3> getPoints();
	std::vector<glm::vec3>* getPtspointer();
	void savePointcloud_withNAN(std::string filename);
	void savePointcloud(std::string filename);
	void savePointcloud_without000(std::string filename);
	void savedepth(std::string filename, int width, int height);
	int getNumPoints();

private:
    // x y z color
    std::vector<glm::vec3> points;
	int numPoints;
	std::ifstream fp_in;
};