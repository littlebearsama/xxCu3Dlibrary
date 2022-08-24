#include <iostream>
#include "glmpointcloud.h"
#include <cstring>
#include <glm/gtx/string_cast.hpp>

using namespace std;

GLMPointCloud::GLMPointCloud() {
	numPoints = 0;
}

GLMPointCloud::~GLMPointCloud() {
	this->points.clear();
	this->numPoints = 0;
}
//通过txt文件构造vector类型,通过sep分开数据流ifstream fp_in
GLMPointCloud::GLMPointCloud(std::string filename, int freq, char sep) {
    cout << "Reading Point Cloud From " << filename << "..." << endl;
    char* fname = (char*)filename.c_str();
    this->fp_in.open(fname);

    if (!this->fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
	int count = 0;
    while (this->fp_in.good()) {
        string line;
        utilityCore::safeGetline(this->fp_in, line);
        if (!line.empty()) {
            ++count;
            if (count > freq) count = 1;
            if (count == 1){
                vector<string> tokens = utilityCore::tokenizeString(line, sep);
				//这个元素原地构造，不需要触发拷贝构造和转移构造。
				float x = atof(tokens[0].c_str());
				float y = atof(tokens[1].c_str());
				float z = atof(tokens[2].c_str());
				glm::vec3 pt = glm::vec3(x, y, z);
				if (abs(x) < 1e-6&&abs(y) < 1e-6&&abs(z) < 1e-6)
				{
					pt = glm::vec3(NAN, NAN, NAN);
				}
				points.emplace_back(pt);
			}		
        }
    }
	this->numPoints = points.size();
}


void GLMPointCloud::SetPointCloud(int N, glm::vec3*pts)
{
	numPoints = N;
	points = vector<glm::vec3>(pts, pts + N);
}

std::vector<glm::vec3> GLMPointCloud::getPoints(){
    return this->points;
}


std::vector<glm::vec3>* GLMPointCloud::getPtspointer()
{
	return &points;
}


void GLMPointCloud::savePointcloud_withNAN(std::string filename)
{
	std::ofstream outfile;
	outfile.open(filename);
	for (size_t i = 0; i < numPoints; i++)
	{
		auto pt = points[i];
		outfile << pt.x << " " <<
			pt.y << " " <<
			pt.z << " " << std::endl;
	}
}

void GLMPointCloud::savePointcloud(string filename)
{
	std::ofstream outfile;
	outfile.open(filename);
	for (size_t i = 0; i < numPoints; i++)
	{
		auto pt = points[i];
		if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z))
		{
			continue;
		}
		outfile << pt.x << " " <<
			pt.y << " " <<
			pt.z << " " << std::endl;
	}
}

void GLMPointCloud::savePointcloud_without000(string filename)
{
	std::ofstream outfile;
	outfile.open(filename);
	for (size_t i = 0; i < numPoints; i++)
	{
		auto pt = points[i];
		if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z))
		{
			continue;
		}
		if (abs(pt.x) < 1e-6 &&abs(pt.y) < 1e-6&&abs(pt.z) < 1e-6)
		{
			continue;
		}
		outfile << pt.x << " " <<
			pt.y << " " <<
			pt.z << " " << std::endl;
	}
}

void GLMPointCloud::savedepth(std::string filename, int width, int height)
{
	std::ofstream outfile;
	outfile.open(filename);
	//for (size_t i = 0; i < numPoints; i++)
	//{
	//	auto pt = points[i];
	//	if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z))
	//	{
	//		outfile <<
	//			NAN << " " << std::endl;
	//	}
	//	outfile << 
	//		pt.z << " " << std::endl;
	//}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = j + i*width;
			auto pt = points[index];
			if (isnan(pt.z))
			{
				outfile << "-nan(ind) ";
			}
			else
			{
				outfile <<
					pt.z << " ";
			}
			//outfile << pt.z << " ";
		}
		outfile << std::endl;
	}
}

int GLMPointCloud::getNumPoints() {
	return this->numPoints;
}



