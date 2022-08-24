#pragma once
#include <thread>
#include <functional>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include "../cu3DBase/glslUtility.hpp"
#include "../cu3DBase/utilityCore.hpp"
#include "../cu3DBase/glmpointcloud.h"
#include "VisualBase.h"

static int GLPointCloudWinCount = 0;

struct GLPointVertex
{
	//数据
	GLfloat bodiesVec[4];
	GLfloat rgbsVec[4];
};

struct GLPointCloud
{
	//内存中的数据
	std::vector<GLPointVertex> points_with_colors;
	std::vector<GLuint> bindicesVec;
	//缓冲区对象
	GLuint pointVAO;//顶点数组对象名称
	GLuint pointIBO;//索引数组对象名称
	GLuint pointVBO_positions;//GPU缓存对象名称
	GLuint pointVBO_velocities;
	//数组对象
	GLuint positionLocation;
	GLuint velocitiesLocation;

};

class PoitCLoudWIN
{
public:
	PoitCLoudWIN();
	~PoitCLoudWIN();

public:
	void addpointcloud(GLMPointCloud& cloudin);
	void addpointcloud_Dev(GLMPointCloud& cloudin_dev);
	void addpointcloud_Dev(glm::vec3* cloudin_dev, int num);

	void joinThread();
	void show();//每次更新数据的时候调用
	//std::thread m_Thread;

private:
	GLFWwindow *window = nullptr;
	std::string m_windowName = std::string("GLvisualizationWin");
	bool init();//首先被调用，用于初始化OpenGL函数，一个窗口只被调用一次
	std::vector<GLPointCloud> m_pointclouds;
	//属性
	float m_Rscale = FLT_MAX;
	//总点数
	int m_N = 0;
};



