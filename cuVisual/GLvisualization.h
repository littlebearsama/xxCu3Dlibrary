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
	//����
	GLfloat bodiesVec[4];
	GLfloat rgbsVec[4];
};

struct GLPointCloud
{
	//�ڴ��е�����
	std::vector<GLPointVertex> points_with_colors;
	std::vector<GLuint> bindicesVec;
	//����������
	GLuint pointVAO;//���������������
	GLuint pointIBO;//���������������
	GLuint pointVBO_positions;//GPU�����������
	GLuint pointVBO_velocities;
	//�������
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
	void show();//ÿ�θ������ݵ�ʱ�����
	//std::thread m_Thread;

private:
	GLFWwindow *window = nullptr;
	std::string m_windowName = std::string("GLvisualizationWin");
	bool init();//���ȱ����ã����ڳ�ʼ��OpenGL������һ������ֻ������һ��
	std::vector<GLPointCloud> m_pointclouds;
	//����
	float m_Rscale = FLT_MAX;
	//�ܵ���
	int m_N = 0;
};



