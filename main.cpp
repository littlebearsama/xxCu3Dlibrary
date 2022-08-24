//#include "src/VisualBase.h"
//#include "src/testICP.h"
#include <iostream>
#include "cuRegistration/testICP.h"
#include "cuFilter/testFilters.h"
#include <utility>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>
#include "cu3DBase/utilityCore.hpp"
#include <string>
#include "cu3DBase/glmpointcloud.h"
#include "cuVisual/GLvisualization.h"
#include <thread>
#include <future>
using namespace std;

//#include "cuStereo/testStereo.h"
//0.glfw���ӻ�
void testGLFWviusalization()
{
	//���ص���
	std::string input_filename("testdata//B_000000.txt");
	std::string ext = utilityCore::getFilePathExtension(input_filename);
	//����һ�����Ʊ���
	GLMPointCloud* pointcloud;
	GLMPointCloud* pointcloud2;
	int N, N2;
#define FREQ 1 // sample 1 pt from every FREQ pts in original
#define SEP ' '
	if (ext.compare("txt") == 0) {
		pointcloud = new GLMPointCloud(input_filename, FREQ, SEP);
		N = pointcloud->getNumPoints();
		std::cout << "������Ƶĵ���Ϊ��" << N << std::endl;

	}
	else {
		printf("Non Supported pc1 Format\n");
		return;
	}

	std::string input_filename2("testdata//B_000001.txt");
	std::string ext2 = utilityCore::getFilePathExtension(input_filename2);

	if (ext2.compare("txt") == 0) {
		pointcloud2 = new GLMPointCloud(input_filename2, FREQ, SEP);
		N2 = pointcloud2->getNumPoints();
		std::cout << "������Ƶĵ���Ϊ��" << N2 << std::endl;
	}
	else {
		printf("Non Supported pc1 Format\n");
		return;
	}

	PoitCLoudWIN win;
	//PoitCLoudWIN win2;
	win.addpointcloud(*pointcloud);
	win.addpointcloud(*pointcloud2);
	win.show();
	//win.show();
	//win2.show();
	cout << "���������˵�����ʾ��������" << endl;
	//
}
//1.����cudaicp
void testCUDAICP()
{
	std::string cloudfixed_filename = "testdata//A_000001.txt";
	std::string cloudmove_filename = "testdata//A_000002.txt";
	cuICPSample(cloudfixed_filename, cloudmove_filename);
}
//��������
int main(int argc, char* argv[])
{
	//ȡ��ע��������
	//0. GLFW���ӻ�
	//testGLFWviusalization();
	//1. cuicp
	testCUDAICP();

	std::cout << "functionEND" << std::endl;
	getchar();
}