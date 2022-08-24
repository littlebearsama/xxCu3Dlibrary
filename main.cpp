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
//0.glfw可视化
void testGLFWviusalization()
{
	//加载点云
	std::string input_filename("testdata//B_000000.txt");
	std::string ext = utilityCore::getFilePathExtension(input_filename);
	//定义一个点云变量
	GLMPointCloud* pointcloud;
	GLMPointCloud* pointcloud2;
	int N, N2;
#define FREQ 1 // sample 1 pt from every FREQ pts in original
#define SEP ' '
	if (ext.compare("txt") == 0) {
		pointcloud = new GLMPointCloud(input_filename, FREQ, SEP);
		N = pointcloud->getNumPoints();
		std::cout << "输入点云的点数为：" << N << std::endl;

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
		std::cout << "输入点云的点数为：" << N2 << std::endl;
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
	cout << "函数跳到了调用显示函数后面" << endl;
	//
}
//1.测试cudaicp
void testCUDAICP()
{
	std::string cloudfixed_filename = "testdata//A_000001.txt";
	std::string cloudmove_filename = "testdata//A_000002.txt";
	cuICPSample(cloudfixed_filename, cloudmove_filename);
}
//函数声明
int main(int argc, char* argv[])
{
	//取消注释跑例程
	//0. GLFW可视化
	//testGLFWviusalization();
	//1. cuicp
	testCUDAICP();

	std::cout << "functionEND" << std::endl;
	getchar();
}