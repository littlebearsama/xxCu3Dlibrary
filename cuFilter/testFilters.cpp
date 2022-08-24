#include <string>
#include "../cu3DBase/utilityCore.hpp"
#include "../cu3DBase/glmpointcloud.h"
#include <iostream>
#include "cuFilters.h"
#include "cuda_runtime.h"
void testdeleleOutliersDepthData(int argc, char* argv[])
{
	//���ص���
	std::string input_filename(argv[1]);
	std::string ext = utilityCore::getFilePathExtension(input_filename);
	//����һ�����Ʊ���
	GLMPointCloud* cloudin_host;
	GLMPointCloud cloudout_host;
	GLMPointCloud outliers_host;
	int N, N2;
#define FREQ 1 // sample 1 pt from every FREQ pts in original
#define SEP ' '
	if (ext.compare("txt") == 0) {
		cloudin_host = new GLMPointCloud(input_filename, FREQ, SEP);
		N = cloudin_host->getNumPoints();
		std::cout << "������Ƶĵ���Ϊ��" << N << std::endl;
	}
	else {
		printf("Non Supported pc1 Format\n");
		return;
	}
	//�������
	int width = 2592;
	int height = 2048;
	std::cout << "��ʼִ��GPU����" << std::endl;
	//��ʱ��ʼ
	cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//ֻ��һ���˲��ͺ��ˣ���������Ϊ���ٽ��ڵ�Ϊ3���߶���ֵ��Ϊ0.1
	//cuFilter_passthroughZ(*cloudin_host, N, cloudout_host, outliers_host, 2, -2);
	//cuFilter_DeleleOutliers(*cloudin_host, width, height, cloudout_host, outliers_host, 3, 3, 0.1);
	cuFilter_DeleleOutliers(cloudout_host, width, height, cloudout_host, outliers_host, 5, 12, 0.1);
	//��ʱ����
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float   elapsedTime;//ms
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//�����������Ҫ��ʱ���
	std::cout << "�˲�ʱ��Ϊ��" << elapsedTime << "ms" << std::endl;
	//�������
	std::string filename("outputdata/filtered_cloud5.txt");
	cloudout_host.savePointcloud(filename);
	std::string filename2("outputdata/outliers_cloud5.txt");
	outliers_host.savePointcloud(filename2);
	////�������ͼ
	//std::string depthname1("outputdata/cloudin_host_height410_width519.txt");
	//cloudin_host->savedepth(depthname1,width,height);
	//std::string depthname2("outputdata/filtered_cloud_height410_width519.txt");
	//cloudout_host.savedepth(depthname2,width,height);
	//std::string depthname3("outputdata/outliers_cloud_height410_width519.txt");
	//outliers_host.savedepth(depthname3,width,height);
	return;
}

