#pragma  once
#include <glm/glm.hpp>
#include <vector>

#ifdef  DLL_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)   
#endif

//����ο����� Host��
//�������׼���� Host��
//�����׼��ĵ��� Host��
//��������뾶
//�����������������
//��������������
//�����Ӧ��ƽ�����
//�����������
//����任����
DLLEXPORT bool cuICP(
	std::vector<glm::vec3>& cloudFixed,
	std::vector<glm::vec3>& cloudMoved,
	std::vector<glm::vec3>& cloudOut,
	float neighbourDistanceThreshold,
	float convergencePrecision,
	int maxTimes,
	float& averageDis,
	int& times,
	glm::mat4& transfromationMat);
//
