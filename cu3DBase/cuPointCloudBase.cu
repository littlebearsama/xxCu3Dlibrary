#include"cuPointCloudBase.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda.h"
#include "cuda_texture_types.h"


//�����㣬��flags��ֵΪflag_value�ĵ㱣��������������ΪNAN����
__global__ void cuGetSubPts(int N, glm::vec3* pts_in_dev, glm::vec3* pts_out_dev, int* flags, int flag_value)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N)
	{
		if (flags[index] != flag_value)
		{
			pts_out_dev[index] = glm::vec3(NAN, NAN, NAN);
		}
		else
		{
			pts_out_dev[index] = pts_in_dev[index];
		}
	}
}
