#include"cuRegistrationBase.h"


//ͨ��4X4�任����任ÿ����
__global__ void translatePts(int N, glm::vec3* pos_in, glm::vec3* pos_out, glm::mat4 translation) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		pos_out[index] = glm::vec3(translation * glm::vec4(pos_in[index], 1.f));
	}
}
//�㼯�任����
__global__ void transformPoints(int N, glm::vec3 *pos_in, glm::vec3 *pos_out, glm::mat4 transformation) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//��ÿ������б任
	if (index < N) {
		pos_out[index] = glm::vec3(transformation * glm::vec4(pos_in[index], 1.0f));
	}
}

//�������������ŵ���
__global__ void scalingVec3Array(int N, glm::vec3 *pos, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		pos[index].x *= scale;
		pos[index].y *= scale;
		pos[index].z *= scale;
	}
}

//��int�����ʼ��Ϊvalue
__global__ void setIntArray(int N, int* vec, int value){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N)
	{
		vec[index] = value;
	}
}


//��ʽת����3X1ƽ��+3X3��ת-->4X4�任
glm::mat4 constructTransformationMatrix2(const glm::vec3 &translation, const glm::vec3& rotation, const glm::vec3& scale) {
	glm::mat4 translation_matrix = glm::translate(glm::mat4(), translation);
	glm::mat4 rotation_matrix = glm::rotate(glm::mat4(), rotation.x, glm::vec3(1, 0, 0));
	rotation_matrix *= glm::rotate(glm::mat4(), rotation.y, glm::vec3(0, 1, 0));
	rotation_matrix *= glm::rotate(glm::mat4(), rotation.z, glm::vec3(0, 0, 1));
	glm::mat4 scale_matrix = glm::scale(glm::mat4(), scale);
	return translation_matrix* rotation_matrix * scale_matrix;
}

//��ʽת����3X1ƽ��-->4X4�任
glm::mat4 constructTranslationMatrix2(const glm::vec3 &translation) {
	glm::mat4 translation_matrix = glm::translate(glm::mat4(), translation);
	return translation_matrix;
}

//����Э�������
__global__ void calW(int N, glm::vec3* pos_rotated, glm::vec3* pos_cor, glm::mat3* w) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		w[index] = glm::outerProduct(pos_rotated[index], pos_cor[index]);//���;
	}
}

//��������ֵ�ڵĵ��flags��Ϊ1��������Ϊ0
__global__ void isNeighbourhood_setflags(int N, int*flag, float*distance, float threshold)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N)
	{
		flag[index] = distance[index] < threshold ? 1 : 0;
	}
}

//���㳬ƽ����룬���������㼯����kdtree����
__device__ float calHyperPlaneDist(const glm::vec3* pt1, const glm::vec3* pt2, int axis){
	if (axis == 0) {
		return pt1->x - pt2->x;
	}
	else if (axis == 1) {
		return pt1->y - pt2->y;
	}
	else
		return pt1->z - pt2->z;
}

//kdtree�����õ�����ڶ�Ӧ�㼯
__global__ void getNearestNeighborKDTree(int N, const glm::vec3 *source, const Node *tree, glm::vec3 *corr, float* corr_distance){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N){
		glm::vec3 pt = source[index];//�õ����̵߳ĵ�
		float d_closest = glm::distance(tree[0].data, pt);//��ʼ��һ�����롣
		bool explored = false;
		float hyper_dist = calHyperPlaneDist(&pt, &(tree[0].data), tree[0].axis);//
		int curr_node = hyper_dist < 0 ? tree[0].left : tree[0].right;
		int closest_node = 0;
		bool done = false;
		//std::cout << "Closest distance is " << d_closest << ",  source" << pt << ", closest_pt" << tree[closest_node].data << std::endl;
		while (!done){
			// explore current node & below
			while (curr_node != -1){
				float d = glm::distance(tree[curr_node].data, pt);
				if (d < d_closest){
					d_closest = d;
					closest_node = curr_node;
					explored = false;
				}
				hyper_dist = calHyperPlaneDist(&pt, &(tree[curr_node].data), tree[curr_node].axis);
				/*if (index == 0) {
				printf("1. Closest distance is  %f, source: %f, %f, %f,  curr_node :  %f, %f, %f , axis: %d \n", d_closest, pt.x, pt.y, pt.z,
				tree[curr_node].data.x, tree[curr_node].data.y, tree[curr_node].data.z, tree[curr_node].axis);

				}*/
				curr_node = hyper_dist < 0 ? tree[curr_node].left : tree[curr_node].right;

			}
			if (explored || tree[closest_node].parent == -1) {
				done = true;
			}
			else{
				int parent = tree[closest_node].parent;
				hyper_dist = calHyperPlaneDist(&pt, &(tree[parent].data), tree[parent].axis);
				/*if (index == 0) {
				printf("2. Closest distance is  %f, source: %f, %f, %f,  curr_node :  %f, %f, %f , axis: %d \n", d_closest, pt.x, pt.y, pt.z,
				tree[parent].data.x, tree[parent].data.y, tree[parent].data.z, tree[parent].axis);

				}*/
				if (abs(hyper_dist) < d_closest){
					curr_node = hyper_dist < 0 ? tree[parent].left : tree[parent].right;
					explored = true;
				}
				else {
					done = true;
				}
			}
		}
		if (index == 0) {
			/*printf("3. Closest distance is  %f, source: %f, %f, %f,  closest pt :  %f, %f, %f  \n", d_closest, pt.x, pt.y, pt.z,
			tree[closest_node].data.x, tree[closest_node].data.y, tree[closest_node].data.z);*/
		}

		corr[index] = tree[closest_node].data;

		corr_distance[index] = glm::distance(corr[index], pt);

	}
}

//ͨ���������Ϊ0��0��0�ķ�����������ȥ����������һ��������ͬʱ��referance pointcloud �� source pointcloudͬʱ���������Լ���Э�������ʱ������0���������Ϊ0�������ԶԽ��û��Ӱ�졣
__global__ void getInlierPointset(int N, int*flags, glm::vec3* input, glm::vec3* output) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	glm::mat3 flag;
	if (index < N) {
		if (flags[index])
		{
			flag = glm::mat3(1.0f);//������λ����
		}
		else
		{
			flag = glm::mat3(0.0f);//�����
		}
		output[index] = flag*input[index];//���;
	}
}