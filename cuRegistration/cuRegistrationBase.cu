#include"cuRegistrationBase.h"


//通过4X4变换矩阵变换每个点
__global__ void translatePts(int N, glm::vec3* pos_in, glm::vec3* pos_out, glm::mat4 translation) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		pos_out[index] = glm::vec3(translation * glm::vec4(pos_in[index], 1.f));
	}
}
//点集变换函数
__global__ void transformPoints(int N, glm::vec3 *pos_in, glm::vec3 *pos_out, glm::mat4 transformation) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//对每个点进行变换
	if (index < N) {
		pos_out[index] = glm::vec3(transformation * glm::vec4(pos_in[index], 1.0f));
	}
}

//辅助函数，缩放点云
__global__ void scalingVec3Array(int N, glm::vec3 *pos, float scale) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		pos[index].x *= scale;
		pos[index].y *= scale;
		pos[index].z *= scale;
	}
}

//将int数组初始化为value
__global__ void setIntArray(int N, int* vec, int value){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N)
	{
		vec[index] = value;
	}
}


//格式转换：3X1平移+3X3旋转-->4X4变换
glm::mat4 constructTransformationMatrix2(const glm::vec3 &translation, const glm::vec3& rotation, const glm::vec3& scale) {
	glm::mat4 translation_matrix = glm::translate(glm::mat4(), translation);
	glm::mat4 rotation_matrix = glm::rotate(glm::mat4(), rotation.x, glm::vec3(1, 0, 0));
	rotation_matrix *= glm::rotate(glm::mat4(), rotation.y, glm::vec3(0, 1, 0));
	rotation_matrix *= glm::rotate(glm::mat4(), rotation.z, glm::vec3(0, 0, 1));
	glm::mat4 scale_matrix = glm::scale(glm::mat4(), scale);
	return translation_matrix* rotation_matrix * scale_matrix;
}

//格式转换：3X1平移-->4X4变换
glm::mat4 constructTranslationMatrix2(const glm::vec3 &translation) {
	glm::mat4 translation_matrix = glm::translate(glm::mat4(), translation);
	return translation_matrix;
}

//计算协方差矩阵
__global__ void calW(int N, glm::vec3* pos_rotated, glm::vec3* pos_cor, glm::mat3* w) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		w[index] = glm::outerProduct(pos_rotated[index], pos_cor[index]);//外积;
	}
}

//将邻域阈值内的点的flags置为1，否则置为0
__global__ void isNeighbourhood_setflags(int N, int*flag, float*distance, float threshold)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < N)
	{
		flag[index] = distance[index] < threshold ? 1 : 0;
	}
}

//计算超平面距离，输入两个点集，被kdtree调用
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

//kdtree方法得到最近邻对应点集
__global__ void getNearestNeighborKDTree(int N, const glm::vec3 *source, const Node *tree, glm::vec3 *corr, float* corr_distance){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N){
		glm::vec3 pt = source[index];//得到该线程的点
		float d_closest = glm::distance(tree[0].data, pt);//初始化一个距离。
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

//通过将外点置为0，0，0的方法将噪声点去除，由于这一步操作是同时将referance pointcloud 和 source pointcloud同时操作，所以计算协方差矩阵时，两个0向量的外积为0矩阵，所以对结果没有影响。
__global__ void getInlierPointset(int N, int*flags, glm::vec3* input, glm::vec3* output) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	glm::mat3 flag;
	if (index < N) {
		if (flags[index])
		{
			flag = glm::mat3(1.0f);//创建单位矩阵
		}
		else
		{
			flag = glm::mat3(0.0f);//零矩阵
		}
		output[index] = flag*input[index];//外积;
	}
}