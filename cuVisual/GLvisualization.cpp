#include "GLvisualization.h"
//cuda
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
//
#include "cuGLFWbaseFunction.h"
#include <functional>
#include <time.h>
using namespace std;



// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;


GLuint program[2];
const unsigned int PROG_POINT = 0;//

const float fovy = (float)(PI / 4);
const float zNear = 0.10f;
const float zFar = 10.0f;

//�������
int WinWidth = 1280;
int WinHeight = 720;
int pointSize = 1;
bool BreakLoop = false;
double lastX;
double lastY;
float theta = 1.22f;
float phi = -0.70f;
float zoom = 4.0f;
glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraPosition;
glm::mat4 projection;
//�ô��ڵĵ������ϵ��


//�ⲿ��������
int numObjects_fixed = 1000;
int numObjects_rotated = 1000;
int blockSize = 128;

//��init()����
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);//������Ӧ
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);//��갴����Ӧ
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);//���λ����Ӧ
void updateCamera();
void initVAO();
void initShaders(GLuint *program);
//
void runcuda();

PoitCLoudWIN::PoitCLoudWIN()
{
	if (GLPointCloudWinCount==0)
	{
		//����glfwInit��������ʼ��glfw���ڳ�����ֹ֮ǰ��������ֹglfw���������ֻ�������߳��ϱ�����
		if (!glfwInit()) {
			std::cout
				<< "Error: Could not initialize GLFW!"
				<< " Perhaps OpenGL 3.3 isn't available?"
				<< std::endl;
			return;
		}
	}
	GLPointCloudWinCount++;
	//ָ�����ڴ�С
	//ָ����������
	//��ʼ��openGL���
	init();
}

PoitCLoudWIN::~PoitCLoudWIN()
{
	//��ʼ������ܱ�����
	if (GLPointCloudWinCount==1)
	{
		glfwTerminate();
	}
	GLPointCloudWinCount--;
}

//��ӵ��Ʋ�����Ⱦһ���������ɫ
void PoitCLoudWIN::addpointcloud(GLMPointCloud& cloudin)
{

	GLPointCloud currCloud;
	m_pointclouds.push_back(currCloud);
	GLPointCloud* currCloudPt = &(m_pointclouds.back());

	auto points = cloudin.getPtspointer();
	int num = cloudin.getPtspointer()->size();
	if (num == 0)
		return;
	currCloudPt->points_with_colors.resize(num * 4);
	currCloudPt->bindicesVec.resize(num);

	//ȷ��scale�������е��һ��
	if (m_pointclouds.size() == 1)//���յ�һ�����Ƶ�Scale��������
	{
		glm::vec3 lefttop(0, 0, 0);
		glm::vec3 rightdown(1, 1, 1);
		getAABBBox(*points, lefttop, rightdown);
		m_Rscale = m_Rscale < 1.0f / abs(lefttop.x - rightdown.x) ? m_Rscale : 1.0f / abs(lefttop.x - rightdown.x);
		m_Rscale = m_Rscale < 1.0f / abs(lefttop.y - rightdown.y) ? m_Rscale : 1.0f / abs(lefttop.y - rightdown.y);
		m_Rscale = m_Rscale < 1.0f / abs(lefttop.z - rightdown.z) ? m_Rscale : 1.0f / abs(lefttop.z - rightdown.z);
	}
	m_N = m_N + num;

	//�����ɫ
	auto t = time(nullptr);
	srand(t);
	float r = float(rand()) / 3.33f;
	r = abs(r - (int)r);
	float g = float(rand()) / 3.33f;
	g = abs(g - (int)g);
	float b = float(rand()) / 3.33f;
	b = abs(b - (int)b);

	//�����һ�����ƽ������ţ������ɫ
#pragma omp parallel for
	for (int i = 0; i < num; i++) {
		GLPointVertex currPt{ { m_Rscale*(*points)[i].x, m_Rscale*(*points)[i].y, m_Rscale*(*points)[i].z, 1.0f }, { r, g, b, 1.0f } };
		currCloudPt->points_with_colors[i] = currPt;
		currCloudPt->bindicesVec[i] = i;
	}

	//����VAO ��������Ҫ�����ӵĶ�����ճ������
	glGenVertexArrays(1, &(currCloudPt->pointVAO)); // ���ɶ��������������
	//�ú����������ɻ�������������ƣ���һ��������Ҫ���ɵĻ�����������������ڶ�����Ҫ�����洢����������Ƶ�����
	glGenBuffers(1, &(currCloudPt->pointVBO_positions));//����
	glGenBuffers(1, &(currCloudPt->pointIBO));//��������������������
	glBindVertexArray(currCloudPt->pointVAO);//��һ�������������


	// Bind the positions array to the pointVAO by way of the pointVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, currCloudPt->pointVBO_positions); // bind the buffer pointVBO_positions�����һ�����㻺������
	glBufferData(GL_ARRAY_BUFFER, 8 * num * sizeof(GLfloat), (currCloudPt->points_with_colors.begin())._Ptr, GL_DYNAMIC_DRAW);// transfer data�������ͳ�ʼ��һ��buffer object�����ݴ洢��
	
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), 0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(4*sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	//����������Ҳ��buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, currCloudPt->pointIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, num * sizeof(GLuint), ((currCloudPt->bindicesVec).begin())._Ptr, GL_STATIC_DRAW);//GL_STATIC_DRAW
	//glBindVertexArray(0);


}

bool PoitCLoudWIN::init()
{
	//��ʼ��ǰ���ô���ص�����
	glfwSetErrorCallback(errorCallback);
	//�ڴ�������֮ǰ����glfwWindowHint������һЩ���ڵ���Ϣ
	//��Щhints�������Ժ󽫻ᱣ�ֲ��䣬ֻ����glfwWindowHint��glfwDefaultWindowHints����glfwTerminate�޸ġ�
	//���汾�źʹ˰汾�Ŷ���Ϊ3����OpenGL3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);//ǰ�����
	//glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);//�Ƿ��ܵ�����С
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);//ʹ�õ��Ǻ���ģʽ
	//�������windowʧ�����ս�glfw��ָ���ߴ�
	//(int width, int height, const char* title, GLFWmonitor* monitor, GLFWwidnow* share);
	window = glfwCreateWindow(WinWidth, WinHeight, m_windowName.c_str(), NULL, NULL);
	if (!window) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();//glfwTerminate�����ٴ����ͷ���Դ������ڵ��øú����������ʹ��glfw�⺯�����ͱ������³�ʼ��
		return false;
	}

	//
	glfwMakeContextCurrent(window);//������������״̬��ֵ��window
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;//����������GLEW�ڹ���OpenGL�ĺ���ָ��ʱ�����ʹ���ִ����ļ����������������ΪGL_FALSE�Ļ����ܻ���ʹ��OpenGL�ĺ���ģʽʱ����һЩ���⡣
	if (glewInit() != GLEW_OK) {
		std::cout << "Failed to initialize GLEW" << std::endl;
		return false;
	}
	//�����ӿ�view port���ӿ���glfw���ڲ�һ����ͬ��
	int widthGet, heightGet;
	glfwGetFramebufferSize(window, &widthGet, &heightGet);
	glViewport(0, 0, widthGet, heightGet);//�����ӿ��봰�ڴ�Сһ�£������½�0��0+���
	//����ʵ����Ҳ���Խ��ӿڵ�ά������Ϊ��GLFW��ά��С��������֮�����е�OpenGL��Ⱦ������һ����С�Ĵ�������ʾ�������ӵĻ�����Ҳ���Խ�һЩ����Ԫ����ʾ��OpenGL�ӿ�֮�⡣

	// ��ʼ����ͼ״̬
	//initVAO();
	//������
	//cudaGLSetGLDevice(0);//����CUDA����
	////��cuda�Ǽǻ����������������OpenGL��CUDA ��������û�����Ϊ���߹�ͬʹ�á�
	//cudaGLRegisterBufferObject(pointVBO_positions);//pointVBO_positions=0 
	//cudaGLRegisterBufferObject(pointVBO_velocities);//pointVBO_velocities=0

	updateCamera();
	initShaders(program);
	glEnable(GL_DEPTH_TEST);
	return true;
}

void PoitCLoudWIN::joinThread()
{
	std::thread t(&PoitCLoudWIN::show, this);
	t.join();
}

void PoitCLoudWIN::show()
{


	double fps = 0;
	double timebase = 0;
	int frame = 0;

	//����ָ�������Ƿ�رյ�flag�������������κ��߳��б����á�
	while (!glfwWindowShouldClose(window) && !BreakLoop)//game loop
	{
		//���������Ҫ���������Ѿ����¼������е��¼���ͨ�������ڵĻص��¼����������룬���ڵ��ƶ������ڴ�С�ĸı�ȣ�
		//�ص����������Լ��ֶ����ã�����֮ǰ��д�����ô��ڴ�С�Ļص����������û�иú������򲻻���ûص�������ͬʱҲ��������û����룬������������ܵİ��������Ͳ��ᱻ��Ӧ��
		glfwPollEvents();//���ʱ��
		frame++;
		double time = glfwGetTime();//��ǰʱ��
		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}
		//��Ⱦָ��
#pragma region rendering_code 
		
		//runcuda();//���µ���

		//״̬���ú���
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		//״̬Ӧ�ú���
		//glClear(GL_COLOR_BUFFER_BIT);//������glClear�����������ɫ����֮��������ɫ���嶼�ᱻ���ΪglClearColor�������õ���ɫ��
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//
		glUseProgram(program[PROG_POINT]);

		for (int i = 0; i < m_pointclouds.size(); i++)
		{
			glBindBuffer(GL_ARRAY_BUFFER, m_pointclouds[i].pointVBO_positions);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_pointclouds[i].pointIBO);
			glBindVertexArray(m_pointclouds[i].pointVAO);
			glPointSize((GLfloat)pointSize);
			glDrawElements(GL_POINTS, m_pointclouds[i].bindicesVec.size() + 1, GL_UNSIGNED_INT, 0);
		}
		glUseProgram(0);
		glBindVertexArray(0);
#pragma region end
		//��������
		glfwSwapBuffers(window);//opengl����˫������ƣ��ú������ڽ���ǰ����ɫ������������
	}
	//����Ϸѭ��������������Ҫ��ȷ�ͷ� / ɾ��֮ǰ�ķ����������Դ�����ǿ�����main������������glfwTerminate�������ͷ�GLFW������ڴ档
	glfwDestroyWindow(window);
}






void runcuda()
{
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	// ��CUDA�ڴ��ָ��ָ��OpenGL�Ļ��������������ֻ��һ��GPU���Ͳ���Ҫ���ݴ��ݡ���ӳ����ɺ�OpenGL������ʹ�øû�����
	//float4 *dptr = NULL;
	//float *dptrVertPositions = NULL;
	//float *dptrVertVelocities = NULL;
	//cudaGLMapBufferObject((void**)&dptrVertPositions, pointVBO_positions);
	//cudaGLMapBufferObject((void**)&dptrVertVelocities, pointVBO_velocities);
	////
	//float c_scale = 1;
	////�Լ�����ĸ��ƺ���
	//copyPointsToVBO(dptrVertPositions, dptrVertVelocities, numObjects_fixed, numObjects_rotated, blockSize, c_scale);//�Դ�����������໥��ֵ
	//setPointsCUDATOVBO(dptrVertPositions, dptrVertVelocities);

	////ȡ��ӳ��
	//cudaGLUnmapBufferObject(pointVBO_positions);
	//cudaGLUnmapBufferObject(pointVBO_velocities);
}
//������ɫ��
void initShaders(GLuint *program)
{
	GLint location;
	const char *vertexShaderPath = "shaders/boid.vert.glsl";
	const char *geometryShaderPath = "shaders/boid.geom.glsl";
	const char *fragmentShaderPath = "shaders/boid.frag.glsl";
	const char *attributeLocations[] = { "Position", "Velocity" };
	program[PROG_POINT] = glslUtility::createProgram(vertexShaderPath, geometryShaderPath, fragmentShaderPath, attributeLocations, GLuint(2));

	glUseProgram(program[PROG_POINT]);
	if ((location = glGetUniformLocation(program[PROG_POINT], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[PROG_POINT], "u_cameraPos")) != -1) {
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
}

//��gl���õķ��ش�����
void errorCallback(int error, const char *description)
{
	fprintf(stderr, "error %d: %s\n", error, description);
}

void initVAO()
{
	//std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (2 * N)] };//����
	//std::unique_ptr<GLuint[]> bindices{ new GLuint[2 * N] };//����
	//glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	//glm::vec4 lr(1.0, 1.0, 0.0, 0.0);
	//for (int i = 0; i < 2 * N; i++) {
	//	bodies[4 * i + 0] = 0.0f;
	//	bodies[4 * i + 1] = 0.0f;
	//	bodies[4 * i + 2] = 0.0f;
	//	bodies[4 * i + 3] = 1.0f;
	//	bindices[i] = i;
	//}
	////����VAO ��������Ҫ�����ӵĶ�����ճ������
	//glGenVertexArrays(1, &pointVAO); // Attach everything needed to draw a particle to this
	//glGenBuffers(1, &pointVBO_positions);//�ú����������ɻ�������������ƣ���һ��������Ҫ���ɵĻ�����������������ڶ�����Ҫ�����洢����������Ƶ�����
	//glGenBuffers(1, &pointVBO_velocities);
	//glGenBuffers(1, &pointIBO);//��������������������
	//glBindVertexArray(pointVAO);//��һ�������������

	//// Bind the positions array to the pointVAO by way of the pointVBO_positions
	//glBindBuffer(GL_ARRAY_BUFFER, pointVBO_positions); // bind the buffer pointVBO_positions�����һ�����㻺������
	//glBufferData(GL_ARRAY_BUFFER, 4 * (2 * N) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data�������ͳ�ʼ��һ��buffer object�����ݴ洢��
	//glEnableVertexAttribArray(positionLocation);
	//glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	//// Bind the velocities array to the pointVAO by way of the pointVBO_velocities
	//glBindBuffer(GL_ARRAY_BUFFER, pointVBO_velocities);
	//glBufferData(GL_ARRAY_BUFFER, 4 * (2 * N) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	//glEnableVertexAttribArray(velocitiesLocation);
	//glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	////����������Ҳ��buffer
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pointIBO);
	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, (2 * N) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);
	//glBindVertexArray(0);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);//�����½�glfwWindowShouldClose״̬����Ϊtrue
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (leftMousePressed) {
		// compute new camera parameters
		phi += (xpos - lastX) / WinWidth;
		theta -= (ypos - lastY) / WinHeight;
		theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
		updateCamera();
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / WinHeight;
		zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
		updateCamera();
	}
	else if (middleMousePressed){
		glm::vec3 forward = -glm::normalize(cameraPosition);
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = glm::cross(forward, glm::vec3(0, 1, 0));
		right.y = 0.0f;
		right = glm::normalize(right);

		lookAt -= (float)(xpos - lastX) * right * 0.01f;
		lookAt += (float)(ypos - lastY) * forward * 0.01f;
		updateCamera();
	}

	lastX = xpos;
	lastY = ypos;
}

void updateCamera()
{
	cameraPosition.x = zoom * sin(phi) * sin(theta);
	cameraPosition.z = zoom * cos(theta);
	cameraPosition.y = zoom * cos(phi) * sin(theta);
	cameraPosition += lookAt;
	cout << lookAt.x << ", " << lookAt.y << ", " << lookAt.z << "," << endl;
	projection = glm::perspective(fovy, float(WinWidth) / float(WinHeight), zNear, zFar);
	glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
	projection = projection * view;
	GLint location;
	glUseProgram(program[PROG_POINT]);
	if ((location = glGetUniformLocation(program[PROG_POINT], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}