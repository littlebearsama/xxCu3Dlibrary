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

//窗口相关
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
//该窗口的点的缩放系数


//外部输入数据
int numObjects_fixed = 1000;
int numObjects_rotated = 1000;
int blockSize = 128;

//被init()调用
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);//键盘响应
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);//鼠标按键响应
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);//鼠标位置响应
void updateCamera();
void initVAO();
void initShaders(GLuint *program);
//
void runcuda();

PoitCLoudWIN::PoitCLoudWIN()
{
	if (GLPointCloudWinCount==0)
	{
		//调用glfwInit函数，初始化glfw，在程序终止之前，必须终止glfw，这个函数只能在主线程上被调用
		if (!glfwInit()) {
			std::cout
				<< "Error: Could not initialize GLFW!"
				<< " Perhaps OpenGL 3.3 isn't available?"
				<< std::endl;
			return;
		}
	}
	GLPointCloudWinCount++;
	//指定窗口大小
	//指定窗口名字
	//初始化openGL相关
	init();
}

PoitCLoudWIN::~PoitCLoudWIN()
{
	//初始化后才能被调用
	if (GLPointCloudWinCount==1)
	{
		glfwTerminate();
	}
	GLPointCloudWinCount--;
}

//添加点云并且渲染一种随机的颜色
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

	//确定scale，将所有点归一化
	if (m_pointclouds.size() == 1)//按照第一个点云的Scale进行缩放
	{
		glm::vec3 lefttop(0, 0, 0);
		glm::vec3 rightdown(1, 1, 1);
		getAABBBox(*points, lefttop, rightdown);
		m_Rscale = m_Rscale < 1.0f / abs(lefttop.x - rightdown.x) ? m_Rscale : 1.0f / abs(lefttop.x - rightdown.x);
		m_Rscale = m_Rscale < 1.0f / abs(lefttop.y - rightdown.y) ? m_Rscale : 1.0f / abs(lefttop.y - rightdown.y);
		m_Rscale = m_Rscale < 1.0f / abs(lefttop.z - rightdown.z) ? m_Rscale : 1.0f / abs(lefttop.z - rightdown.z);
	}
	m_N = m_N + num;

	//随机颜色
	auto t = time(nullptr);
	srand(t);
	float r = float(rand()) / 3.33f;
	r = abs(r - (int)r);
	float g = float(rand()) / 3.33f;
	g = abs(g - (int)g);
	float b = float(rand()) / 3.33f;
	b = abs(b - (int)b);

	//对最后一个点云进行缩放，添加颜色
#pragma omp parallel for
	for (int i = 0; i < num; i++) {
		GLPointVertex currPt{ { m_Rscale*(*points)[i].x, m_Rscale*(*points)[i].y, m_Rscale*(*points)[i].z, 1.0f }, { r, g, b, 1.0f } };
		currCloudPt->points_with_colors[i] = currPt;
		currCloudPt->bindicesVec[i] = i;
	}

	//创建VAO 把所有需要画粒子的东西都粘在上面
	glGenVertexArrays(1, &(currCloudPt->pointVAO)); // 生成顶点数组对象名称
	//该函数用来生成缓冲区对象的名称，第一个参数是要生成的缓冲区对象的数量，第二个是要用来存储缓冲对象名称的数组
	glGenBuffers(1, &(currCloudPt->pointVBO_positions));//顶点
	glGenBuffers(1, &(currCloudPt->pointIBO));//生成索引缓存区的名字
	glBindVertexArray(currCloudPt->pointVAO);//绑定一个顶点数组对象


	// Bind the positions array to the pointVAO by way of the pointVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, currCloudPt->pointVBO_positions); // bind the buffer pointVBO_positions变成了一个顶点缓冲类型
	glBufferData(GL_ARRAY_BUFFER, 8 * num * sizeof(GLfloat), (currCloudPt->points_with_colors.begin())._Ptr, GL_DYNAMIC_DRAW);// transfer data，创建和初始化一个buffer object的数据存储。
	
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), 0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(4*sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	//给索引数据也绑定buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, currCloudPt->pointIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, num * sizeof(GLuint), ((currCloudPt->bindicesVec).begin())._Ptr, GL_STATIC_DRAW);//GL_STATIC_DRAW
	//glBindVertexArray(0);


}

bool PoitCLoudWIN::init()
{
	//初始化前设置错误回调函数
	glfwSetErrorCallback(errorCallback);
	//在创建窗口之前调用glfwWindowHint，设置一些窗口的信息
	//这些hints，设置以后将会保持不变，只能由glfwWindowHint、glfwDefaultWindowHints或者glfwTerminate修改。
	//主版本号和此版本号都设为3，即OpenGL3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);//前向兼容
	//glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);//是否能调整大小
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);//使用的是核心模式
	//如果创建window失败则终结glfw，指定尺寸
	//(int width, int height, const char* title, GLFWmonitor* monitor, GLFWwidnow* share);
	window = glfwCreateWindow(WinWidth, WinHeight, m_windowName.c_str(), NULL, NULL);
	if (!window) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();//glfwTerminate会销毁窗口释放资源，因此在调用该函数后，如果想使用glfw库函数，就必须重新初始化
		return false;
	}

	//
	glfwMakeContextCurrent(window);//将瑞金的上下文状态赋值给window
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;//这样做能让GLEW在管理OpenGL的函数指针时更多地使用现代化的技术，如果把它设置为GL_FALSE的话可能会在使用OpenGL的核心模式时出现一些问题。
	if (glewInit() != GLEW_OK) {
		std::cout << "Failed to initialize GLEW" << std::endl;
		return false;
	}
	//设置视口view port（视口与glfw窗口不一定相同）
	int widthGet, heightGet;
	glfwGetFramebufferSize(window, &widthGet, &heightGet);
	glViewport(0, 0, widthGet, heightGet);//设置视口与窗口大小一致，从左下角0，0+宽高
	//我们实际上也可以将视口的维度设置为比GLFW的维度小，这样子之后所有的OpenGL渲染将会在一个更小的窗口中显示，这样子的话我们也可以将一些其它元素显示在OpenGL视口之外。

	// 初始化绘图状态
	//initVAO();
	//互操作
	//cudaGLSetGLDevice(0);//设置CUDA环境
	////用cuda登记缓冲区，该命令告诉OpenGL和CUDA 驱动程序该缓冲区为二者共同使用。
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

	//返回指定窗口是否关闭的flag变量，可以在任何线程中被调用。
	while (!glfwWindowShouldClose(window) && !BreakLoop)//game loop
	{
		//这个函数主要用来处理已经在事件队列中的事件，通常处理窗口的回调事件，包括输入，窗口的移动，窗口大小的改变等，
		//回调函数可以自己手动设置，比如之前所写的设置窗口大小的回调函数；如果没有该函数，则不会调用回调函数，同时也不会接收用户输入，例如接下来介绍的按键交互就不会被响应；
		glfwPollEvents();//检查时间
		frame++;
		double time = glfwGetTime();//当前时间
		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}
		//渲染指令
#pragma region rendering_code 
		
		//runcuda();//更新点云

		//状态设置函数
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		//状态应用函数
		//glClear(GL_COLOR_BUFFER_BIT);//当调用glClear函数，清除颜色缓冲之后，整个颜色缓冲都会被填充为glClearColor里所设置的颜色。
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
		//交换缓冲
		glfwSwapBuffers(window);//opengl采用双缓冲机制，该函数用于交换前后颜色缓冲区的内容
	}
	//当游戏循环结束后我们需要正确释放 / 删除之前的分配的所有资源。我们可以在main函数的最后调用glfwTerminate函数来释放GLFW分配的内存。
	glfwDestroyWindow(window);
}






void runcuda()
{
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	// 将CUDA内存的指针指向OpenGL的缓冲区，这样如果只有一个GPU，就不需要数据传递。当映射完成后，OpenGL不能再使用该缓冲区
	//float4 *dptr = NULL;
	//float *dptrVertPositions = NULL;
	//float *dptrVertVelocities = NULL;
	//cudaGLMapBufferObject((void**)&dptrVertPositions, pointVBO_positions);
	//cudaGLMapBufferObject((void**)&dptrVertVelocities, pointVBO_velocities);
	////
	//float c_scale = 1;
	////自己定义的复制函数
	//copyPointsToVBO(dptrVertPositions, dptrVertVelocities, numObjects_fixed, numObjects_rotated, blockSize, c_scale);//显存里面的数据相互赋值
	//setPointsCUDATOVBO(dptrVertPositions, dptrVertVelocities);

	////取消映射
	//cudaGLUnmapBufferObject(pointVBO_positions);
	//cudaGLUnmapBufferObject(pointVBO_velocities);
}
//加载着色器
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

//被gl调用的返回错误函数
void errorCallback(int error, const char *description)
{
	fprintf(stderr, "error %d: %s\n", error, description);
}

void initVAO()
{
	//std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (2 * N)] };//数据
	//std::unique_ptr<GLuint[]> bindices{ new GLuint[2 * N] };//索引
	//glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	//glm::vec4 lr(1.0, 1.0, 0.0, 0.0);
	//for (int i = 0; i < 2 * N; i++) {
	//	bodies[4 * i + 0] = 0.0f;
	//	bodies[4 * i + 1] = 0.0f;
	//	bodies[4 * i + 2] = 0.0f;
	//	bodies[4 * i + 3] = 1.0f;
	//	bindices[i] = i;
	//}
	////创建VAO 把所有需要画粒子的东西都粘在上面
	//glGenVertexArrays(1, &pointVAO); // Attach everything needed to draw a particle to this
	//glGenBuffers(1, &pointVBO_positions);//该函数用来生成缓冲区对象的名称，第一个参数是要生成的缓冲区对象的数量，第二个是要用来存储缓冲对象名称的数组
	//glGenBuffers(1, &pointVBO_velocities);
	//glGenBuffers(1, &pointIBO);//生成索引缓存区的名字
	//glBindVertexArray(pointVAO);//绑定一个顶点数组对象

	//// Bind the positions array to the pointVAO by way of the pointVBO_positions
	//glBindBuffer(GL_ARRAY_BUFFER, pointVBO_positions); // bind the buffer pointVBO_positions变成了一个顶点缓冲类型
	//glBufferData(GL_ARRAY_BUFFER, 4 * (2 * N) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data，创建和初始化一个buffer object的数据存储。
	//glEnableVertexAttribArray(positionLocation);
	//glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	//// Bind the velocities array to the pointVAO by way of the pointVBO_velocities
	//glBindBuffer(GL_ARRAY_BUFFER, pointVBO_velocities);
	//glBufferData(GL_ARRAY_BUFFER, 4 * (2 * N) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	//glEnableVertexAttribArray(velocitiesLocation);
	//glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	////给索引数据也绑定buffer
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pointIBO);
	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, (2 * N) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);
	//glBindVertexArray(0);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);//弱按下将glfwWindowShouldClose状态设置为true
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