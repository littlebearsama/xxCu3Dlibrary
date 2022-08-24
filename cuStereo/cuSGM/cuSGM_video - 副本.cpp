#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "libsgm.h"

//////计时函数
#include <chrono>

#ifndef ASSERT_MSG
#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

#endif // !ASSERT_MSG


int main()
{
	//读取双目模型参数
	cv::Mat M1, M2, D1, D2;
	cv::Mat R, T, E, F, R1, R2, P1, P2, Q;
	cv::FileStorage fs1("data/intrinsics.yml", cv::FileStorage::READ);
	if (fs1.isOpened())
	{
		std::cout << "read double cam parameters..." << std::endl;
		fs1["M1"] >> M1;
		fs1["D1"] >> D1;
		fs1["M2"] >> M2;
		fs1["D2"] >> D2;
		fs1["R"] >> R;
		fs1["T"] >> T;
		fs1["R1"] >> R1;
		fs1["P1"] >> P1;
		fs1["R2"] >> R2;
		fs1["P2"] >> P2;
		fs1["Q"] >> Q;

		std::cout << "M1" << M1 << std::endl;
		std::cout << "D1" << D1 << std::endl;
		std::cout << "M2" << M2 << std::endl;
		std::cout << "D2" << D2 << std::endl;
		std::cout << "R" << R << std::endl;
		std::cout << "T" << T << std::endl;
		std::cout << "R1" << R1 << std::endl;
		std::cout << "P1" << P1 << std::endl;
		std::cout << "R2" << R2 << std::endl;
		std::cout << "P2" << P2 << std::endl;
		std::cout << "Q" << Q << std::endl;
		fs1.release();
	}
	else
		return 0;
	//捕获图像
	cv::VideoCapture cap;
	cap.open(0);    //摄像头Index
					//1280X960 X2  YUV 30fps
					//1280X720 X2  YUV 60fps
					//640X480  X2  YUV 90fps
					//1280X960 X2  MJPG30fps
					//1280X720 X2  MJPG30fps
	const int Image_Width = 1280;
	const int Image_height = 960;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, Image_Width * 2); //设置捕获图像的宽度，为双目图像的宽度
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, Image_height);  //设置捕获图像的高度
	cv::Size imageSize(Image_Width, Image_height);
	//捕获当前帧以及拆分左右帧
	cv::Mat frame, frame_L, frame_R, rectifyImageL, rectifyImageR;
	cv::Mat disparity, disparity_8u, disparity_color;
	cv::Mat mask;
	disparity = cv::Mat(imageSize.height, imageSize.width, CV_16S, cv::Scalar::all(0));


	while (1)
	{
		auto at = std::chrono::steady_clock::now();
		cap >> frame;    
		frame_L = frame(cv::Rect(0, 0, Image_Width, Image_height));             //获取左Camera的图像
		frame_R = frame(cv::Rect(Image_Width, 0, Image_Width, Image_height));   //获取右Camera的图像
		//frame_L = frame(cv::Rect(Image_Width, 0, Image_Width, Image_height));   //获取左Camera的图像
		//frame_R = frame(cv::Rect(0, 0, Image_Width, Image_height));             //获取右Camera的图像
		//矫正
		//stereoRectify
		cv::Mat rmapFirst[2], rmapSec[2], rviewFirst, rviewSec;
		cv::initUndistortRectifyMap(M1, D1, R1, P1,
			imageSize, CV_16SC2, rmapFirst[0], rmapFirst[1]);//CV_16SC2
		cv::initUndistortRectifyMap(M2, D2, R2, P2,//CV_16SC2
			imageSize, CV_16SC2, rmapSec[0], rmapSec[1]);

		cv::remap(frame_L, rectifyImageL, rmapFirst[0], rmapFirst[1], cv::INTER_LINEAR);
		cv::remap(frame_R, rectifyImageR, rmapSec[0], rmapSec[1], cv::INTER_LINEAR);
		
		//转成灰度
		cvtColor(rectifyImageL, rectifyImageL, CV_BGR2GRAY);
		cvtColor(rectifyImageR, rectifyImageR, CV_BGR2GRAY);

		ASSERT_MSG(!rectifyImageL.empty() && !rectifyImageR.empty(), "imread failed.");
		ASSERT_MSG(rectifyImageL.size() == rectifyImageR.size() && rectifyImageL.type() == rectifyImageR.type(), "input images must be same size and type.");
		ASSERT_MSG(rectifyImageL.type() == CV_8U || rectifyImageR.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
		//CUSGM
		//参数相关

		int disp_size = 256;//64
		int P1 = 10;//10
		int P2 = 120;//120
		float uniqueness = 0.95;//0.95
		int uniqueness_percent = 95;
		int num_paths = 8;//8
		int min_disp = 0;//0
		int LR_max_diff = 1;//1
		//
		uniqueness = (float)(uniqueness_percent % 100) / 100;
		//
		const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
		const int input_depth = rectifyImageL.type() == CV_8U ? 8 : 16;
		const int output_depth = 16;

		const sgm::StereoSGM::Parameters param(P1, P2, uniqueness, false, path_type, min_disp, LR_max_diff);
		sgm::StereoSGM ssgm(rectifyImageL.cols, rectifyImageL.rows, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_HOST2HOST, param);
		ssgm.execute(rectifyImageL.data, rectifyImageR.data, disparity.data);
		mask = disparity == ssgm.get_invalid_disparity();
		//显示
		disparity.convertTo(disparity_8u, CV_8U, 255. / disp_size);
		cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
		disparity_8u.setTo(0, mask);
		disparity_color.setTo(cv::Scalar(0, 0, 0), mask);
		cv::namedWindow("disparity_color", 1);
		cv::imshow("disparity_color", disparity_color);

		//
		cv::namedWindow("Video_LR", cv::WINDOW_FREERATIO);
		cv::imshow("Video_LR", frame);
		cv::waitKey(1);

		auto a1t = std::chrono::steady_clock::now();
		auto a11t = std::chrono::duration_cast<std::chrono::microseconds>(a1t - at);
		printf("时间：%lf us，帧率 %i fps\n", double(a11t.count()), 1000000 / int(a11t.count()));
	}

	cap.release();         //释放对相机的控制
	return 0;

}

