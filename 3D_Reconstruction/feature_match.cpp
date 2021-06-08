///*
//* Author：		Yangbo Chen
//* Describe:		特征点提取与匹配
//* Update date：	2021.06.08
//*/
//
//#include <iostream>
//#include <vector>
//#include<opencv2/opencv.hpp>
//#include<opencv2/xfeatures2d.hpp>
////using namespace cv;
////using namespace cv::xfeatures2d;
//
//
////特征点提取（SIFT和SURF两种方法）
//void feature_point()
//{
//	cv::Mat img = cv::imread("./Piano/im0.png", CV_LOAD_IMAGE_GRAYSCALE);
//
//	cv::Mat out_img_1, out_img_2, des_1, des_2;
//	std::vector<cv::KeyPoint> pts_1, pts_2;
//
//	//SIFT检测
//	cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
//	sift->detectAndCompute(img, cv::noArray(), pts_1, des_1);
//	//检测到的特征为pts,同时计算描述子为des
//	// sift->detect(img,pts);	//只进行检测特征
//	//绘制特征点
//	cv::drawKeypoints(img, pts_1, out_img_1);
//
//	//SURF检测
//	cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
//	surf->detectAndCompute(img, cv::noArray(),pts_2, des_2);
//	// surf->detect(img,pts);
//	cv::drawKeypoints(img, pts_2, out_img_2);
//
//
//	cv::imshow("Image_SIFT", out_img_1);
//	cv::imshow("Image_SURF", out_img_2);
//	//cv::imwrite("sift.png",out_img);
//	cv::waitKey(0);
//}
//
////特征点提取并匹配（FLANN对高维数据较快） https://www.jb51.net/article/176202.htm
//void point_match()
//{
//	cv:: Mat src1, src2;
//	src1 = cv::imread("./data/640/5_left_0.jpg");
//	src2 = cv::imread("./data/640/5_right_0.jpg");
//	if (src1.empty() || src2.empty())
//	{
//		printf("can ont load images....\n");
//		return;
//	}
//	imshow("image1", src1);
//	imshow("image2", src2);
//
//	int minHessian = 400;
//	//选择SURF特征
//	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
//	std::vector<cv::KeyPoint> keypoints1;
//	std::vector<cv::KeyPoint> keypoints2;
//	cv::Mat descriptor1, descriptor2;
//	//检测关键点并计算描述符
//	detector->detectAndCompute(src1, cv::Mat(), keypoints1, descriptor1);
//	detector->detectAndCompute(src2, cv::Mat(), keypoints2, descriptor2);
//
//	//绘制特征点
//	cv::Mat out1, out2;
//	cv::drawKeypoints(src1, keypoints1, out1);
//	cv::drawKeypoints(src2, keypoints2, out2);
//	cv::imshow("Image_SURF1", out1);
//	cv::imshow("Image_SURF2", out2);
//	//cv::waitKey();
//
//	//基于Flann的描述符匹配器
//	cv::FlannBasedMatcher matcher;
//	std::vector<cv::DMatch> matches;
//	//从查询集中查找每个描述符的最佳匹配
//	matcher.match(descriptor1, descriptor2, matches);
//	double minDist = 1000;
//	double maxDist = 0;
//	for (int i = 0; i < descriptor1.rows; i++) {
//		double dist = matches[i].distance;
//		printf("%f \n", dist);
//		if (dist > maxDist) {
//			maxDist = dist;
//		}
//		if (dist < minDist) {
//			minDist = dist;
//		}
//
//	}
//	//DMatch类用于匹配关键点描述符的
//	std::vector<cv::DMatch>goodMatches;
//	for (int i = 0; i < descriptor1.rows; i++) {
//		double dist = matches[i].distance;
//		if (dist < std::max(2.5*minDist, 0.02)) {
//			goodMatches.push_back(matches[i]);
//		}
//	}
//	cv::Mat matchesImg;
//	drawMatches(src1, keypoints1, src2, keypoints2, goodMatches, matchesImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//	imshow("output", matchesImg);
//
//	cv::waitKey();
//	return;
//}
//
//
//int main()
//{
//	//feature_point();
//	point_match();
//}
