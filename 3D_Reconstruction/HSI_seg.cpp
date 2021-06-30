///*
//* Author：		Yangbo Chen
//* Describe:		图像分割——通过HSV颜色空间
//* Update date：	2021.06.12
//*/
//#define _CRT_SECURE_NO_WARNINGS
//#include <iostream>
//#include <utility>
//#include <opencv2/opencv.hpp>
//using namespace std;
//
//
////******保存像素点的值——三通道图片******
//int savePoint3Channel(const char* filename, const cv::Mat img)
//{
//	if (img.channels() != 3) {
//		cout << "img.channels != 3" << endl;
//		return -1;
//	}
//
//	FILE* fp = fopen(filename, "w");
//	for (int i = 0; i < img.rows; ++i) {
//		for (int j = 0; j < img.cols; ++j) {
//			cv::Vec3b point = img.at<cv::Vec3b>(i, j);
//			fprintf(fp, "%d %d %d,", int(point[0]), int(point[1]), int(point[2]));
//
//		}
//		fprintf(fp, "\n");
//	}
//	fclose(fp);
//	return 1;
//}
//
//
////******保存像素点的值——单通道图片******
//int savePoint1Channel(const char* filename, const cv::Mat img)
//{
//	if (img.channels() != 1) {
//		cout << "img.channels != 3" << endl;
//		return -1;
//	}
//
//	FILE* fp = fopen(filename, "w");
//	for (int i = 0; i < img.rows; ++i) {
//		for (int j = 0; j < img.cols; ++j) {
//			uchar point = img.at<uchar>(i, j);
//			fprintf(fp, "%d,", int(point));
//		}
//		fprintf(fp, "\n");
//	}
//	fclose(fp);
//	return 1;
//}
//
//
//void on_mouse(int Event, int x, int y, int flag, void* usrdata)
//{
//	cv::Mat img;
//	img = *(cv::Mat*)usrdata;
//	cv::Point p(x, y);		//先定义点
//	switch (Event)
//	{
//		case cv::EVENT_LBUTTONDOWN:
//		{
//			printf("h=%d\t", img.at<cv::Vec3b>(p)[0]);
//			printf("s=%d\t", img.at<cv::Vec3b>(p)[1]);
//			printf("v=%d\n", img.at<cv::Vec3b>(p)[2]);
//			//cout << "h/b=" << int(img.at<cv::Vec3b>(p)[0]) << "\t";
//			//cout << "s/g=" << int(img.at<cv::Vec3b>(p)[1]) << "\t";
//			//cout << "v/r=" << int(img.at<cv::Vec3b>(p)[2]) << "\n";
//			cv::circle(img, p, 2, cv::Scalar(255), 3);
//		}
//		default:
//			break;
//	}
//}
//
//
//
////******通过HSI颜色空间进行分割******
//int HSISegmentation(cv::Mat img_hsi, cv::Mat& mask_hsi, 
//	std::pair<int, int> low_H, std::pair<int, int> up_H, std::pair<int, int> S, std::pair<int, int> V)
//{
//	if (img_hsi.empty()) {
//		cout << "img_hsi is NULL" << endl;
//		return -1;
//	}
//	if (img_hsi.channels() != 3) {
//		cout << "img_hsi.channels != 3" << endl;
//		return -1;
//	}
//
//	//通道分割
//	vector<cv::Mat> img_split;
//	cv::split(img_hsi, img_split);
//
//	//阈值分割（三个通道分别进行）
//	cv::Mat mask_H, mask_S, mask_V;
//	cv::Mat mask_H_low, mask_H_up;
//
//	cv::Mat img_H = img_split[0];
//	cv::imshow("img_H", img_H);
//	savePoint1Channel("img_H.csv", img_H);
//	cv::inRange(img_H, low_H.first, low_H.second, mask_H_low);		//下边界范围
//	cv::inRange(img_H, up_H.first, up_H.second, mask_H_up);			//上边界范围
//	cv::bitwise_or(mask_H_low, mask_H_up, mask_H);		//合并（或操作）
//	//自适应阈值分割
//	//cv::Mat imageOtsu;
//	//cv::threshold(img_H, mask_H, 0, 255, CV_THRESH_OTSU);
//	//cv::bitwise_not(mask_H, mask_H);
//	cv::imshow("mask_H", mask_H);
//
//	cv::Mat img_S = img_split[1];
//	cv::imshow("img_S", img_S);
//	savePoint1Channel("img_S.csv", img_S);
//	//自适应阈值分割
//	cv::threshold(img_S, mask_S, 0, 255, CV_THRESH_OTSU);
//	cv::bitwise_not(mask_S, mask_S);
//	//cv::inRange(img_S, S.first, S.second, mask_S);
//	cv::imshow("mask_S", mask_S);
//
//	cv::Mat img_V = img_split[2];
//	cv::imshow("img_V", img_V);
//	savePoint1Channel("img_V.csv", img_V);
//	cv::inRange(img_V, V.first, V.second, mask_V);
//	//cv::threshold(img_S, mask1, S_min, 255, cv::THRESH_BINARY);		//大于S_min的留白
//	//cv::threshold(img_S, mask2, S_max, 255, cv::THRESH_BINARY_INV);	//小于S_max的留白
//	//cv::multiply(mask1, mask2, mask_S);		//矩阵对应元素相乘（合并结果）
//	cv::imshow("mask_V", mask_V);
//
//	//掩码合并
//	cv::bitwise_and(mask_H, mask_S, mask_hsi);
//	cv::bitwise_and(mask_hsi, mask_V, mask_hsi);
//	cv::imshow("mask_hsi", mask_hsi);
//
//	//cv::inRange(img_hsi, cv::Scalar(H_min, S_min, V_min), cv::Scalar(H_max, S_max, V_max), mask_hsi);	//直接检测
//
//	return 1;
//
//}
//
//
//int main()
//{
//	string img_path = "./data02/rect_1280_light/left_3.jpg";
//	//string img_path = "./data_fire/3.jpg";
//	cv::Mat img_src = cv::imread(img_path);
//	cv::imshow("Soure Image", img_src);
//
//	////Otsu阈值分割
//	//cv::Mat imageGray;
//	//cv::cvtColor(imageSrc, imageGray, CV_RGB2GRAY);
//	//cv::Mat imageOtsu;
//	//cv::threshold(imageGray, imageOtsu, 0, 255, CV_THRESH_OTSU);
//	//cv::imshow("Opencv Otsu", imageOtsu);
//
//	/*转换为HSV并保存*/
//	cv::Mat img_hsi;
//	cv::cvtColor(img_src, img_hsi, CV_BGR2HSV);
//	string img_name = img_path + ".HSV.png";
//	cv::imwrite(img_name, img_hsi);
//
//	/*HSV分割*/
//	//HSV掩码获取
//	cv::Mat mask_hsi;
//	std::pair<int, int> low_H(0, 50), up_H(160, 180), S(0, 255), V(127, 255);
//	HSISegmentation(img_hsi, mask_hsi, low_H, up_H, S, V);
//	//形态学处理
//	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
//	cv::morphologyEx(mask_hsi, mask_hsi, cv::MORPH_CLOSE, kernel);		//闭运算，填充前景黑点
//	cv::morphologyEx(mask_hsi, mask_hsi, cv::MORPH_OPEN, kernel);		//开运算，填充背景白点（去噪）
//	cv::morphologyEx(mask_hsi, mask_hsi, cv::MORPH_DILATE, kernel);		//膨胀，增加白色区域
//	//掩码操作
//	cv::Mat img_res;
//	img_src.copyTo(img_res, mask_hsi);
//	cv::imshow("img res", img_res);
//	//savePoint3Channel("./mask_hsi.csv", img_res);
//
//	///*聚类分割*/
//	//cv::Mat img_labels;		//像素点标签
//	//std::vector<cv::Point2f> centers;
//	//cv::kmeans(img_hsi, 3, img_labels, 
//	//	cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
//	//	3, cv::KMEANS_PP_CENTERS, centers);
//	///*欧几里得聚类分割*/
//	//vector<cv::Point> vecPoint;
//	//vector<int> labels;
//	//cv::partition(vecPoint, labels, []() {});
//
//	/*坐标显示*/
//	cv::namedWindow("【HSV Image Display】");
//	cv::setMouseCallback("【HSV Image Display】", on_mouse, &img_hsi);
//	//cv::imshow("HSV Image", img_hsi);
//	//以40ms刷新显示
//	while (1)
//	{
//		imshow("【HSV Image Display】", img_hsi);
//		cv::waitKey(40);
//	}
//
//
//
//	//cv::waitKey();
//	//return 0;
//}
