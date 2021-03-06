///* 
//* Author：		Yangbo Chen
//* Describe:		阈值分割
//* Update date：	2021.06.08
//*/
//#define _CRT_SECURE_NO_WARNINGS
//#include <opencv2/highgui/highgui.hpp>  
//#include <opencv2/imgproc/imgproc.hpp>  
//#include <opencv2/core/core.hpp> 
//#include <iostream>
//
////using namespace cv;
//using namespace std;
//
//
////******Otsu算法通过求类间方差极大值求自适应阈值******
//int OtsuAlgThreshold(const cv::Mat image)
//{
//	if (image.channels() != 1)
//	{
//		cout << "Please input Gray-image!" << endl;
//		return 0;
//	}
//	int T = 0; //Otsu算法阈值
//	double varValue = 0; //类间方差中间值保存
//	double w0 = 0; //前景像素点数所占比例
//	double w1 = 0; //背景像素点数所占比例
//	double u0 = 0; //前景平均灰度
//	double u1 = 0; //背景平均灰度
//	double Histogram[256] = { 0 }; //灰度直方图，下标是灰度值，保存内容是灰度值对应的像素点总数
//	int Histogram1[256] = { 0 };
//	uchar *data = image.data;
//	double totalNum = image.rows*image.cols; //像素总数
//	//计算灰度直方图分布，Histogram数组下标是灰度值，保存内容是灰度值对应像素点数
//	for (int i = 0; i < image.rows; i++)   //为表述清晰，并没有把rows和cols单独提出来
//	{
//		for (int j = 0; j < image.cols; j++)
//		{
//			Histogram[data[i*image.step + j]]++;
//			Histogram1[data[i*image.step + j]]++;
//		}
//	}
//
//	//***********画出图像直方图*******************
//	cv::Mat image1(255, 255, CV_8UC3);
//	for (int i = 0; i < 255; i++)
//	{
//		Histogram1[i] = Histogram1[i] % 200;
//		cv::line(image1, cv::Point(i, 235), cv::Point(i, 235 - Histogram1[i]), cv::Scalar(255, 0, 0), 1, 8, 0);
//		if (i % 50 == 0)
//		{
//			char ch[255];
//			sprintf(ch, "%d", i);
//			string str = ch;
//			putText(image1, str, cv::Point(i, 250), 1, 1, cv::Scalar(0, 0, 255));
//		}
//	}
//
//	//遍历每个灰度值，找到最佳分割值T
//	for (int i = 0; i < 255; i++)
//	{
//		//每次遍历之前初始化各变量
//		w1 = 0;		u1 = 0;		w0 = 0;		u0 = 0;
//		//背景各分量值计算**************************
//		for (int j = 0; j <= i; j++){ 
//			w1 += Histogram[j];  //背景部分像素点总数
//			u1 += j * Histogram[j]; //背景部分像素总灰度和
//		}
//		//if (w1 == 0) //背景部分像素点数为0时退出
//		//{
//		//	break;
//		//}
//		u1 = u1 / w1; //背景像素平均灰度
//		w1 = w1 / totalNum; // 背景部分像素点数所占比例
//
//		//前景各分量值计算**************************
//		for (int k = i + 1; k < 255; k++){
//			w0 += Histogram[k];  //前景部分像素点总数
//			u0 += k * Histogram[k]; //前景部分像素总灰度和
//		}
//		//if (w0 == 0) //前景部分像素点数为0时退出
//		//{
//		//	break;
//		//}
//		u0 = u0 / w0; //前景像素平均灰度
//		w0 = w0 / totalNum; // 前景部分像素点数所占比例
//
//		//类间方差计算******************************
//		double varValueI = w0 * w1*(u1 - u0)*(u1 - u0); //当前类间方差计算
//		if (varValue < varValueI){
//			varValue = varValueI;
//			T = i;
//		}
//	}
//	//画出以T为阈值的分割线
//	cv::line(image1, cv::Point(T, 235), cv::Point(T, 0), cv::Scalar(0, 0, 255), 2, 8);
//	cv::imshow("直方图", image1);
//	return T;
//}
//
//
//
//
//int main(int argc, char *argv[])
//{
//	string img_path = "./data01/1280/1_left_4.jpg";
//	//cv::Mat image = cv::imread("./data02/rect_1280/left_1.jpg");
//	cv::Mat imageSrc = cv::imread(img_path);
//	cv::imshow("Soure Image", imageSrc);
//
//	cv::Mat imageGray;
//	cv::cvtColor(imageSrc, imageGray, CV_RGB2GRAY);
//	//Otsu算法
//	cv::Mat imageOutput;
//	int thresholdValue = OtsuAlgThreshold(imageGray);
//	cout << "类间方差为： " << thresholdValue << endl;
//	cv::threshold(imageGray, imageOutput, thresholdValue, 255, CV_THRESH_BINARY);
//	cv::imshow("Output Image", imageOutput);
//
//	//Opencv自带的Otsu算法
//	cv::Mat imageOtsu;
//	cv::threshold(imageGray, imageOtsu, 0, 255, CV_THRESH_OTSU);
//	cv::imshow("Opencv Otsu", imageOtsu);
//
//	//转换为HSV
//	cv::Mat imageHSV;
//	cv::cvtColor(imageSrc, imageHSV, CV_BGR2HSV);
//	cv::imshow("HSV Image", imageHSV);
//	string img_name = img_path + ".HSV.png";
//	cv::imwrite(img_name, imageHSV);
//
//	cv::waitKey();
//	return 0;
//}
