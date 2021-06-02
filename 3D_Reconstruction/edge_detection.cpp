#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
//using namespace cv;


//Canny边缘检测
int edge_detect()
{
	cv::Mat src = cv::imread("./data/640/5_left_0.jpg");
	if (!src.data) {
		cout << "Can not load the image!" << endl;
		return -1;
	}
	cv::namedWindow("原图");
	cv::imshow("原图", src);

	//方法一：直接边缘检测
	cv::Mat out;
	Canny(src, out, 150, 100);
	cv::namedWindow("方法一的效果图");
	cv::imshow("方法一的效果图", out);

	//方法二：先灰度+模糊处理，再边缘检测
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);		//将原图像转为灰度
	blur(gray, gray, cv::Size(3, 3));	//滤波(降噪)
	cv::namedWindow("模糊图像");
	cv::imshow("模糊图像", gray);
	//canny
	cv::Mat out2;
	Canny(gray, out2, 15, 10);
	cv::namedWindow("方法二的效果图");
	cv::imshow("方法二的效果图", out2);

	cv::waitKey();
	return 0;
}


int main(int argc, char *argv[])
{
	cv::Mat imageSource = cv::imread("./data/640/5_left_0.jpg", 0);
	cv::imshow("Source Image", imageSource);

	//边缘检测（参考上面）
	cv::Mat image;
	cv::GaussianBlur(imageSource, image, cv::Size(3, 3), 0);		//高斯模糊
	cv::Canny(imageSource, image, 150, 100);	//Canny边缘检测
	cv::imshow("Canny Image", image);

	//轮廓提取：获取图像所有的边界连续像素序列
	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	findContours(image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	cv::Mat imageContours = cv::Mat::zeros(image.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); i++)
	{
		cv::drawContours(imageContours, contours, i, cv::Scalar(255), 1, 8, hierarchy);
	}
	cv::imshow("Contours Image", imageContours);

	cv::waitKey(0);
	return 0;
}

