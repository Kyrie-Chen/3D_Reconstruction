#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
//using namespace cv;


//Canny��Ե���
int edge_detect()
{
	cv::Mat src = cv::imread("./data/640/5_left_0.jpg");
	if (!src.data) {
		cout << "Can not load the image!" << endl;
		return -1;
	}
	cv::namedWindow("ԭͼ");
	cv::imshow("ԭͼ", src);

	//����һ��ֱ�ӱ�Ե���
	cv::Mat out;
	Canny(src, out, 150, 100);
	cv::namedWindow("����һ��Ч��ͼ");
	cv::imshow("����һ��Ч��ͼ", out);

	//���������ȻҶ�+ģ�������ٱ�Ե���
	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);		//��ԭͼ��תΪ�Ҷ�
	blur(gray, gray, cv::Size(3, 3));	//�˲�(����)
	cv::namedWindow("ģ��ͼ��");
	cv::imshow("ģ��ͼ��", gray);
	//canny
	cv::Mat out2;
	Canny(gray, out2, 15, 10);
	cv::namedWindow("��������Ч��ͼ");
	cv::imshow("��������Ч��ͼ", out2);

	cv::waitKey();
	return 0;
}


int main(int argc, char *argv[])
{
	cv::Mat imageSource = cv::imread("./data/640/5_left_0.jpg", 0);
	cv::imshow("Source Image", imageSource);

	//��Ե��⣨�ο����棩
	cv::Mat image;
	cv::GaussianBlur(imageSource, image, cv::Size(3, 3), 0);		//��˹ģ��
	cv::Canny(imageSource, image, 150, 100);	//Canny��Ե���
	cv::imshow("Canny Image", image);

	//������ȡ����ȡͼ�����еı߽�������������
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

