///* 
//* Author��		Yangbo Chen
//* Describe:		��ֵ�ָ�
//* Update date��	2021.06.08
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
////******Otsu�㷨ͨ������䷽���ֵ������Ӧ��ֵ******
//int OtsuAlgThreshold(const cv::Mat image)
//{
//	if (image.channels() != 1)
//	{
//		cout << "Please input Gray-image!" << endl;
//		return 0;
//	}
//	int T = 0; //Otsu�㷨��ֵ
//	double varValue = 0; //��䷽���м�ֵ����
//	double w0 = 0; //ǰ�����ص�����ռ����
//	double w1 = 0; //�������ص�����ռ����
//	double u0 = 0; //ǰ��ƽ���Ҷ�
//	double u1 = 0; //����ƽ���Ҷ�
//	double Histogram[256] = { 0 }; //�Ҷ�ֱ��ͼ���±��ǻҶ�ֵ�����������ǻҶ�ֵ��Ӧ�����ص�����
//	int Histogram1[256] = { 0 };
//	uchar *data = image.data;
//	double totalNum = image.rows*image.cols; //��������
//	//����Ҷ�ֱ��ͼ�ֲ���Histogram�����±��ǻҶ�ֵ�����������ǻҶ�ֵ��Ӧ���ص���
//	for (int i = 0; i < image.rows; i++)   //Ϊ������������û�а�rows��cols���������
//	{
//		for (int j = 0; j < image.cols; j++)
//		{
//			Histogram[data[i*image.step + j]]++;
//			Histogram1[data[i*image.step + j]]++;
//		}
//	}
//
//	//***********����ͼ��ֱ��ͼ*******************
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
//	//����ÿ���Ҷ�ֵ���ҵ���ѷָ�ֵT
//	for (int i = 0; i < 255; i++)
//	{
//		//ÿ�α���֮ǰ��ʼ��������
//		w1 = 0;		u1 = 0;		w0 = 0;		u0 = 0;
//		//����������ֵ����**************************
//		for (int j = 0; j <= i; j++){ 
//			w1 += Histogram[j];  //�����������ص�����
//			u1 += j * Histogram[j]; //�������������ܻҶȺ�
//		}
//		//if (w1 == 0) //�����������ص���Ϊ0ʱ�˳�
//		//{
//		//	break;
//		//}
//		u1 = u1 / w1; //��������ƽ���Ҷ�
//		w1 = w1 / totalNum; // �����������ص�����ռ����
//
//		//ǰ��������ֵ����**************************
//		for (int k = i + 1; k < 255; k++){
//			w0 += Histogram[k];  //ǰ���������ص�����
//			u0 += k * Histogram[k]; //ǰ�����������ܻҶȺ�
//		}
//		//if (w0 == 0) //ǰ���������ص���Ϊ0ʱ�˳�
//		//{
//		//	break;
//		//}
//		u0 = u0 / w0; //ǰ������ƽ���Ҷ�
//		w0 = w0 / totalNum; // ǰ���������ص�����ռ����
//
//		//��䷽�����******************************
//		double varValueI = w0 * w1*(u1 - u0)*(u1 - u0); //��ǰ��䷽�����
//		if (varValue < varValueI){
//			varValue = varValueI;
//			T = i;
//		}
//	}
//	//������TΪ��ֵ�ķָ���
//	cv::line(image1, cv::Point(T, 235), cv::Point(T, 0), cv::Scalar(0, 0, 255), 2, 8);
//	cv::imshow("ֱ��ͼ", image1);
//	return T;
//}
//
//
//int main(int argc, char *argv[])
//{
//	//cv::Mat image = cv::imread("./data02/rect_1280/left_1.jpg");
//	cv::Mat image = cv::imread("./data01/1280/1_left_4.jpg");
//	cv::imshow("SoureImage", image);
//	cv::cvtColor(image, image, CV_RGB2GRAY);
//
//	cv::Mat imageOutput;
//	int thresholdValue = OtsuAlgThreshold(image);
//	cout << "��䷽��Ϊ�� " << thresholdValue << endl;
//	cv::threshold(image, imageOutput, thresholdValue, 255, CV_THRESH_BINARY);
//	
//	//Opencv�Դ���Otsu�㷨
//	cv::Mat imageOtsu;
//	cv::threshold(image, imageOtsu, 0, 255, CV_THRESH_OTSU); 
//
//	//imshow("SoureImage",image);
//	cv::imshow("Output Image", imageOutput);
//	cv::imshow("Opencv Otsu", imageOtsu);
//	cv::waitKey();
//	return 0;
//}
