///* 
//* Author��		Yangbo Chen
//* Describe:		�Ӳ�ͼת��Ϊ���ͼ
//* Update date��	2021.06.08
//*/
//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//using namespace std;
//using namespace cv;
//
//
///*��ӡ���ص�*/
//void printImg(cv::Mat img, int printRows, int printCols)
//{
//	/*
//	�����ĵ㿪ʼ������������
//		printRows:�����ӡ���������м�㵽��ʼ��ӡ��ľ��룩
//		printCols:ͬ��
//	*/
//	int row_begin = img.rows / 2 - printRows / 2;
//	int row_end = img.rows / 2 + printRows / 2;
//	int col_begin = img.cols / 2 - printCols / 2;
//	int col_end = img.cols / 2 + printCols / 2;
//
//	for (int i = row_begin; i < row_end; ++i) {
//		for (int j = col_begin; j < col_end; ++j) {
//			int id = i * img.cols + j;
//			cout << (float)img.data[id] << " ";
//			//cout << (float)img.at<uchar>(i, j) << " ";
//		}
//		cout << endl;
//	}
//
//}
//
//
///*
//�������ã��Ӳ�ͼת���ͼ
//���룺����
//	dispMap ----�Ӳ�ͼ��8λ��ͨ����CV_8UC1����
//	K       ----�ڲξ���float����
//���������
//	depthMap ----���ͼ��16λ�޷��ŵ�ͨ����CV_16UC1
//
//���㹫ʽ��
//	depth = (f * baseline) / disp
//	���У�f��ʾ��һ���Ľ��࣬Ҳ�����ڲ��е�fx�� baseline�������������֮��ľ��룬�������߾���
//*/
//void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, cv::Mat K)
//{
//	
//	int type = dispMap.type();
//
//	float fx = K.at<float>(0, 0);
//	float fy = K.at<float>(1, 1);
//	float cx = K.at<float>(0, 2);
//	float cy = K.at<float>(1, 2);
//	float baseline = 178.089;	//���߾���65mm
//
//	if (type == CV_8U)
//	{
//		const float PI = 3.14159265358;
//		int height = dispMap.rows;
//		int width = dispMap.cols;
//
//		uchar* dispData = (uchar*)dispMap.data;
//		ushort* depthData = (ushort*)depthMap.data;
//		for (int i = 0; i < height; i++)
//		{
//			for (int j = 0; j < width; j++)
//			{
//				int id = i * width + j;
//				if (!dispData[id])
//					continue;  //��ֹ0��
//				depthData[id] = ushort((float)fx *baseline / ((float)dispData[id]));
//			}
//		}
//	}
//	else
//	{
//		cout << "please confirm dispImg's type!" << endl;
//		cv::waitKey(0);
//	}
//}
//
//
//
//int main()
//{
//	/* �������� */
//	//���������
//	//fx 0 cx
//	//0 fy cy
//	//0 0  1
//	Mat cameraMatrixL = (Mat_<float>(3, 3) << 1426.379, 0, 712.043,
//		0, 1426.379, 476.526,
//		0, 0, 1);
//
//
//	/* ���ͼ���㣨�Ӳ�ͼת���õ��� */
//	Mat imgDisparity = imread("./Piano/mask0nocc.png", CV_8UC1);
//	imshow("disp Image", imgDisparity);
//	printImg(imgDisparity, 10, 10);
//
//	Mat imgDepth(imgDisparity.rows, imgDisparity.cols, CV_16UC1);	//���ͼ
//	disp2Depth(imgDisparity, imgDepth, cameraMatrixL);
//	imshow("Depth Image", imgDepth);
//
//	/*Mat img(imgDepth.rows, imgDepth.cols, CV_8UC3);
//	float2color(imgDepth, img, --);*/
//
//	waitKey(0);
//}
//
//
///*
//// translate value x in [0..1] into color triplet using "jet" color map
//// if out of range, use darker colors
//// variation of an idea by http://www.metastine.com/?p=7
//void jet(float x, int& r, int& g, int& b)
//{
//	if (x < 0) x = -0.05;
//	if (x > 1) x = 1.05;
//	x = x / 1.15 + 0.1; // use slightly asymmetric range to avoid darkest shades of blue.
//	r = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .75))))));
//	g = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .5))))));
//	b = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .25))))));
//}
//
//
//// get min and max (non-INF) values
//void getMinMax(cv::Mat fimg, float& vmin, float& vmax)
//{
//	CShape sh = fimg.Shape();
//	int width = sh.width, height = sh.height;
//
//	vmin = INFINITY;
//	vmax = -INFINITY;
//
//	for (int y = 0; y < height; y++) {
//		for (int x = 0; x < width; x++) {
//			float f = fimg.Pixel(x, y, 0);
//			if (f == INFINITY)
//				continue;
//			vmin = min(f, vmin);
//			vmax = max(f, vmax);
//		}
//	}
//}
//
//
//// convert float disparity image into a color image using jet colormap
//void float2color(cv::Mat fimg, cv::Mat &img, float dmin, float dmax)
//{
//	CShape sh = fimg.Shape();
//	int width = sh.width, height = sh.height;
//	sh.nBands = 3;
//	img.ReAllocate(sh);
//
//	float scale = 1.0 / (dmax - dmin);
//
//	for (int y = 0; y < height; y++) {
//		for (int x = 0; x < width; x++) {
//			float f = fimg.Pixel(x, y, 0);
//			int r = 0;
//			int g = 0;
//			int b = 0;
//
//			if (f != INFINITY) {
//				float val = scale * (f - dmin);
//				jet(val, r, g, b);
//			}
//
//			img.Pixel(x, y, 0) = b;
//			img.Pixel(x, y, 1) = g;
//			img.Pixel(x, y, 2) = r;
//		}
//	}
//}
//*/