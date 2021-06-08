///* 
//* Author：		Yangbo Chen
//* Describe:		视差图转换为深度图
//* Update date：	2021.06.08
//*/
//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//using namespace std;
//using namespace cv;
//
//
///*打印像素点*/
//void printImg(cv::Mat img, int printRows, int printCols)
//{
//	/*
//	从中心点开始，向两边延伸
//		printRows:所需打印的行数（中间点到起始打印点的距离）
//		printCols:同上
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
//函数作用：视差图转深度图
//输入：　　
//	dispMap ----视差图，8位单通道，CV_8UC1　　
//	K       ----内参矩阵，float类型
//输出：　　
//	depthMap ----深度图，16位无符号单通道，CV_16UC1
//
//计算公式：
//	depth = (f * baseline) / disp
//	其中，f表示归一化的焦距，也就是内参中的fx； baseline是两个相机光心之间的距离，称作基线距离
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
//	float baseline = 178.089;	//基线距离65mm
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
//					continue;  //防止0除
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
//	/* 参数设置 */
//	//左相机参数
//	//fx 0 cx
//	//0 fy cy
//	//0 0  1
//	Mat cameraMatrixL = (Mat_<float>(3, 3) << 1426.379, 0, 712.043,
//		0, 1426.379, 476.526,
//		0, 0, 1);
//
//
//	/* 深度图计算（视差图转换得到） */
//	Mat imgDisparity = imread("./Piano/mask0nocc.png", CV_8UC1);
//	imshow("disp Image", imgDisparity);
//	printImg(imgDisparity, 10, 10);
//
//	Mat imgDepth(imgDisparity.rows, imgDisparity.cols, CV_16UC1);	//深度图
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