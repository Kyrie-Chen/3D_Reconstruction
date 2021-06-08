#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


/* 参数设置 */
//左相机参数
//fx 0 cx
//0 fy cy
//0 0  1
Mat cameraMatrixL = (Mat_<float>(3, 3) << 1426.379, 0, 712.043,
	0, 1426.379, 476.526,
	0, 0, 1);


/*给深度图上色*/
void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)
{
	// color map  
	float max_val = 255.0f;
	float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
	{ 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
	float sum = 0;
	for (int i = 0; i < 8; i++)
		sum += map[i][3];

	float weights[8]; // relative   weights  
	float cumsum[8];  // cumulative weights  
	cumsum[0] = 0;
	for (int i = 0; i < 7; i++) {
		weights[i] = sum / map[i][3];
		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
	}

	int height_ = src.rows;
	int width_ = src.cols;
	// for all pixels do  
	for (int v = 0; v < height_; v++) {
		for (int u = 0; u < width_; u++) {

			// get normalized value  
			float val = std::min(std::max(src.data[v*width_ + u] / max_val, 0.0f), 1.0f);

			// find bin  
			int i;
			for (i = 0; i < 7; i++)
				if (val < cumsum[i + 1])
					break;

			// compute red/green/blue values  
			float   w = 1.0 - (val - cumsum[i])*weights[i];
			uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
			uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
			uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
			//rgb内存连续存放  
			disp.data[v*width_ * 3 + 3 * u + 0] = b;
			disp.data[v*width_ * 3 + 3 * u + 1] = g;
			disp.data[v*width_ * 3 + 3 * u + 2] = r;
		}
	}
}

/*保存三维坐标*/
static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 16.0e4;
	FILE* fp = fopen(filename, "wt");
	printf("%d %d \n", mat.rows, mat.cols);
	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			int id = y * mat.cols + x;
			fprintf(fp, "%f %f %f\n", x, y, (float)mat.data[id]);

			//Vec3f point = mat.at<Vec3f>(y, x);
			//if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) 
			//	continue;
			//fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

/*打印像素点*/
void printImg(cv::Mat img, int printRows, int printCols)
{
	/*
	从中心点开始，向两边延伸
		printRows:所需打印的行数（中间点到起始打印点的距离）
		printCols:同上
	*/
	int row_begin = img.rows / 2 - printRows / 2;
	int row_end = img.rows / 2 + printRows / 2;
	int col_begin = img.cols / 2 - printCols / 2;
	int col_end = img.cols / 2 + printCols / 2;

	for (int i = row_begin; i < row_end; ++i) {
		for (int j = col_begin; j < col_end; ++j) {
			int id = i * img.cols + j;
			cout << (float)img.data[id] << " ";
			//cout << (float)img.at<uchar>(i, j) << " ";
		}
		cout << endl;
	}

}

/*----------------------------------------------------------------------------*/

/* 立体匹配：StereoBM算法 */
void calDispWithBM(Mat imgL, Mat imgR, Mat &imgDisparity8U)
{
	Mat imgDisparity16S = Mat(imgL.rows, imgL.cols, CV_16S);
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);

	//--Call the constructor for StereoBM
	cv::Size imgSize = imgL.size();
	//int numberOfDisparities = ((imgSize.width / 8) + 15) & -16;
	int numberOfDisparities = 320;

	//--Calculate the disparity image
	/*
	左右视图的有效像素区域，一般由双目校正阶段的 cvStereoRectify 函数传递，也可以自行设定。
	一旦在状态参数中设定了 roi1 和 roi2，OpenCV 会通过cvGetValidDisparityROI 函数计算出视差图的有效区域，在有效区域外的视差值将被清零。
	*/
	cv::Rect roi1, roi2;
	bm->setROI1(roi1);
	bm->setROI2(roi2);

	//---预处理滤波参数---
	/*
	预处理滤波器的类型，主要是用于降低亮度失真（photometric distortions）、消除噪声和增强纹理等,
	有两种可选类型：CV_STEREO_BM_NORMALIZED_RESPONSE（归一化响应） 或者 CV_STEREO_BM_XSOBEL（水平方向Sobel算子，默认类型）,
	该参数为 int 型;
	*/
	bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
	bm->setPreFilterSize(9);	//预处理滤波器窗口大小，容许范围是[5,255]，一般应该在5x5 -- 21x21之间，必须为奇数值
	bm->setPreFilterCap(31);	//预处理滤波器的截断值，预处理的输出值仅保留[-preFilterCap, preFilterCap]范围内的值，参数范围：1-31

	//---SAD 参数---
	bm->setBlockSize(9);	//SAD窗口大小，容许范围是[5,255]，一般应该在 5x5 至 21x21 之间，参数必须是奇数
	bm->setMinDisparity(0);		//最小视差，默认值为 0, 可以是负值
	bm->setNumDisparities(numberOfDisparities);		//视差窗口，即最大视差值与最小视差值之差, 窗口大小必须是 16 的整数倍，int 型

	//---后处理参数---
	/*
	低纹理区域的判断阈值:
	如果当前SAD窗口内所有邻居像素点的x导数绝对值之和小于指定阈值，则该窗口对应的像素点的视差值为 0
	（That is, if the sum of absolute values of x-derivatives computed over SADWindowSize by SADWindowSize
	pixel neighborhood is smaller than the parameter, no disparity is computed at the pixel），
	该参数不能为负值，int 型;
	*/
	bm->setTextureThreshold(20);
	/*
	视差唯一性百分比:
	视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，
	否则该像素点的视差为 0 （the minimum margin in percents between the best (minimum) cost function value and the second best value to accept
	the computed disparity, that is, accept the computed disparity d^ only if SAD(d) >= SAD(d^) x (1 + uniquenessRatio/100.) for any d != d*+/-1 within the search range ），
	该参数不能为负值，一般5-15左右的值比较合适，int 型*/
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);	//检查视差连通区域变化度的窗口大小, 值为 0 时取消 speckle 检查
	bm->setSpeckleRange(32);	//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
	/*	左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异；超过该阈值的视差值将被清零。
	该参数默认为 -1，即不执行左右视差检查。	注意在程序调试阶段最好保持该值为 -1，以便查看不同视差窗口生成的视差效果。	*/
	bm->setDisp12MaxDiff(1);

	/*计算视差*/
	bm->compute(imgL, imgR, imgDisparity16S);

	//-- Check its extreme values
	double minVal; double maxVal;
	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
	cout << minVal << "\t" << maxVal << endl;

	//--Display it as a CV_8UC1 image：16位有符号转为8位无符号
	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));

	//计算深度值
	//Mat xyz;
	//reprojectImageTo3D(imgDisparity16S, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	//xyz = xyz * 16;
	//saveXYZ("xyz.xls", xyz);


	////转换为彩色图
	//Mat imgColor(imgDisparity16S.size(), CV_8UC3);
	//GenerateFalseMap(imgDisparity8U, imgColor);//转成彩图
	//imshow("disparityBM_color", imgColor);
}


/* 立体匹配：SGBM算法 */
void calDispWithSGBM(cv::Mat imgL, cv::Mat imgR, cv::Mat &imgDisparity8U)
{
	Mat imgDisparity16S = Mat(imgL.rows, imgL.cols, CV_16S);

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
	sgbm->setPreFilterCap(32);	//预处理滤波器截断值

	cv::Size imgSize = imgL.size();
	//int numberOfDisparities = ((imgSize.width / 8) + 15) & -16;		//视差窗口，即最大视差值与最小视差值之差
	int numberOfDisparities = 320;

	int SADWindowSize = 7;		//SAD窗口大小
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm->setBlockSize(sgbmWinSize);	//！！设置SAD窗口大小
	int cn = imgL.channels();	//图像通道数	
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);		//动态规划参数设置
	sgbm->setP2(64 * cn*sgbmWinSize*sgbmWinSize);

	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);	//！！视差窗口
	sgbm->setUniquenessRatio(25);		//！！用来防止误匹配15
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);	//MODE_SGBM、MODE_HH、MODE_SGBM_3WAY

	sgbm->compute(imgL, imgR, imgDisparity16S);
	//-- Check its extreme values
	double minVal, maxVal;
	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
	cout << minVal << "\t" << maxVal << endl;

	//去黑边
	//Mat img1p, img2p;
	//copyMakeBorder(imgL, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//copyMakeBorder(imgR, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//imgDisparity16S = imgDisparity16S.colRange(numberOfDisparities, img2p.cols - NumDisparities);

	//转换为8位无符号图 Display it as a CV_8UC1 image：16位有符号转为8位无符号
	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));

	//计算深度值
	//reprojectImageTo3D(imgDisparity16S, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	//xyz = xyz * 16;
	//saveXYZ("xyz.xls", xyz);

	//转换为彩色图
	//Mat imgColor(imgDisparity16S.size(), CV_8UC3);
	//GenerateFalseMap(imgDisparity8U, imgColor);//转成彩图
	//imshow("disparitySGBM_color", imgColor);
}


/* 视差图转深度图 */
void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, cv::Mat K)
{
	/*
	函数作用：视差图转深度图
	输入：　　
		dispMap ----视差图，8位单通道，CV_8UC1　　
		K       ----内参矩阵，float类型
	输出：　　
		depthMap ----深度图，16位无符号单通道，CV_16UC1

	计算公式：
		depth = (f * baseline) / disp
		其中，f表示归一化的焦距，也就是内参中的fx； baseline是两个相机光心之间的距离，称作基线距离
	*/
	int type = dispMap.type();

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);
	float baseline = 178.089;	//基线距离65mm

	if (type == CV_8U)
	{
		const float PI = 3.14159265358;
		int height = dispMap.rows;
		int width = dispMap.cols;

		uchar* dispData = (uchar*)dispMap.data;
		ushort* depthData = (ushort*)depthMap.data;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int id = i * width + j;
				if (!dispData[id])  
					continue;  //防止0除
				depthData[id] = ushort((float)fx *baseline / ((float)dispData[id]));
			}
		}
	}
	else
	{
		cout << "please confirm dispImg's type!" << endl;
		cv::waitKey(0);
	}
}



int main()
{	
	std::string path_left = "./data02/rect_1280_light/left_1.jpg";
	std::string path_right = "./data02/rect_1280_light/right_1.jpg";

	/* 视差图计算（由极线校正后的左右图像计算得到） */
	Mat imgL = imread(path_left, 0);		//1_left_4
	Mat imgR = imread(path_right, 0);		//1_right_4
	imshow("imgL", imgL);
	imshow("imgR", imgR);
	// And create the image in which we will save our disparities
	//---BM---
	Mat imgDisparityBM = Mat(imgL.rows, imgL.cols, CV_8UC1);	//视差图
	calDispWithBM(imgL, imgR, imgDisparityBM);
	imshow("disparity_BM", imgDisparityBM);
	//转换为彩色图
	Mat imgColorBM(imgDisparityBM.size(), CV_8UC3);
	GenerateFalseMap(imgDisparityBM, imgColorBM);//转成彩图
	imshow("disparityBM_color", imgColorBM);
	//保存图片
	std::string disp_map_path_BM = path_left + ".BM.d.jpg";
	std::string disp_color_map_path_BM = path_left + ".BM.c.jpg";
	cv::imwrite(disp_map_path_BM, imgDisparityBM);
	cv::imwrite(disp_color_map_path_BM, imgColorBM);

	////---SGBM---
	//Mat imgDisparitySGBM = Mat(imgL.rows, imgL.cols, CV_8UC1);
	//calDispWithSGBM(imgL, imgR, imgDisparitySGBM);
	//imshow("disparity_SGBM", imgDisparitySGBM);
	////转换为彩色图
	//Mat imgColorSGBM(imgDisparitySGBM.size(), CV_8UC3);
	//GenerateFalseMap(imgDisparitySGBM, imgColorSGBM);//转成彩图
	//imshow("disparitySGBM_color", imgColorSGBM);
	////保存图片
	//std::string disp_map_path_SGBM = path_left + ".SGBM.d.jpg";
	//std::string disp_color_map_path_SGBM = path_left + ".SGBM.c.jpg";
	//cv::imwrite(disp_map_path_SGBM, imgDisparitySGBM);
	//cv::imwrite(disp_color_map_path_SGBM, imgColorSGBM);

	/* 深度图计算（视差图转换得到） */
	//方法1：公式计算
	//Mat imgDepth(imgDisparityBM.size(), CV_16UC1);	//深度图
	//disp2Depth(imgDisparityBM, imgDepth, cameraMatrixL);
	//saveXYZ("xyz_1.xlsx", imgDepth);
	//imshow("Depth Image", imgDepth);
	//方法2：库函数
	//Mat xyz;
	//reprojectImageTo3D(imgDisparitySGBM, xyz, cameraMatrixL, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	//xyz = xyz * 16;
	//saveXYZ("xyz_2.xlsx", xyz);

	cout << "OK" << endl;
	waitKey(0);
}