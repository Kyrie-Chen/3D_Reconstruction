#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <utility>
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
	int numberOfDisparities = 128 * 2;

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


//* 通过HSI颜色空间进行分割 */
int HSISegmentation(cv::Mat img_hsi, cv::Mat& mask_hsi,
	std::pair<int, int> low_H, std::pair<int, int> up_H, std::pair<int, int> S, std::pair<int, int> V)
{
	if (img_hsi.empty()) {
		cout << "img_hsi is NULL" << endl;
		return -1;
	}
	if (img_hsi.channels() != 3) {
		cout << "img_hsi.channels != 3" << endl;
		return -1;
	}

	//通道分割
	vector<cv::Mat> img_split;
	cv::split(img_hsi, img_split);

	//阈值分割（三个通道分别进行）
	cv::Mat mask_H, mask_S, mask_V;
	cv::Mat mask_H_low, mask_H_up;

	cv::Mat img_H = img_split[0];
	cv::imshow("img_H", img_H);
	//savePoint1Channel("img_H.csv", img_H);
	cv::inRange(img_H, low_H.first, low_H.second, mask_H_low);		//下边界范围
	cv::inRange(img_H, up_H.first, up_H.second, mask_H_up);			//上边界范围
	cv::bitwise_or(mask_H_low, mask_H_up, mask_H);		//合并（或操作）
	//自适应阈值分割
	//cv::Mat imageOtsu;
	//cv::threshold(img_H, mask_H, 0, 255, CV_THRESH_OTSU);
	//cv::bitwise_not(mask_H, mask_H);
	//cv::imshow("mask_H", mask_H);

	cv::Mat img_S = img_split[1];
	cv::imshow("img_S", img_S);
	//savePoint1Channel("img_S.csv", img_S);
	//自适应阈值分割
	//cv::threshold(img_S, mask_S, 0, 255, CV_THRESH_OTSU);
	//cv::bitwise_not(mask_S, mask_S);
	cv::inRange(img_S, S.first, S.second, mask_S);
	cv::imshow("mask_S", mask_S);

	cv::Mat img_V = img_split[2];
	cv::imshow("img_V", img_V);
	//savePoint1Channel("img_V.csv", img_V);
	cv::inRange(img_V, V.first, V.second, mask_V);
	//cv::threshold(img_S, mask1, S_min, 255, cv::THRESH_BINARY);		//大于S_min的留白
	//cv::threshold(img_S, mask2, S_max, 255, cv::THRESH_BINARY_INV);	//小于S_max的留白
	//cv::multiply(mask1, mask2, mask_S);		//矩阵对应元素相乘（合并结果）
	//cv::imshow("mask_V", mask_V);

	//掩码合并
	cv::bitwise_and(mask_H, mask_S, mask_hsi);
	cv::bitwise_and(mask_hsi, mask_V, mask_hsi);
	cv::imshow("mask_hsi", mask_hsi);

	//cv::inRange(img_hsi, cv::Scalar(H_min, S_min, V_min), cv::Scalar(H_max, S_max, V_max), mask_hsi);	//直接检测
	return 1;
}



int main()
{	
	std::string path_left = "./data02/rect_640/left_4.jpg";
	std::string path_right = "./data02/rect_640/right_4.jpg";
	Mat imgL_src = imread(path_left);		//1_left_4
	Mat imgR_src = imread(path_right);		//1_right_4
	imshow("imgL", imgL_src);
	imshow("imgR", imgR_src);
	//cv::waitKey();

	string mode = "HSV_00";

	cv::Mat imgL_res, imgR_res;		//分割后图像
	//HSV分割
	if (mode == "HSV") {
		/*HSV分割*/
		//转换为HSV
		cv::Mat imgL_hsi, imgR_hsi;
		cv::cvtColor(imgL_src, imgL_hsi, CV_BGR2HSV);
		cv::cvtColor(imgR_src, imgR_hsi, CV_BGR2HSV);
		//HSV掩码获取
		cv::Mat maskL_hsi, maskR_hsi;
		std::pair<int, int> low_H(0, 50), up_H(160, 180), S(0, 255), V(127, 255);
		HSISegmentation(imgL_hsi, maskL_hsi, low_H, up_H, S, V);
		HSISegmentation(imgR_hsi, maskR_hsi, low_H, up_H, S, V);
		//形态学处理
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::morphologyEx(maskL_hsi, maskL_hsi, cv::MORPH_CLOSE, kernel);		//闭运算，填充前景黑点
		cv::morphologyEx(maskL_hsi, maskL_hsi, cv::MORPH_OPEN, kernel);		//开运算，填充背景白点（去噪）
		cv::morphologyEx(maskL_hsi, maskL_hsi, cv::MORPH_DILATE, kernel);		//膨胀，增加白色区域

		cv::morphologyEx(maskR_hsi, maskR_hsi, cv::MORPH_CLOSE, kernel);		//闭运算，填充前景黑点
		cv::morphologyEx(maskR_hsi, maskR_hsi, cv::MORPH_OPEN, kernel);		//开运算，填充背景白点（去噪）
		cv::morphologyEx(maskR_hsi, maskR_hsi, cv::MORPH_DILATE, kernel);		//膨胀，增加白色区域
		//掩码操作
		//cv::Mat imgL_res, imgR_res;
		imgL_src.copyTo(imgL_res, maskL_hsi);
		imgR_src.copyTo(imgR_res, maskR_hsi);
		cv::imshow("Left Image Res", imgL_res);
		cv::imshow("Right Image Res", imgR_res);
	}
	//不作处理
	else {
		imgL_res = imgL_src;
		imgR_res = imgR_src;
	}	

	/* 视差图计算（由极线校正后的左右图像计算得到） */
	//转换为灰度图 —— ！修改视差图计算的对象！
	cv::Mat imgL, imgR;
	cv::cvtColor(imgL_res, imgL, CV_BGR2GRAY);
	cv::cvtColor(imgR_res, imgR, CV_BGR2GRAY);
	// And create the image in which we will save our disparities
	//---BM---
	Mat imgDisparityBM = Mat(imgL.rows, imgL.cols, CV_8UC1);	//视差图
	calDispWithBM(imgL, imgR, imgDisparityBM);
	imshow("disparity_BM", imgDisparityBM);
	//转换为彩色图
	Mat imgColorBM(imgDisparityBM.size(), CV_8UC3);
	GenerateFalseMap(imgDisparityBM, imgColorBM);	//转成彩图
	imshow("disparityBM_color", imgColorBM);
	//保存图片
	std::string disp_map_path_BM, disp_color_map_path_BM;
	if (mode == "HSV") {
		disp_map_path_BM = path_left + ".BM.HSV.d.jpg";		//.HSV
		disp_color_map_path_BM = path_left + ".BM.HSV.c.jpg";		//.HSV
	}
	else {
		disp_map_path_BM = path_left + ".BM.d.jpg";	
		disp_color_map_path_BM = path_left + ".BM.c.jpg";
	}
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