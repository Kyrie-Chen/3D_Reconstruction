///*
//
//立体匹配算法：根据左右两幅图像计算得到视差图，主要包括BM和SGBM两种算法
//
//算法主要调整参数（其余可使用默认值）：
//	UniquenessRatio主要可以防止误匹配，此参数对于最后的匹配结果是有很大的影响。立体匹配中，宁愿区域无法匹配，也不要误匹配。如果有误匹配的话，碰到障碍检测这种应用，就会很麻烦。该参数不能为负值，一般5-15左右的值比较合适，int型。
//	BlockSize：SAD窗口大小，容许范围是[5,255]，一般应该在 5x5..21x21 之间，参数必须为奇数值, int型。
//	NumDisparities：视差窗口，即最大视差值与最小视差值之差,窗口大小必须是 16的整数倍，int型。
//
//*/
//
//
//#include <iostream>
//#include <opencv2\opencv.hpp>
//using namespace std;
//using namespace cv;
//
//
///*给深度图上色*/
//void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)
//{
//	// color map  
//	float max_val = 255.0f;
//	float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
//	{ 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
//	float sum = 0;
//	for (int i = 0; i < 8; i++)
//		sum += map[i][3];
//
//	float weights[8]; // relative   weights  
//	float cumsum[8];  // cumulative weights  
//	cumsum[0] = 0;
//	for (int i = 0; i < 7; i++) {
//		weights[i] = sum / map[i][3];
//		cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
//	}
//
//	int height_ = src.rows;
//	int width_ = src.cols;
//	// for all pixels do  
//	for (int v = 0; v < height_; v++) {
//		for (int u = 0; u < width_; u++) {
//
//			// get normalized value  
//			float val = std::min(std::max(src.data[v*width_ + u] / max_val, 0.0f), 1.0f);
//
//			// find bin  
//			int i;
//			for (i = 0; i < 7; i++)
//				if (val < cumsum[i + 1])
//					break;
//
//			// compute red/green/blue values  
//			float   w = 1.0 - (val - cumsum[i])*weights[i];
//			uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
//			uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
//			uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
//			//rgb内存连续存放  
//			disp.data[v*width_ * 3 + 3 * u + 0] = b;
//			disp.data[v*width_ * 3 + 3 * u + 1] = g;
//			disp.data[v*width_ * 3 + 3 * u + 2] = r;
//		}
//	}
//}
//
//
///*
//立体匹配：StereoBM算法 https://blog.csdn.net/zfjBIT/article/details/91429770
//*/
//void calDispWithBM(cv::Mat imgL, cv::Mat imgR, cv::Mat &imgDisparity8U)
//{
//	Mat imgDisparity16S = Mat(imgL.rows, imgL.cols, CV_16S);
//
//	//--Call the constructor for StereoBM
//	cv::Size imgSize = imgL.size();
//	int numberOfDisparities = ((imgSize.width / 8) + 15) & -16;
//	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
//
//	//--Calculate the disparity image
//
//	/*
//	左右视图的有效像素区域，一般由双目校正阶段的 cvStereoRectify 函数传递，也可以自行设定。
//	一旦在状态参数中设定了 roi1 和 roi2，OpenCV 会通过cvGetValidDisparityROI 函数计算出视差图的有效区域，在有效区域外的视差值将被清零。
//	*/
//	cv::Rect roi1, roi2;
//	bm->setROI1(roi1);
//	bm->setROI2(roi2);
//
//	//---预处理滤波参数---
//	/*
//	预处理滤波器的类型，主要是用于降低亮度失真（photometric distortions）、消除噪声和增强纹理等,
//	有两种可选类型：CV_STEREO_BM_NORMALIZED_RESPONSE（归一化响应） 或者 CV_STEREO_BM_XSOBEL（水平方向Sobel算子，默认类型）,
//	该参数为 int 型;
//	*/
//	bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
//	/*预处理滤波器窗口大小，容许范围是[5,255]，一般应该在 5x5..21x21 之间，参数必须为奇数值, int 型*/
//	bm->setPreFilterSize(9);
//	/*预处理滤波器的截断值，预处理的输出值仅保留[-preFilterCap, preFilterCap]范围内的值，参数范围：1 - 31,int 型*/
//	bm->setPreFilterCap(31);
//
//	//---SAD 参数---
//	/*SAD窗口大小，容许范围是[5,255]，一般应该在 5x5 至 21x21 之间，参数必须是奇数，int 型*/
//	bm->setBlockSize(9);
//	/*最小视差，默认值为 0, 可以是负值，int 型*/
//	bm->setMinDisparity(0);
//	/*视差窗口，即最大视差值与最小视差值之差, 窗口大小必须是 16 的整数倍，int 型*/
//	bm->setNumDisparities(numberOfDisparities);
//
//	//---后处理参数---
//	/*
//	低纹理区域的判断阈值:
//	如果当前SAD窗口内所有邻居像素点的x导数绝对值之和小于指定阈值，则该窗口对应的像素点的视差值为 0
//	（That is, if the sum of absolute values of x-derivatives computed over SADWindowSize by SADWindowSize
//	pixel neighborhood is smaller than the parameter, no disparity is computed at the pixel），
//	该参数不能为负值，int 型;
//	*/
//	bm->setTextureThreshold(10);
//
//	/*
//	视差唯一性百分比:
//	视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，
//	否则该像素点的视差为 0 （the minimum margin in percents between the best (minimum) cost function value and the second best value to accept
//	the computed disparity, that is, accept the computed disparity d^ only if SAD(d) >= SAD(d^) x (1 + uniquenessRatio/100.) for any d != d*+/-1 within the search range ），
//	该参数不能为负值，一般5-15左右的值比较合适，int 型*/
//	bm->setUniquenessRatio(15);
//	/*检查视差连通区域变化度的窗口大小, 值为 0 时取消 speckle 检查，int 型*/
//	bm->setSpeckleWindowSize(100);
//	/*视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零，int 型*/
//	bm->setSpeckleRange(32);
//	/*
//	左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异。
//	超过该阈值的视差值将被清零。该参数默认为 -1，即不执行左右视差检查。int 型。
//	注意在程序调试阶段最好保持该值为 -1，以便查看不同视差窗口生成的视差效果。
//	*/
//	bm->setDisp12MaxDiff(1);
//
//	/*计算视差*/
//	bm->compute(imgL, imgR, imgDisparity16S);
//
//	//-- Check its extreme values
//	double minVal; double maxVal;
//	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
//	cout << minVal << "\t" << maxVal << endl;
//
//	//--Display it as a CV_8UC1 image：16位有符号转为8位无符号
//	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));
//
//}
//
///*
//立体匹配：SGBM算法 https://blog.csdn.net/weixin_39449570/article/details/79033314
//*/
//void calDispWithSGBM(cv::Mat imgL, cv::Mat imgR, cv::Mat &imgDisparity8U)
//{
//	Mat imgDisparity16S = Mat(imgL.rows, imgL.cols, CV_16S);
//
//	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
//	sgbm->setPreFilterCap(32);	//预处理滤波器截断值
//
//	cv::Size imgSize = imgL.size();
//	int numberOfDisparities = ((imgSize.width / 8) + 15) & -16;		//视差窗口，即最大视差值与最小视差值之差
//
//	int SADWindowSize = 9;		//SAD窗口大小
//	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
//	sgbm->setBlockSize(sgbmWinSize);	//！！设置SAD窗口大小
//	int cn = imgL.channels();	//图像通道数	
//	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);		//动态规划参数设置
//	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
//
//	sgbm->setMinDisparity(0);
//	sgbm->setNumDisparities(numberOfDisparities);	//！！视差窗口
//	sgbm->setUniquenessRatio(10);		//！！用来防止误匹配
//	sgbm->setSpeckleWindowSize(100);
//	sgbm->setSpeckleRange(32);
//	sgbm->setDisp12MaxDiff(1);
//	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
//	//int alg = STEREO_SGBM;
//	//if (alg == STEREO_HH)
//	//	sgbm->setMode(cv::StereoSGBM::MODE_HH);
//	//else if (alg == STEREO_SGBM)
//	//	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
//	//else if (alg == STEREO_3WAY)
//	//	sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
//
//	sgbm->compute(imgL, imgR, imgDisparity16S);
//	//-- Check its extreme values
//	double minVal, maxVal;
//	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
//	cout << minVal << "\t" << maxVal << endl;
//
//	//去黑边
//	//Mat img1p, img2p;
//	//copyMakeBorder(imgL, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
//	//copyMakeBorder(imgR, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
//	//imgDisparity16S = imgDisparity16S.colRange(numberOfDisparities, img2p.cols - NumDisparities);
//
//	//转换为8位无符号图 Display it as a CV_8UC1 image：16位有符号转为8位无符号
//	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));
//	imshow("disparity", imgDisparity8U);
//
//	//计算深度值
//	//reprojectImageTo3D(imgDisparity16S, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
//	//xyz = xyz * 16;
//	//saveXYZ("xyz.xls", xyz);
//
//	//转换为彩色图
//	Mat imgColor(imgDisparity16S.size(), CV_8UC3);
//	GenerateFalseMap(imgDisparity8U, imgColor);//转成彩图
//	imshow("disparity_color", imgColor);
//}
//
//
//int main()
//{
//	//--读取图像
//	Mat imgL = imread("./Piano/im0.png", 0);
//	Mat imgR = imread("./Piano/im1.png", 0);
//	imshow("imgL", imgL);
//	imshow("imgR", imgR);
//
//	//--And create the image in which we will save our disparities
//	//BM
//	Mat imgDisparity8U = Mat(imgL.rows, imgL.cols, CV_8UC1);
//	calDispWithBM(imgL, imgR, imgDisparity8U);
//	imshow("disparity_BM", imgDisparity8U);
//
//	//SGBM
//	Mat imgDisparity_2 = Mat(imgL.rows, imgL.cols, CV_8UC1);
//	calDispWithSGBM(imgL, imgR, imgDisparity_2);
//	imshow("disparity_SGBM", imgDisparity_2);
//	cout << "OK" << endl;
//	
//	waitKey(0);
//
//}