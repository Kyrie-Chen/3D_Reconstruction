///*
//
//����ƥ���㷨��������������ͼ�����õ��Ӳ�ͼ����Ҫ����BM��SGBM�����㷨
//
//�㷨��Ҫ���������������ʹ��Ĭ��ֵ����
//	UniquenessRatio��Ҫ���Է�ֹ��ƥ�䣬�˲�����������ƥ�������кܴ��Ӱ�졣����ƥ���У���Ը�����޷�ƥ�䣬Ҳ��Ҫ��ƥ�䡣�������ƥ��Ļ��������ϰ��������Ӧ�ã��ͻ���鷳���ò�������Ϊ��ֵ��һ��5-15���ҵ�ֵ�ȽϺ��ʣ�int�͡�
//	BlockSize��SAD���ڴ�С������Χ��[5,255]��һ��Ӧ���� 5x5..21x21 ֮�䣬��������Ϊ����ֵ, int�͡�
//	NumDisparities���Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��,���ڴ�С������ 16����������int�͡�
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
///*�����ͼ��ɫ*/
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
//			//rgb�ڴ��������  
//			disp.data[v*width_ * 3 + 3 * u + 0] = b;
//			disp.data[v*width_ * 3 + 3 * u + 1] = g;
//			disp.data[v*width_ * 3 + 3 * u + 2] = r;
//		}
//	}
//}
//
//
///*
//����ƥ�䣺StereoBM�㷨 https://blog.csdn.net/zfjBIT/article/details/91429770
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
//	������ͼ����Ч��������һ����˫ĿУ���׶ε� cvStereoRectify �������ݣ�Ҳ���������趨��
//	һ����״̬�������趨�� roi1 �� roi2��OpenCV ��ͨ��cvGetValidDisparityROI ����������Ӳ�ͼ����Ч��������Ч��������Ӳ�ֵ�������㡣
//	*/
//	cv::Rect roi1, roi2;
//	bm->setROI1(roi1);
//	bm->setROI2(roi2);
//
//	//---Ԥ�����˲�����---
//	/*
//	Ԥ�����˲��������ͣ���Ҫ�����ڽ�������ʧ�棨photometric distortions����������������ǿ�����,
//	�����ֿ�ѡ���ͣ�CV_STEREO_BM_NORMALIZED_RESPONSE����һ����Ӧ�� ���� CV_STEREO_BM_XSOBEL��ˮƽ����Sobel���ӣ�Ĭ�����ͣ�,
//	�ò���Ϊ int ��;
//	*/
//	bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
//	/*Ԥ�����˲������ڴ�С������Χ��[5,255]��һ��Ӧ���� 5x5..21x21 ֮�䣬��������Ϊ����ֵ, int ��*/
//	bm->setPreFilterSize(9);
//	/*Ԥ�����˲����Ľض�ֵ��Ԥ��������ֵ������[-preFilterCap, preFilterCap]��Χ�ڵ�ֵ��������Χ��1 - 31,int ��*/
//	bm->setPreFilterCap(31);
//
//	//---SAD ����---
//	/*SAD���ڴ�С������Χ��[5,255]��һ��Ӧ���� 5x5 �� 21x21 ֮�䣬����������������int ��*/
//	bm->setBlockSize(9);
//	/*��С�ӲĬ��ֵΪ 0, �����Ǹ�ֵ��int ��*/
//	bm->setMinDisparity(0);
//	/*�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��, ���ڴ�С������ 16 ����������int ��*/
//	bm->setNumDisparities(numberOfDisparities);
//
//	//---�������---
//	/*
//	������������ж���ֵ:
//	�����ǰSAD�����������ھ����ص��x��������ֵ֮��С��ָ����ֵ����ô��ڶ�Ӧ�����ص���Ӳ�ֵΪ 0
//	��That is, if the sum of absolute values of x-derivatives computed over SADWindowSize by SADWindowSize
//	pixel neighborhood is smaller than the parameter, no disparity is computed at the pixel����
//	�ò�������Ϊ��ֵ��int ��;
//	*/
//	bm->setTextureThreshold(10);
//
//	/*
//	�Ӳ�Ψһ�԰ٷֱ�:
//	�Ӳ�ڷ�Χ����ʹ����Ǵεʹ��۵�(1 + uniquenessRatio/100)��ʱ����ʹ��۶�Ӧ���Ӳ�ֵ���Ǹ����ص���Ӳ
//	��������ص���Ӳ�Ϊ 0 ��the minimum margin in percents between the best (minimum) cost function value and the second best value to accept
//	the computed disparity, that is, accept the computed disparity d^ only if SAD(d) >= SAD(d^) x (1 + uniquenessRatio/100.) for any d != d*+/-1 within the search range ����
//	�ò�������Ϊ��ֵ��һ��5-15���ҵ�ֵ�ȽϺ��ʣ�int ��*/
//	bm->setUniquenessRatio(15);
//	/*����Ӳ���ͨ����仯�ȵĴ��ڴ�С, ֵΪ 0 ʱȡ�� speckle ��飬int ��*/
//	bm->setSpeckleWindowSize(100);
//	/*�Ӳ�仯��ֵ�����������Ӳ�仯������ֵʱ���ô����ڵ��Ӳ����㣬int ��*/
//	bm->setSpeckleRange(32);
//	/*
//	���Ӳ�ͼ��ֱ�Ӽ���ó��������Ӳ�ͼ��ͨ��cvValidateDisparity����ó���֮������������졣
//	��������ֵ���Ӳ�ֵ�������㡣�ò���Ĭ��Ϊ -1������ִ�������Ӳ��顣int �͡�
//	ע���ڳ�����Խ׶���ñ��ָ�ֵΪ -1���Ա�鿴��ͬ�Ӳ�����ɵ��Ӳ�Ч����
//	*/
//	bm->setDisp12MaxDiff(1);
//
//	/*�����Ӳ�*/
//	bm->compute(imgL, imgR, imgDisparity16S);
//
//	//-- Check its extreme values
//	double minVal; double maxVal;
//	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
//	cout << minVal << "\t" << maxVal << endl;
//
//	//--Display it as a CV_8UC1 image��16λ�з���תΪ8λ�޷���
//	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));
//
//}
//
///*
//����ƥ�䣺SGBM�㷨 https://blog.csdn.net/weixin_39449570/article/details/79033314
//*/
//void calDispWithSGBM(cv::Mat imgL, cv::Mat imgR, cv::Mat &imgDisparity8U)
//{
//	Mat imgDisparity16S = Mat(imgL.rows, imgL.cols, CV_16S);
//
//	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
//	sgbm->setPreFilterCap(32);	//Ԥ�����˲����ض�ֵ
//
//	cv::Size imgSize = imgL.size();
//	int numberOfDisparities = ((imgSize.width / 8) + 15) & -16;		//�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��
//
//	int SADWindowSize = 9;		//SAD���ڴ�С
//	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
//	sgbm->setBlockSize(sgbmWinSize);	//��������SAD���ڴ�С
//	int cn = imgL.channels();	//ͼ��ͨ����	
//	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);		//��̬�滮��������
//	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
//
//	sgbm->setMinDisparity(0);
//	sgbm->setNumDisparities(numberOfDisparities);	//�����Ӳ��
//	sgbm->setUniquenessRatio(10);		//����������ֹ��ƥ��
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
//	//ȥ�ڱ�
//	//Mat img1p, img2p;
//	//copyMakeBorder(imgL, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
//	//copyMakeBorder(imgR, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
//	//imgDisparity16S = imgDisparity16S.colRange(numberOfDisparities, img2p.cols - NumDisparities);
//
//	//ת��Ϊ8λ�޷���ͼ Display it as a CV_8UC1 image��16λ�з���תΪ8λ�޷���
//	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));
//	imshow("disparity", imgDisparity8U);
//
//	//�������ֵ
//	//reprojectImageTo3D(imgDisparity16S, xyz, Q, true); //��ʵ�������ʱ��ReprojectTo3D������X / W, Y / W, Z / W��Ҫ����16(Ҳ����W����16)�����ܵõ���ȷ����ά������Ϣ��
//	//xyz = xyz * 16;
//	//saveXYZ("xyz.xls", xyz);
//
//	//ת��Ϊ��ɫͼ
//	Mat imgColor(imgDisparity16S.size(), CV_8UC3);
//	GenerateFalseMap(imgDisparity8U, imgColor);//ת�ɲ�ͼ
//	imshow("disparity_color", imgColor);
//}
//
//
//int main()
//{
//	//--��ȡͼ��
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