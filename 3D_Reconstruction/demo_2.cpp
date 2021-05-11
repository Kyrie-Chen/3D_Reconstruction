//#include <iostream>
//#include <opencv2\opencv.hpp>
//
//using namespace std;
//using namespace cv;
//
//void calDispWithBM(Mat imgL, Mat imgR, Mat &imgDisparity8U)
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
//	//==Ԥ�����˲�����
//	/*
//	Ԥ�����˲��������ͣ���Ҫ�����ڽ�������ʧ�棨photometric distortions����������������ǿ�����,
//	�����ֿ�ѡ���ͣ�CV_STEREO_BM_NORMALIZED_RESPONSE����һ����Ӧ�� ���� CV_STEREO_BM_XSOBEL��ˮƽ����Sobel���ӣ�Ĭ�����ͣ�,
//	�ò���Ϊ int ��;
//	*/
//	bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
//
//	/*Ԥ�����˲������ڴ�С������Χ��[5,255]��һ��Ӧ���� 5x5..21x21 ֮�䣬��������Ϊ����ֵ, int ��*/
//	bm->setPreFilterSize(9);
//
//	/*Ԥ�����˲����Ľض�ֵ��Ԥ��������ֵ������[-preFilterCap, preFilterCap]��Χ�ڵ�ֵ��������Χ��1 - 31,int ��*/
//	bm->setPreFilterCap(31);
//
//	//==SAD ����
//	/*SAD���ڴ�С������Χ��[5,255]��һ��Ӧ���� 5x5 �� 21x21 ֮�䣬����������������int ��*/
//	bm->setBlockSize(9);
//
//	/*��С�ӲĬ��ֵΪ 0, �����Ǹ�ֵ��int ��*/
//	bm->setMinDisparity(0);
//
//	/*�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��, ���ڴ�С������ 16 ����������int ��*/
//	bm->setNumDisparities(numberOfDisparities);
//
//	//==�������
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
//
//	/*����Ӳ���ͨ����仯�ȵĴ��ڴ�С, ֵΪ 0 ʱȡ�� speckle ��飬int ��*/
//	bm->setSpeckleWindowSize(100);
//
//	/*�Ӳ�仯��ֵ�����������Ӳ�仯������ֵʱ���ô����ڵ��Ӳ����㣬int ��*/
//	bm->setSpeckleRange(32);
//
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
//
//	cout << minVal << "\t" << maxVal << endl;
//
//	//--Display it as a CV_8UC1 image��16λ�з���תΪ8λ�޷���
//	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));
//
//	imshow("disparity", imgDisparity8U);
//
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
//	Mat imgDisparity8U = Mat(imgL.rows, imgL.cols, CV_8UC1);
//
//	calDispWithBM(imgL, imgR, imgDisparity8U);
//	waitKey(0);
//
//}