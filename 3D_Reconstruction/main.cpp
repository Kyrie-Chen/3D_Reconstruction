#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


/* �������� */
//���������
//fx 0 cx
//0 fy cy
//0 0  1
Mat cameraMatrixL = (Mat_<float>(3, 3) << 1426.379, 0, 712.043,
	0, 1426.379, 476.526,
	0, 0, 1);


/*�����ͼ��ɫ*/
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
			//rgb�ڴ��������  
			disp.data[v*width_ * 3 + 3 * u + 0] = b;
			disp.data[v*width_ * 3 + 3 * u + 1] = g;
			disp.data[v*width_ * 3 + 3 * u + 2] = r;
		}
	}
}

/*������ά����*/
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

/*��ӡ���ص�*/
void printImg(cv::Mat img, int printRows, int printCols)
{
	/*
	�����ĵ㿪ʼ������������
		printRows:�����ӡ���������м�㵽��ʼ��ӡ��ľ��룩
		printCols:ͬ��
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

/* ����ƥ�䣺StereoBM�㷨 */
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
	������ͼ����Ч��������һ����˫ĿУ���׶ε� cvStereoRectify �������ݣ�Ҳ���������趨��
	һ����״̬�������趨�� roi1 �� roi2��OpenCV ��ͨ��cvGetValidDisparityROI ����������Ӳ�ͼ����Ч��������Ч��������Ӳ�ֵ�������㡣
	*/
	cv::Rect roi1, roi2;
	bm->setROI1(roi1);
	bm->setROI2(roi2);

	//---Ԥ�����˲�����---
	/*
	Ԥ�����˲��������ͣ���Ҫ�����ڽ�������ʧ�棨photometric distortions����������������ǿ�����,
	�����ֿ�ѡ���ͣ�CV_STEREO_BM_NORMALIZED_RESPONSE����һ����Ӧ�� ���� CV_STEREO_BM_XSOBEL��ˮƽ����Sobel���ӣ�Ĭ�����ͣ�,
	�ò���Ϊ int ��;
	*/
	bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);
	bm->setPreFilterSize(9);	//Ԥ�����˲������ڴ�С������Χ��[5,255]��һ��Ӧ����5x5 -- 21x21֮�䣬����Ϊ����ֵ
	bm->setPreFilterCap(31);	//Ԥ�����˲����Ľض�ֵ��Ԥ��������ֵ������[-preFilterCap, preFilterCap]��Χ�ڵ�ֵ��������Χ��1-31

	//---SAD ����---
	bm->setBlockSize(9);	//SAD���ڴ�С������Χ��[5,255]��һ��Ӧ���� 5x5 �� 21x21 ֮�䣬��������������
	bm->setMinDisparity(0);		//��С�ӲĬ��ֵΪ 0, �����Ǹ�ֵ
	bm->setNumDisparities(numberOfDisparities);		//�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��, ���ڴ�С������ 16 ����������int ��

	//---�������---
	/*
	������������ж���ֵ:
	�����ǰSAD�����������ھ����ص��x��������ֵ֮��С��ָ����ֵ����ô��ڶ�Ӧ�����ص���Ӳ�ֵΪ 0
	��That is, if the sum of absolute values of x-derivatives computed over SADWindowSize by SADWindowSize
	pixel neighborhood is smaller than the parameter, no disparity is computed at the pixel����
	�ò�������Ϊ��ֵ��int ��;
	*/
	bm->setTextureThreshold(20);
	/*
	�Ӳ�Ψһ�԰ٷֱ�:
	�Ӳ�ڷ�Χ����ʹ����Ǵεʹ��۵�(1 + uniquenessRatio/100)��ʱ����ʹ��۶�Ӧ���Ӳ�ֵ���Ǹ����ص���Ӳ
	��������ص���Ӳ�Ϊ 0 ��the minimum margin in percents between the best (minimum) cost function value and the second best value to accept
	the computed disparity, that is, accept the computed disparity d^ only if SAD(d) >= SAD(d^) x (1 + uniquenessRatio/100.) for any d != d*+/-1 within the search range ����
	�ò�������Ϊ��ֵ��һ��5-15���ҵ�ֵ�ȽϺ��ʣ�int ��*/
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);	//����Ӳ���ͨ����仯�ȵĴ��ڴ�С, ֵΪ 0 ʱȡ�� speckle ���
	bm->setSpeckleRange(32);	//�Ӳ�仯��ֵ�����������Ӳ�仯������ֵʱ���ô����ڵ��Ӳ�����
	/*	���Ӳ�ͼ��ֱ�Ӽ���ó��������Ӳ�ͼ��ͨ��cvValidateDisparity����ó���֮������������죻��������ֵ���Ӳ�ֵ�������㡣
	�ò���Ĭ��Ϊ -1������ִ�������Ӳ��顣	ע���ڳ�����Խ׶���ñ��ָ�ֵΪ -1���Ա�鿴��ͬ�Ӳ�����ɵ��Ӳ�Ч����	*/
	bm->setDisp12MaxDiff(1);

	/*�����Ӳ�*/
	bm->compute(imgL, imgR, imgDisparity16S);

	//-- Check its extreme values
	double minVal; double maxVal;
	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
	cout << minVal << "\t" << maxVal << endl;

	//--Display it as a CV_8UC1 image��16λ�з���תΪ8λ�޷���
	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));

	//�������ֵ
	//Mat xyz;
	//reprojectImageTo3D(imgDisparity16S, xyz, Q, true); //��ʵ�������ʱ��ReprojectTo3D������X / W, Y / W, Z / W��Ҫ����16(Ҳ����W����16)�����ܵõ���ȷ����ά������Ϣ��
	//xyz = xyz * 16;
	//saveXYZ("xyz.xls", xyz);


	////ת��Ϊ��ɫͼ
	//Mat imgColor(imgDisparity16S.size(), CV_8UC3);
	//GenerateFalseMap(imgDisparity8U, imgColor);//ת�ɲ�ͼ
	//imshow("disparityBM_color", imgColor);
}


/* ����ƥ�䣺SGBM�㷨 */
void calDispWithSGBM(cv::Mat imgL, cv::Mat imgR, cv::Mat &imgDisparity8U)
{
	Mat imgDisparity16S = Mat(imgL.rows, imgL.cols, CV_16S);

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
	sgbm->setPreFilterCap(32);	//Ԥ�����˲����ض�ֵ

	cv::Size imgSize = imgL.size();
	//int numberOfDisparities = ((imgSize.width / 8) + 15) & -16;		//�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��
	int numberOfDisparities = 320;

	int SADWindowSize = 7;		//SAD���ڴ�С
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm->setBlockSize(sgbmWinSize);	//��������SAD���ڴ�С
	int cn = imgL.channels();	//ͼ��ͨ����	
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);		//��̬�滮��������
	sgbm->setP2(64 * cn*sgbmWinSize*sgbmWinSize);

	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);	//�����Ӳ��
	sgbm->setUniquenessRatio(25);		//����������ֹ��ƥ��15
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);	//MODE_SGBM��MODE_HH��MODE_SGBM_3WAY

	sgbm->compute(imgL, imgR, imgDisparity16S);
	//-- Check its extreme values
	double minVal, maxVal;
	minMaxLoc(imgDisparity16S, &minVal, &maxVal);
	cout << minVal << "\t" << maxVal << endl;

	//ȥ�ڱ�
	//Mat img1p, img2p;
	//copyMakeBorder(imgL, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//copyMakeBorder(imgR, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//imgDisparity16S = imgDisparity16S.colRange(numberOfDisparities, img2p.cols - NumDisparities);

	//ת��Ϊ8λ�޷���ͼ Display it as a CV_8UC1 image��16λ�з���תΪ8λ�޷���
	imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (numberOfDisparities*16.));

	//�������ֵ
	//reprojectImageTo3D(imgDisparity16S, xyz, Q, true); //��ʵ�������ʱ��ReprojectTo3D������X / W, Y / W, Z / W��Ҫ����16(Ҳ����W����16)�����ܵõ���ȷ����ά������Ϣ��
	//xyz = xyz * 16;
	//saveXYZ("xyz.xls", xyz);

	//ת��Ϊ��ɫͼ
	//Mat imgColor(imgDisparity16S.size(), CV_8UC3);
	//GenerateFalseMap(imgDisparity8U, imgColor);//ת�ɲ�ͼ
	//imshow("disparitySGBM_color", imgColor);
}


/* �Ӳ�ͼת���ͼ */
void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, cv::Mat K)
{
	/*
	�������ã��Ӳ�ͼת���ͼ
	���룺����
		dispMap ----�Ӳ�ͼ��8λ��ͨ����CV_8UC1����
		K       ----�ڲξ���float����
	���������
		depthMap ----���ͼ��16λ�޷��ŵ�ͨ����CV_16UC1

	���㹫ʽ��
		depth = (f * baseline) / disp
		���У�f��ʾ��һ���Ľ��࣬Ҳ�����ڲ��е�fx�� baseline�������������֮��ľ��룬�������߾���
	*/
	int type = dispMap.type();

	float fx = K.at<float>(0, 0);
	float fy = K.at<float>(1, 1);
	float cx = K.at<float>(0, 2);
	float cy = K.at<float>(1, 2);
	float baseline = 178.089;	//���߾���65mm

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
					continue;  //��ֹ0��
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

	/* �Ӳ�ͼ���㣨�ɼ���У���������ͼ�����õ��� */
	Mat imgL = imread(path_left, 0);		//1_left_4
	Mat imgR = imread(path_right, 0);		//1_right_4
	imshow("imgL", imgL);
	imshow("imgR", imgR);
	// And create the image in which we will save our disparities
	//---BM---
	Mat imgDisparityBM = Mat(imgL.rows, imgL.cols, CV_8UC1);	//�Ӳ�ͼ
	calDispWithBM(imgL, imgR, imgDisparityBM);
	imshow("disparity_BM", imgDisparityBM);
	//ת��Ϊ��ɫͼ
	Mat imgColorBM(imgDisparityBM.size(), CV_8UC3);
	GenerateFalseMap(imgDisparityBM, imgColorBM);//ת�ɲ�ͼ
	imshow("disparityBM_color", imgColorBM);
	//����ͼƬ
	std::string disp_map_path_BM = path_left + ".BM.d.jpg";
	std::string disp_color_map_path_BM = path_left + ".BM.c.jpg";
	cv::imwrite(disp_map_path_BM, imgDisparityBM);
	cv::imwrite(disp_color_map_path_BM, imgColorBM);

	////---SGBM---
	//Mat imgDisparitySGBM = Mat(imgL.rows, imgL.cols, CV_8UC1);
	//calDispWithSGBM(imgL, imgR, imgDisparitySGBM);
	//imshow("disparity_SGBM", imgDisparitySGBM);
	////ת��Ϊ��ɫͼ
	//Mat imgColorSGBM(imgDisparitySGBM.size(), CV_8UC3);
	//GenerateFalseMap(imgDisparitySGBM, imgColorSGBM);//ת�ɲ�ͼ
	//imshow("disparitySGBM_color", imgColorSGBM);
	////����ͼƬ
	//std::string disp_map_path_SGBM = path_left + ".SGBM.d.jpg";
	//std::string disp_color_map_path_SGBM = path_left + ".SGBM.c.jpg";
	//cv::imwrite(disp_map_path_SGBM, imgDisparitySGBM);
	//cv::imwrite(disp_color_map_path_SGBM, imgColorSGBM);

	/* ���ͼ���㣨�Ӳ�ͼת���õ��� */
	//����1����ʽ����
	//Mat imgDepth(imgDisparityBM.size(), CV_16UC1);	//���ͼ
	//disp2Depth(imgDisparityBM, imgDepth, cameraMatrixL);
	//saveXYZ("xyz_1.xlsx", imgDepth);
	//imshow("Depth Image", imgDepth);
	//����2���⺯��
	//Mat xyz;
	//reprojectImageTo3D(imgDisparitySGBM, xyz, cameraMatrixL, true); //��ʵ�������ʱ��ReprojectTo3D������X / W, Y / W, Z / W��Ҫ����16(Ҳ����W����16)�����ܵõ���ȷ����ά������Ϣ��
	//xyz = xyz * 16;
	//saveXYZ("xyz_2.xlsx", xyz);

	cout << "OK" << endl;
	waitKey(0);
}