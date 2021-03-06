///******************************/
///*        立体匹配和测距        */
///******************************/
//
//#include <opencv2/opencv.hpp>  
//#include <iostream>  
//
//using namespace std;
//using namespace cv;
//
//const int imageWidth = 1280;                             //摄像头的分辨率  1414×962
//const int imageHeight = 720;
//Size imageSize = Size(imageWidth, imageHeight);
//
//Mat rgbImageL, grayImageL;
//Mat rgbImageR, grayImageR;
//Mat rectifyImageL, rectifyImageR;
//
//Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
//Rect validROIR;
//
//Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
//Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
//Mat xyz;              //三维坐标
//
//Point origin;         //鼠标按下的起始点
//Rect selection;      //定义矩形选框
//bool selectObject = false;    //是否选择对象
//
//int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
//Ptr<StereoBM> bm = StereoBM::create(16, 9);
//
///*
//事先标定好的相机的参数
//fx 0 cx
//0 fy cy
//0 0  1
//*/
////Mat cameraMatrixL = (Mat_<double>(3, 3) << 1426.379, 0, 712.043,
////	0, 1426.379, 476.526,
////	0, 0, 1);
////Mat distCoeffL = (Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
////
////Mat cameraMatrixR = (Mat_<double>(3, 3) << 1426.379, 0, 774.722,
////	0, 1426.379, 476.526,
////	0, 0, 1);
////Mat distCoeffR = (Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
////
////Mat T = (Mat_<double>(3, 1) << -39.7389449993974, 0.0740619639178984, 0.411914303245886);//T平移向量，左相机变换至右相机的平移向量
////Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量
////Mat R = (Mat_<double>(3, 3) << 0.999957725513956, -0.00103511880221423, 0.00913650447492805,
////	0.00114462826834523, 0.999927476064641, -0.0119888463633882,
////	-0.00912343197938050, 0.0119987974423658, 0.999886389470751);//R 旋转矩阵，相机1通过R变换得到相机2的位资
//
//Mat cameraMatrixL = (Mat_<double>(3, 3) << 707.268, 0, 648.263,
//	0, 707.796, 340.080,
//	0, 0, 1);
//Mat distCoeffL = (Mat_<double>(5, 1) << -0.3010, 0.0826, -0.0002, 0.0004, 0);
//
//Mat cameraMatrixR = (Mat_<double>(3, 3) << 704.986, 0, 649.719,
//	0, 706.100, 340.405,
//	0, 0, 1);
//Mat distCoeffR = (Mat_<double>(5, 1) << -0.2882, 0.0706, -0.0001, -0.0006, 0);
//
//
////Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);	//rec旋转向量
//Mat T = (Mat_<double>(3, 1) << 120.29977416992188, 0, 0);		//T平移向量，左相机变换至右相机的平移向量
//Mat R = (Mat_<double>(3, 3) << 0.99993348121643, 0.00084137916565, 0.01149332523346,
//	0.00114462826835, 0.999927476065, -0.0119888463634,
//	-0.00912343197938, 0.0119987974424, 0.999886389471);	//R 旋转矩阵，相机1通过R变换得到相机2的位资
//
//
//
///*****立体匹配*****/
//void stereo_match(int, void*)
//{	
//	bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
//	bm->setROI1(validROIL);
//	bm->setROI2(validROIR);
//
//	bm->setPreFilterCap(31);
//	bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
//	bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
//	
//	bm->setTextureThreshold(10);
//	bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
//	bm->setSpeckleWindowSize(100);
//	bm->setSpeckleRange(32);
//	bm->setDisp12MaxDiff(-1);
//
//	Mat disp, disp8;
//	bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
//	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
//	reprojectImageTo3D(disp8, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
//	xyz = xyz * 16;
//	imshow("disparity", disp8);
//}
//
///*****描述：鼠标操作回调*****/
//static void onMouse(int event, int x, int y, int, void*)
//{
//	if (selectObject)
//	{
//		selection.x = MIN(x, origin.x);
//		selection.y = MIN(y, origin.y);
//		selection.width = std::abs(x - origin.x);
//		selection.height = std::abs(y - origin.y);
//	}
//
//	switch (event)
//	{
//	case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
//		origin = Point(x, y);
//		selection = Rect(x, y, 0, 0);
//		selectObject = true;
//		cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
//		break;
//	case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
//		selectObject = false;
//		if (selection.width > 0 && selection.height > 0)
//			break;
//	}
//}
//
//
///*****主函数*****/
//int main()
//{
//	/*
//	相关函数说明：
//		stereoRectify()：立体校正函数，为每个相机计算立体校正的映射矩阵（不是直接将图片进行立体矫正）
//		initUndistortRectifyMap()：映射变换计算函数，计算畸变矫正和立体校正的映射变换
//		remap():几何变换函数，对图片进行立体校正
//
//		参考：StereoRectify()函数定义及用法畸变矫正与立体校正 https://blog.csdn.net/qq_36537774/article/details/85005552
//	*/
//	
//
//	/* 1 立体校正(参数计算)	*/
//	//Rodrigues(rec, R); //Rodrigues变换
//	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
//		0, imageSize, &validROIL, &validROIR);
//	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
//	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
//
//	/* 2 读取图片 */
//	rgbImageL = imread("./Fire/left_1.jpg", CV_LOAD_IMAGE_COLOR);
//	cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
//	rgbImageR = imread("./Fire/right_1.jpg", CV_LOAD_IMAGE_COLOR);
//	cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
//
//	imshow("ImageL Before Rectify", grayImageL);
//	imshow("ImageR Before Rectify", grayImageR);
//
//	/* 3 经过remap之后，左右相机的图像已经共面并且行对准了 */
//	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
//	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
//	//rectifyImageL = grayImageL;
//	//rectifyImageR = grayImageR;
//
//	/* 把校正结果显示出来 */
//	Mat rgbRectifyImageL, rgbRectifyImageR;
//	cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
//	cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);
//
//	//单独显示
//	//rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
//	//rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
//	imshow("ImageL After Rectify", rgbRectifyImageL);
//	imshow("ImageR After Rectify", rgbRectifyImageR);
//
//	//显示在同一张图上
//	Mat canvas;
//	double sf;
//	int w, h;
//	sf = 600. / MAX(imageSize.width, imageSize.height);
//	w = cvRound(imageSize.width * sf);
//	h = cvRound(imageSize.height * sf);
//	canvas.create(h, w * 2, CV_8UC3);   //注意通道
//
//	//左图像画到画布上
//	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
//	resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
//	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
//		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
//	//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
//	cout << "Painted ImageL" << endl;
//
//	//右图像画到画布上
//	canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
//	resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
//	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
//		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
//	//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
//	cout << "Painted ImageR" << endl;
//
//	//画上对应的线条
//	for (int i = 0; i < canvas.rows; i += 16)
//		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
//	imshow("rectified", canvas);
//
//	/* 4 立体匹配 */
//	namedWindow("disparity", CV_WINDOW_AUTOSIZE);
//	// 创建SAD窗口 Trackbar
//	createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
//	// 创建视差唯一性百分比窗口 Trackbar
//	createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
//	// 创建视差窗口 Trackbar
//	createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
//	//鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
//	setMouseCallback("disparity", onMouse, 0);
//	stereo_match(0, 0);
//
//	waitKey(0);
//	return 0;
//}