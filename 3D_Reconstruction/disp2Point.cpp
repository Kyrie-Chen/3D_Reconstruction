/* 
* Author：		Yangbo Chen
* Describe:		视差图重建为三维点云
* Update date：	2021.06.30
* 参考：https://blog.csdn.net/YunLaowang/article/details/86644361
*/


/*
相机参数：
	cam0 = [4152.073 0 1288.147; 0 4152.073 973.571; 0 0 1]
	cam1 = [4152.073 0 1501.231; 0 4152.073 973.571; 0 0 1]
	 doffs = 213.084
	baseline = 176.252
	width = 2872
	height = 1984
相机内参数矩阵：
	K=[fx 0 u0; 0 fy v0; 0 0 1]

	doffs = |u1 - u0|
*/
#define _SILENCE_FPOS_SEEKPOS_DEPRECATION_WARNING
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>  
#include <pcl/io/io.h>  
#include <pcl/io/pcd_io.h>  
#include <opencv2/opencv.hpp>  

using namespace cv;
using namespace std;
using namespace pcl;

int user_data;
// 相机内参
const double u0 = 1288.147;
const double v0 = 973.571;
const double fx = 4152.073;
const double fy = 4152.073;
const double baseline = 176.252;
const double doffs = 213.084;	// 代表两个相机主点在x方向上的差距, doffs = |u1 - u0|

void viewerOneOff(visualization::PCLVisualizer& viewer)
{
	viewer.setBackgroundColor(0.0, 0.0, 0.0);
}

int main()
{
	// 读入数据
	Mat color = imread("./data02/rect_1280/left_1.jpg"); // RGB
	Mat depth = imread("./data02/rect_1280/left_1.jpg.BM.d.jpg", IMREAD_UNCHANGED);// depth
	if (color.empty() || depth.empty())
	{
		cout << "The image is empty, please check it!" << endl;
		return -1;
	}

	// 相机坐标系下的点云
	PointCloud<PointXYZRGB>::Ptr cloud(new PointCloud<PointXYZRGB>);

	for (int row = 0; row < depth.rows; row++)
	{
		for (int col = 0; col < depth.cols; col++)
		{
			ushort d = depth.ptr<ushort>(row)[col];

			if (d == 0)
				continue;
			PointXYZRGB p;

			// depth			
			p.z = fx * baseline / (d + doffs); // Zc = baseline * f / (d + doffs)
			p.x = (col - u0) * p.z / fx; // Xc向右，Yc向下为正
			p.y = (row - v0) * p.z / fy;

			p.y = -p.y;  // 为便于显示，绕x轴三维旋转180°
			p.z = -p.z;

			// RGB
			p.b = color.ptr<uchar>(row)[col * 3];
			p.g = color.ptr<uchar>(row)[col * 3 + 1];
			p.r = color.ptr<uchar>(row)[col * 3 + 2];

			cloud->points.push_back(p);
		}
	}

	cloud->height = depth.rows;
	cloud->width = depth.cols;
	cloud->points.resize(cloud->height * cloud->width);

	visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(cloud);
	viewer.runOnVisualizationThreadOnce(viewerOneOff);

	while (!viewer.wasStopped())
	{
		user_data = 9;
	}
	return 0;
}
