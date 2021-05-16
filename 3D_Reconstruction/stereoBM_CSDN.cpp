//#include <iostream>
//#include <opencv2/opencv.hpp>
//using namespace std;
//using namespace cv;
//
//
//struct StereoBMParams
//{
//	StereoBMParams(int _numDisparities = 64, int _SADWindowSize = 21)
//	{
//		preFilterType = StereoBM::PREFILTER_XSOBEL;
//		preFilterSize = 9;
//		preFilterCap = 31;
//		SADWindowSize = _SADWindowSize;
//		minDisparity = 0;
//		numDisparities = _numDisparities > 0 ? _numDisparities : 64;
//		textureThreshold = 10;
//		uniquenessRatio = 15;
//		speckleRange = speckleWindowSize = 0;
//		roi1 = roi2 = Rect(0, 0, 0, 0);
//		disp12MaxDiff = -1;
//		dispType = CV_16S;
//	}
//
//	int preFilterType;		//预处理滤波器类型 
//	int preFilterSize;		//预处理滤波器窗口大小，容许范围是[5,255]，一般应该在 5x5..21x21 之间，参数必须为奇数值
//	int preFilterCap;		//预处理滤波器的截断值，预处理的输出值仅保留[-preFilterCap, preFilterCap]范围内的值，参数范围：1-31
//	int SADWindowSize;		//SAD窗口大小，容许范围是[5,255]，一般应该在 5x5 至 21x21 之间，参数必须是奇数
//	int minDisparity;		//最小视差，默认值为 0, 可以是负值
//	int numDisparities;		//视差窗口，即最大视差值与最小视差值之差, 窗口大小必须是 16 的整数倍
//	
//	int textureThreshold;	//低纹理区域的判断阈值:	如果当前SAD窗口内所有邻居像素点的x导数绝对值之和小于指定阈值，则该窗口对应的像素点的视差值为 0
//	int uniquenessRatio;	//视差唯一性百分比:	视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0
//	int speckleRange;		//视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零
//	int speckleWindowSize;	//检查视差连通区域变化度的窗口大小
//	Rect roi1, roi2;
//	int disp12MaxDiff;		//左视差图（直接计算得出）和右视差图（通过cvValidateDisparity计算得出）之间的最大容许差异。	超过该阈值的视差值将被清零。
//	int dispType;
//};
//
//// 预处理Xsobel代码详解
//void prefilterXSobel(const Mat& src, Mat& dst, int ftzero)
//{
//	int x, y;
//	const int OFS = 256 * 4, TABSZ = OFS * 2 + 256;
//	uchar tab[TABSZ] = { 0 };
//	Size size = src.size();
//	//建立映射表，映射范围０到６１
//	for (x = 0; x < TABSZ; x++)
//		tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);
//	uchar val0 = tab[0 + OFS];
//	//竖直滑窗，每次可以处理两个
//	for (y = 0; y < size.height - 1; y += 2)
//	{
//		//防止越界访问及确定指针位置
//		const uchar* srow1 = src.ptr<uchar>(y);//指向当前操作行
//		//当为首行时，指向下一行，当不为首行时，指向上一行
//		const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
//		//当没有到下边界时，指向当前行的下一行
//		const uchar* srow2 = y < size.height - 1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
//		//当没有到下边界的倒数第二行时，指向当前操作行的下下行
//		const uchar* srow3 = y < size.height - 2 ? srow1 + src.step * 2 : srow1;
//		//初始化数据指针
//		uchar* dptr0 = dst.ptr<uchar>(y);
//		uchar* dptr1 = dptr0 + dst.step;
//		//处理首值问题
//		dptr0[0] = dptr0[size.width - 1] = dptr1[0] = dptr1[size.width - 1] = val0;
//		x = 1;
//		// 处理当前操作行的每一个元素
//		for (; x < size.width - 1; x++)
//		{
//			//计算xsobel的值
//			int d0 = srow0[x + 1] - srow0[x - 1], d1 = srow1[x + 1] - srow1[x - 1],
//				d2 = srow2[x + 1] - srow2[x - 1], d3 = srow3[x + 1] - srow3[x - 1];
//			//映射梯度
//			int v0 = tab[d0 + d1 * 2 + d2 + OFS];
//			int v1 = tab[d1 + d2 * 2 + d3 + OFS];
//			dptr0[x] = (uchar)v0;
//			dptr1[x] = (uchar)v1;
//
//		}
//	}
//	// 下边界处理
//	for (; y < size.height; y++)
//	{
//		uchar* dptr = dst.ptr<uchar>(y);
//		x = 0;
//		for (; x < size.width; x++)
//			dptr[x] = val0;
//	}
//}
//
//
//
//// 预处理的归一化
//void prefilterNorm(const Mat& src, Mat& dst, int winsize, int ftzero, uchar* buf)
//{
//	int x, y, wsz2 = winsize / 2;
//	//初始化做指针对齐
//	int* vsum = (int*)alignPtr(buf + (wsz2 + 1) * sizeof(vsum[0]), 32);
//	//映射因子计算
//	int scale_g = winsize * winsize / 8, scale_s = (1024 + scale_g) / (scale_g * 2);
//	const int OFS = 256 * 5, TABSZ = OFS * 2 + 256;
//	uchar tab[TABSZ];
//	//初始化访问图像指针
//	const uchar* sptr = src.ptr();
//	int srcstep = (int)src.step;
//	Size size = src.size();
//
//	scale_g *= scale_s;
//	//建立规划映射表
//	for (x = 0; x < TABSZ; x++)
//		tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);
//	//初始化vsum数据重复利用表，首先把图像中首行的数据全部乘以 wsz2+2 （该数据根据滑窗大小给出）保存到vsum中
//	//然后将首行以后的[１,wsz2)行的数据累加到vsum中完成竖直滑窗初始化。
//	for (x = 0; x < size.width; x++)
//		vsum[x] = (ushort)(sptr[x] * (wsz2 + 2));
//
//	for (y = 1; y < wsz2; y++)
//	{
//		for (x = 0; x < size.width; x++)
//			vsum[x] = (ushort)(vsum[x] + sptr[srcstep*y + x]);
//	}
//	//主循环开始，一行一行的处理
//	for (y = 0; y < size.height; y++)
//	{
//		// 判断边界问题，防止越界访问错误
//		const uchar* top = sptr + srcstep * MAX(y - wsz2 - 1, 0);
//		const uchar* bottom = sptr + srcstep * MIN(y + wsz2, size.height - 1);
//		const uchar* prev = sptr + srcstep * MAX(y - 1, 0);
//		const uchar* curr = sptr + srcstep * y;
//		const uchar* next = sptr + srcstep * MIN(y + 1, size.height - 1);
//		uchar* dptr = dst.ptr<uchar>(y);
//		//该循环为竖直滑窗向下滑，用第一个滑窗的总和减去第一个滑窗中的第一个元素，
//		//然后在加上第二个滑窗最后一个元素即为第二个滑窗的总和,依次类推对整个ｗｉｄｔｈ做一遍
//		for (x = 0; x < size.width; x++)
//			vsum[x] = (ushort)(vsum[x] + bottom[x] - top[x]);
//		//该循环用来对vsum做边界处理，把最左边值赋值给在指针对齐时预留出来的对应位置，也是对横向滑窗的一个预处理
//		for (x = 0; x <= wsz2; x++)
//		{
//			vsum[-x - 1] = vsum[0];
//			vsum[size.width + x] = vsum[size.width - 1];
//		}
//		//对归一化小窗口内的总和做处理，此处针对的是每次每行处理时的边界
//		int sum = vsum[0] * (wsz2 + 1);
//		for (x = 1; x <= wsz2; x++)
//			sum += vsum[x];
//		//对每行第一个像素进行归一化映射
//		int val = ((curr[0] * 5 + curr[1] + prev[0] + next[0])*scale_g - sum * scale_s) >> 10;
//		dptr[0] = tab[val + OFS];
//		//对每一行零以后的每个像素进行归一化映射
//		for (x = 1; x < size.width - 1; x++)
//		{
//			sum += vsum[x + wsz2] - vsum[x - wsz2 - 1];
//			val = ((curr[x] * 4 + curr[x - 1] + curr[x + 1] + prev[x] + next[x])*scale_g - sum * scale_s) >> 10;
//			dptr[x] = tab[val + OFS];
//		}
//		//处理最后边界问题，对每行最后一个像素进行归一化映射。
//		sum += vsum[x + wsz2] - vsum[x - wsz2 - 1];
//		val = ((curr[x] * 5 + curr[x - 1] + prev[x] + next[x])*scale_g - sum * scale_s) >> 10;
//		dptr[x] = tab[val + OFS];
//	}
//}
//
//
//// BM算法的实现位于源代码 modules/calib3d/src/stereobm.cpp文件中，函数名为 findStereoCorrespondenceBM
//// Computes the disparity map using block matching algorithm.
//// 具体参考：http://opencv.jp/opencv-2.1_org/py/camera_calibration_and_3d_reconstruction.html#findstereocorrespondencebm
//// 首先要对边界或者说是滑动窗口进行初始化，先要计算一个winSAD窗口大小的宽度和图像行数大小的区域的ＡＤ留作后边主循环用，主要采用的是滑窗共享部分数据，减少计算量
//template <typename mType>
//static void
//findStereoCorrespondenceBM(const Mat& left, const Mat& right,
//	Mat& disp, Mat& cost, const StereoBMParams& state,
//	uchar* buf, int _dy0, int _dy1, const int disp_shift)
//{
//	// opencv代码的特点：1.空间换时间：申请足够大的内存，预先计算出可以复用的数据并保存，后期直接查表使用；
//	// 		       2.非常好地定义和使用了各种指针和申请的内存。
//	// cost定义了Mat型，buf是一个指针，都不知道是输入还是输出。
//	const int ALIGN = 16;
//	int x, y, d;
//	int wsz = state.SADWindowSize, wsz2 = wsz / 2;	// windows size
//	int dy0 = MIN(_dy0, wsz2 + 1), dy1 = MIN(_dy1, wsz2 + 1); // dy0, dy1 是滑动窗口中心点到窗口第一行和最后一行的距离，
//					// 由于一般使用奇数大小的方形窗口，因此可以认为dy0 = dy1 = wsz2
//	int ndisp = state.numDisparities;	// 视差范围
//	int mindisp = 0;	// default state.minDisparity is 0;
//	int lofs = MAX(ndisp - 1 + mindisp, 0);	// left of start
//	int rofs = -MIN(ndisp - 1 + mindisp, 0);	// right of start
//	int width = left.cols, height = left.rows;
//	int width1 = width - rofs - ndisp + 1;
//	int ftzero = state.preFilterCap; // 这里是前面预处理做x方向的sobel滤波时的截断值，默认为31.
//					 // 预处理的结果并不是sobel滤波的直接结果，而是做了截断：
//					 // 滤波后的值如果小于-preFilterCap，则说说明纹理较强，结果为0；
//					 // 如果大于preFilterCap，则说明纹理强，结果为2*prefilterCap;
//					 // 如果滤波后结果在[-prefilterCap, preFilterCap]之间（区间表示，下同），对应取[0, 2*preFilterCap]。
//	int textureThreshold = state.textureThreshold;	// 纹理阈值
//	int uniquenessRatio = state.uniquenessRatio;	// 相似点的比率
//	mType FILTERED = (mType)((mindisp - 1) << disp_shift);	// 匹配失败的默认值
//	// 定义各个变量的指针
//	int *sad, *hsad0, *hsad, *hsad_sub, *htext;	// ndisp缓存的行指针
//								// htext 纹理缓存
//	uchar *cbuf0, *cbuf;	// 滑窗列的指针位置
//	// 定义一个指向行指针的变量
//	// cv::Ptr< T >::Ptr ()
//	// The default constructor creates a null Ptr - one that owns and stores a null pointer. 
//	const uchar* lptr0 = left.ptr() + lofs;
//	const uchar* rptr0 = right.ptr() + rofs;
//	const uchar *lptr, *lptr_sub, *rptr;
//	// Mat矩阵中数据指针Mat.data是uchar类型指针，CV_8U系列可以通过计算指针位置快速地定位矩阵中的任意元素。
//	// Mat::ptr()来获得指向某行元素的指针，在通过行数与通道数计算相应点的指针。
//	mType* dptr = disp.ptr<mType>();	// 
//	int sstep = (int)left.step;
//	int dstep = (int)(disp.step / sizeof(dptr[0]));
//	int cstep = (height + dy0 + dy1)*ndisp;
//	int costbuf = 0;
//	int coststep = cost.data ? (int)(cost.step / sizeof(costbuf)) : 0;
//	const int TABSZ = 256;
//	uchar tab[TABSZ];
//	//初始化做指针对齐
//	// cbuf0 -> htext -> hsad0 -> sad -> buf 
//	// 垂直偏移和水平偏移
//	sad = (int*)alignPtr(buf + sizeof(sad[0]), ALIGN); // 注意到sad的前面留了一个sizeof(sad[0])的位置，函数最后要用到。
//	hsad0 = (int*)alignPtr(sad + ndisp + 1 + dy0 * ndisp, ALIGN); // 这里额外说一句，opencv每次确定变量的字节数时都直接使用变量而不是int, double等类型，
//	// 这样当变量类型变化时可以少修改代码。
//	htext = (int*)alignPtr((int*)(hsad0 + (height + dy1)*ndisp) + wsz2 + 2, ALIGN);
//	cbuf0 = (uchar*)alignPtr((uchar*)(htext + height + wsz2 + 2) + dy0 * ndisp, ALIGN);
//
//	// 建立映射表，方便后面直接引用。以之前的x方向的sobel滤波的截断值为中心，距离这个截断值越远，说明纹理越强。
//	for (x = 0; x < TABSZ; x++)
//		tab[x] = (uchar)std::abs(x - ftzero);
//
//	// initialize buffers
//	// void *memset(void *s, int ch, size_t n);
//	// 将s中当前位置后面的n个字节 （typedef unsigned int size_t ）用 ch 替换并返回 s 
//	memset((hsad0 - dy0 * ndisp), 0, (height + dy0 + dy1)*ndisp * sizeof(hsad0[0]));
//	memset((htext - wsz2 - 1), 0, (height + wsz + 1) * sizeof(htext[0]));
//
//	// 首先初始化计算左图 x 在[-wsz2 - 1, wsz2), y 在[-dy0, height + dy1) 范围内的各个像素，
//	// 右图视差为[0. ndisp)像素之间的SAD. 
//	// 注意这里不处理 wsz2 列，并且是从-wsz2 - 1 列开始，（这一列不在第一个窗口[-wsz2, wsz2]中），
//	// 这是为了后续处理时逻辑统一和代码简化的需要。这样就可以在处理第一个滑动窗口时和处理之后的窗口一样，
//	// 剪掉滑出窗口的第一列的数据 (-wsz2 - 1)，加上新一列的数据 (wsz2)。
//	for (x = -wsz2 - 1; x < wsz2; x++)
//	{
//		// 统一先往上减去半个窗口乘以ndisp的距离。
//		hsad = hsad0 - dy0 * ndisp; // 结合下面的循环代码和内存示意图，hsad是累加的，每次回退dy0就好。
//		cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0 * ndisp; // 而cbuf, lptr, rptr 需要根据当前在不同x列的需要，移动指针指向当前所处理的列。
//		// lptr, rptr 相当于 Mat.at[][x]
//		lptr = lptr0 + std::min(std::max(x, -lofs), width - lofs - 1) - dy0 * sstep; // 前面的min, max 是为了防止内存越界而进行的判断。
//		rptr = rptr0 + std::min(std::max(x, -rofs), width - rofs - ndisp) - dy0 * sstep;
//
//		// 从SAD窗口的第一个像素开始。
//		// 循环都是以当前列为主，先处理当前列不同行的像素。
//		for (y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep)
//		{
//			int lval = lptr[0];
//			d = 0;
//
//			// 计算不同视差d 的SAD。也就是指定滑动范围内SAD值
//			for (; d < ndisp; d++)
//			{
//				int diff = std::abs(lval - rptr[d]); // SAD.
//				cbuf[d] = (uchar)diff; // 存储该列所有行各个像素在所有视差下的sad，所以cbuf的大小为wsz * cstep.
//				hsad[d] = (int)(hsad[d] + diff); // 累加同一行内，[-wsz2 - 1, wsz2) 像素，不同d下的SAD（预先进行一点cost aggregation）。
//			}
//			// 累计得到视察范围内的纹理值，判断像素值和阈值的大小
//			htext[y] += tab[lval]; // 利用之前的映射表，统计一行内，窗口大小宽度，左图像素的纹理度。
//				   // 注意到y是从-dy0开始的，而前面buf分配指针位置、hsad0和htext初始化为0的时候已经考虑到这一点了，
//				   // 特别是分配各个指针指向的内存大小的时候，分别都分配了下一个指针变量要往上减去的对应的内存大小。
//				   // 读者可以自己回去看alighPtr语句部分和memset部分。
//		}
//	}
//
//	// initialize the left and right borders of the disparity map
//	// 初始化图像左右边界的视差值
//	for (y = 0; y < height; y++)
//	{
//		for (x = 0; x < lofs; x++)
//			dptr[y*dstep + x] = FILTERED;
//		for (x = lofs + width1; x < width; x++)
//			dptr[y*dstep + x] = FILTERED;
//	}
//	// 移动视差图像位置
//	dptr += lofs; // 然后就可以跳过初始化的部分了。
//
//	// 进入主循环，滑动窗口法进行匹配。注意到该循环很大，包含了很多内循环。
//	// cost 是干什么的？
//	for (x = 0; x < width1; x++, dptr++)
//	{
//		int* costptr = cost.data ? cost.ptr<int>() + lofs + x : &costbuf;
//		int x0 = x - wsz2 - 1, x1 = x + wsz2; // 窗口的首尾x坐标。
//		// 同上，所有指针从窗口的第一行开始，即-dy0行。
//		// 由于之前已经初始化计算过了，x从0开始循环。
//		// cbuf_sub 从cbuf0 的第0行开始，cbuf在cbuf0的最后一行；下一次循环是cbuf_sub在第1行，cbuf在第0行，以此类推，存储了窗口宽度内，每一列的SAD.
//		const uchar* cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1))*cstep - dy0 * ndisp;
//		cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1))*cstep - dy0 * ndisp;
//		hsad = hsad0 - dy0 * ndisp;
//		// 这里了同样地，lptr_sub 从上一个窗口的最后一列开始，即x - wsz2 - 1，lptr从当前窗口的最后一列开始，即x + wsz2.
//		lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width - 1 - lofs) - dy0 * sstep;
//		lptr = lptr0 + MIN(MAX(x1, -lofs), width - 1 - lofs) - dy0 * sstep;
//		rptr = rptr0 + MIN(MAX(x1, -rofs), width - ndisp - rofs) - dy0 * sstep;
//
//		// 只算x1列，y 从-dy0到height + dy1 的SAD，将之更新到对应的变量中。
//		for (y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp, hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep)
//		{
//			int lval = lptr[0];
//			d = 0;
//			// 为什么要引入视差范围
//			for (; d < ndisp; d++)
//			{
//				int diff = std::abs(lval - rptr[d]); // 当前列的SAD.
//				cbuf[d] = (uchar)diff;
//				hsad[d] = hsad[d] + diff - cbuf_sub[d]; // 累加新一列各个像素不同d下的SAD，减去滑出窗口的那一列对应的SAD.
//			}
//			htext[y] += tab[lval] - tab[lptr_sub[0]]; // 同上，利用之前的映射表，统计一行内，窗口大小宽度，左图像素的纹理度。
//		}
//
//		// fill borders
//		// 这是什么意思，设置为常量？
//		for (y = dy1; y <= wsz2; y++)
//			htext[height + y] = htext[height + dy1 - 1];
//		for (y = -wsz2 - 1; y < -dy0; y++)
//			htext[y] = htext[-dy0];
//
//		// initialize sums
//		// 将hsad0存储的第-dy0列的数据乘以2拷贝给sad.
//		// sad终于有操作了，
//		for (d = 0; d < ndisp; d++)
//			sad[d] = (int)(hsad0[d - ndisp * dy0] * (wsz2 + 2 - dy0));
//
//		// 将hsad指向hsad0的第1-dy0行，循环也从1-dy0行开始，并且只处理窗口大小内的数据（到wsz2 - 1为止）。
//		// 不处理wsz2行和之前不处理wsz2列的原因是一样的。
//		hsad = hsad0 + (1 - dy0)*ndisp;
//		for (y = 1 - dy0; y < wsz2; y++, hsad += ndisp)
//		{
//			d = 0;
//
//			// cost aggregation 步骤
//			// 累加不同行、一个滑动窗口内各个像素取相同d 时的SAD。
//			for (; d < ndisp; d++)
//				sad[d] = (int)(sad[d] + hsad[d]);
//		}
//		// 循环累加一个滑动窗口内的纹理值。
//		int tsum = 0;
//		for (y = -wsz2 - 1; y < wsz2; y++)
//			tsum += htext[y];
//
//		// finally, start the real processing
//		// 虽然官方注释说现在才开始真正的处理，但之前已经做了大量的处理工作了。
//		// minsad,sda 最小值；mind，最小视差值；
//		for (y = 0; y < height; y++)
//		{
//			int minsad = INT_MAX, mind = -1;
//			hsad = hsad0 + MIN(y + wsz2, height + dy1 - 1)*ndisp; // 当前窗口的最后一行。
//			hsad_sub = hsad0 + MAX(y - wsz2 - 1, -dy0)*ndisp; // 上个窗口的最后一行。
//			d = 0;
//
//			// 寻找最优视差。
//			for (; d < ndisp; d++)
//			{
//				int currsad = sad[d] + hsad[d] - hsad_sub[d]; // 同上，加上最后一行的SAD，减去滑出那一行的SAD.
//								  // 之前给sad赋值时为何要乘以2也就清楚了。一样是为了使处理第一个窗口的SAD之和时和之后的窗口相同，
//								  // 可以剪掉第一行的SAD，加上新一行的SAD。所以必须乘以2防止计算第一个窗口是漏算了第一行。
//
//				sad[d] = currsad; // 更新当前d下的SAD之和，方便下次计算使用。
//				if (currsad < minsad)
//				{
//					// 得到视差最小值和最小视差位置
//					minsad = currsad;
//					mind = d;
//				}
//			}
//
//			tsum += htext[y + wsz2] - htext[y - wsz2 - 1]; // 同样需要更新纹理值。
//			// 如果一个像素附近的纹理太弱，则视差计算认为无效。
//			if (tsum < textureThreshold)
//			{
//				dptr[y*dstep] = FILTERED;
//				continue;
//			}
//
//			// 唯一性匹配。
//			// 对于前面找到的最优视差mind，及其SAD minsad，自适应阈值为minsad * (1 + uniquenessRatio).
//			// 要求除了mind 前后一个视差之外，其余的视差的SAD都必须比阈值大，否则认为找到的视差无效。
//			// continue语句只结束本次循环，而不终止整个循环的执行。而break语句则是结束整个循环过程，不再判断执行循环的条件是否成立
//			if (uniquenessRatio > 0)
//			{
//				int thresh = minsad + (minsad * uniquenessRatio / 100);
//				// 这句话的作用是什么？ 得到d，有效的话，d 应该等于 ndisp
//				for (d = 0; d < ndisp; d++)
//				{
//					// break 结束当前 for，foreach，while，do-while 或者 switch 结构的执行。
//					if ((d < mind - 1 || d > mind + 1) && sad[d] <= thresh)
//						break;
//				}
//				// 如果，d小于ndisp,表示 uniquenessRatio
//				if (d < ndisp)
//				{
//					// 结束本次循环,即跳过循环体下面尚未执行的语句,接着进行下一次是否执行循环的判断.
//					dptr[y*dstep] = FILTERED;
//					continue;
//				}
//			}
//			// 如果，d < ndisp,dptr[]标注为失败，否则，
//			{
//				// 最后，经过层层校验，终于确定了当前像素的视差。
//				// 回顾之前sad指针在确定其指针位置和指向的大小时，前后都留了一个位置，在这里用到了。
//				sad[-1] = sad[1];
//				sad[ndisp] = sad[ndisp - 2];
//				// 视差优化
//				// 这里留两个位置的作用就很明显了：防止mind为0或ndis-1时下面的语句数组越界。
//				// p是sad最小值的后一个位置，n是前一个位置，这里的d为 p+n-2d+(p-n),判断d是否为0，如果，为0是对称的，不为零，加一个线性偏移量
//				int p = sad[mind + 1], n = sad[mind - 1];
//				d = p + n - 2 * sad[mind] + std::abs(p - n);
//				//  注意到前面将dptr的位置加上了lofs位，所以这里下标为y * dstep。
//				dptr[y*dstep] = (mType)(((ndisp - mind - 1 + mindisp) * 256 + (d != 0 ? (p - n) * 256 / d : 0) + 15) // 这里如果读者留心，会发现之前计算视差d时，计算结果是反过来的。
//														 // 即d=0时，理论上右图像素应该是和左图像素相同的x坐标，
//														 // 但其实之前在设置rptr是，此时右图像素的x坐标为x-(ndisp-1)，
//														 // 因此这里所算的视差要反转过来，为ndisp-mind-1。
//														 // 常数15是因为opencv默认输出类型为16位整数，后面为了获得真正的视差要除以16，
//														 // 这里加的一个针对整数类型除法截断的一个保护。
//														 // 至于为何多了一个(p-n)/d，我也不太懂，应该是针对所计算的SAD的变化率的一个补偿，希望有人可以指点下:)
//					>> (DISPARITY_SHIFT_32S - disp_shift));
//				costptr[y*coststep] = sad[mind]; // 最后opencv默认得到的视差值需要乘以16，所以前面乘以256，后面在右移4位。
//			}
//		} // y 
//	}// x
//}
//
//
