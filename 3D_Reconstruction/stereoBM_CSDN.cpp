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
//	int preFilterType;		//Ԥ�����˲������� 
//	int preFilterSize;		//Ԥ�����˲������ڴ�С������Χ��[5,255]��һ��Ӧ���� 5x5..21x21 ֮�䣬��������Ϊ����ֵ
//	int preFilterCap;		//Ԥ�����˲����Ľض�ֵ��Ԥ��������ֵ������[-preFilterCap, preFilterCap]��Χ�ڵ�ֵ��������Χ��1-31
//	int SADWindowSize;		//SAD���ڴ�С������Χ��[5,255]��һ��Ӧ���� 5x5 �� 21x21 ֮�䣬��������������
//	int minDisparity;		//��С�ӲĬ��ֵΪ 0, �����Ǹ�ֵ
//	int numDisparities;		//�Ӳ�ڣ�������Ӳ�ֵ����С�Ӳ�ֵ֮��, ���ڴ�С������ 16 ��������
//	
//	int textureThreshold;	//������������ж���ֵ:	�����ǰSAD�����������ھ����ص��x��������ֵ֮��С��ָ����ֵ����ô��ڶ�Ӧ�����ص���Ӳ�ֵΪ 0
//	int uniquenessRatio;	//�Ӳ�Ψһ�԰ٷֱ�:	�Ӳ�ڷ�Χ����ʹ����Ǵεʹ��۵�(1 + uniquenessRatio / 100)��ʱ����ʹ��۶�Ӧ���Ӳ�ֵ���Ǹ����ص���Ӳ��������ص���Ӳ�Ϊ 0
//	int speckleRange;		//�Ӳ�仯��ֵ�����������Ӳ�仯������ֵʱ���ô����ڵ��Ӳ�����
//	int speckleWindowSize;	//����Ӳ���ͨ����仯�ȵĴ��ڴ�С
//	Rect roi1, roi2;
//	int disp12MaxDiff;		//���Ӳ�ͼ��ֱ�Ӽ���ó��������Ӳ�ͼ��ͨ��cvValidateDisparity����ó���֮������������졣	��������ֵ���Ӳ�ֵ�������㡣
//	int dispType;
//};
//
//// Ԥ����Xsobel�������
//void prefilterXSobel(const Mat& src, Mat& dst, int ftzero)
//{
//	int x, y;
//	const int OFS = 256 * 4, TABSZ = OFS * 2 + 256;
//	uchar tab[TABSZ] = { 0 };
//	Size size = src.size();
//	//����ӳ���ӳ�䷶Χ��������
//	for (x = 0; x < TABSZ; x++)
//		tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);
//	uchar val0 = tab[0 + OFS];
//	//��ֱ������ÿ�ο��Դ�������
//	for (y = 0; y < size.height - 1; y += 2)
//	{
//		//��ֹԽ����ʼ�ȷ��ָ��λ��
//		const uchar* srow1 = src.ptr<uchar>(y);//ָ��ǰ������
//		//��Ϊ����ʱ��ָ����һ�У�����Ϊ����ʱ��ָ����һ��
//		const uchar* srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
//		//��û�е��±߽�ʱ��ָ��ǰ�е���һ��
//		const uchar* srow2 = y < size.height - 1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
//		//��û�е��±߽�ĵ����ڶ���ʱ��ָ��ǰ�����е�������
//		const uchar* srow3 = y < size.height - 2 ? srow1 + src.step * 2 : srow1;
//		//��ʼ������ָ��
//		uchar* dptr0 = dst.ptr<uchar>(y);
//		uchar* dptr1 = dptr0 + dst.step;
//		//������ֵ����
//		dptr0[0] = dptr0[size.width - 1] = dptr1[0] = dptr1[size.width - 1] = val0;
//		x = 1;
//		// ����ǰ�����е�ÿһ��Ԫ��
//		for (; x < size.width - 1; x++)
//		{
//			//����xsobel��ֵ
//			int d0 = srow0[x + 1] - srow0[x - 1], d1 = srow1[x + 1] - srow1[x - 1],
//				d2 = srow2[x + 1] - srow2[x - 1], d3 = srow3[x + 1] - srow3[x - 1];
//			//ӳ���ݶ�
//			int v0 = tab[d0 + d1 * 2 + d2 + OFS];
//			int v1 = tab[d1 + d2 * 2 + d3 + OFS];
//			dptr0[x] = (uchar)v0;
//			dptr1[x] = (uchar)v1;
//
//		}
//	}
//	// �±߽紦��
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
//// Ԥ����Ĺ�һ��
//void prefilterNorm(const Mat& src, Mat& dst, int winsize, int ftzero, uchar* buf)
//{
//	int x, y, wsz2 = winsize / 2;
//	//��ʼ����ָ�����
//	int* vsum = (int*)alignPtr(buf + (wsz2 + 1) * sizeof(vsum[0]), 32);
//	//ӳ�����Ӽ���
//	int scale_g = winsize * winsize / 8, scale_s = (1024 + scale_g) / (scale_g * 2);
//	const int OFS = 256 * 5, TABSZ = OFS * 2 + 256;
//	uchar tab[TABSZ];
//	//��ʼ������ͼ��ָ��
//	const uchar* sptr = src.ptr();
//	int srcstep = (int)src.step;
//	Size size = src.size();
//
//	scale_g *= scale_s;
//	//�����滮ӳ���
//	for (x = 0; x < TABSZ; x++)
//		tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);
//	//��ʼ��vsum�����ظ����ñ����Ȱ�ͼ�������е�����ȫ������ wsz2+2 �������ݸ��ݻ�����С���������浽vsum��
//	//Ȼ�������Ժ��[��,wsz2)�е������ۼӵ�vsum�������ֱ������ʼ����
//	for (x = 0; x < size.width; x++)
//		vsum[x] = (ushort)(sptr[x] * (wsz2 + 2));
//
//	for (y = 1; y < wsz2; y++)
//	{
//		for (x = 0; x < size.width; x++)
//			vsum[x] = (ushort)(vsum[x] + sptr[srcstep*y + x]);
//	}
//	//��ѭ����ʼ��һ��һ�еĴ���
//	for (y = 0; y < size.height; y++)
//	{
//		// �жϱ߽����⣬��ֹԽ����ʴ���
//		const uchar* top = sptr + srcstep * MAX(y - wsz2 - 1, 0);
//		const uchar* bottom = sptr + srcstep * MIN(y + wsz2, size.height - 1);
//		const uchar* prev = sptr + srcstep * MAX(y - 1, 0);
//		const uchar* curr = sptr + srcstep * y;
//		const uchar* next = sptr + srcstep * MIN(y + 1, size.height - 1);
//		uchar* dptr = dst.ptr<uchar>(y);
//		//��ѭ��Ϊ��ֱ�������»����õ�һ���������ܺͼ�ȥ��һ�������еĵ�һ��Ԫ�أ�
//		//Ȼ���ڼ��ϵڶ����������һ��Ԫ�ؼ�Ϊ�ڶ����������ܺ�,�������ƶ���������������һ��
//		for (x = 0; x < size.width; x++)
//			vsum[x] = (ushort)(vsum[x] + bottom[x] - top[x]);
//		//��ѭ��������vsum���߽紦���������ֵ��ֵ����ָ�����ʱԤ�������Ķ�Ӧλ�ã�Ҳ�ǶԺ��򻬴���һ��Ԥ����
//		for (x = 0; x <= wsz2; x++)
//		{
//			vsum[-x - 1] = vsum[0];
//			vsum[size.width + x] = vsum[size.width - 1];
//		}
//		//�Թ�һ��С�����ڵ��ܺ��������˴���Ե���ÿ��ÿ�д���ʱ�ı߽�
//		int sum = vsum[0] * (wsz2 + 1);
//		for (x = 1; x <= wsz2; x++)
//			sum += vsum[x];
//		//��ÿ�е�һ�����ؽ��й�һ��ӳ��
//		int val = ((curr[0] * 5 + curr[1] + prev[0] + next[0])*scale_g - sum * scale_s) >> 10;
//		dptr[0] = tab[val + OFS];
//		//��ÿһ�����Ժ��ÿ�����ؽ��й�һ��ӳ��
//		for (x = 1; x < size.width - 1; x++)
//		{
//			sum += vsum[x + wsz2] - vsum[x - wsz2 - 1];
//			val = ((curr[x] * 4 + curr[x - 1] + curr[x + 1] + prev[x] + next[x])*scale_g - sum * scale_s) >> 10;
//			dptr[x] = tab[val + OFS];
//		}
//		//�������߽����⣬��ÿ�����һ�����ؽ��й�һ��ӳ�䡣
//		sum += vsum[x + wsz2] - vsum[x - wsz2 - 1];
//		val = ((curr[x] * 5 + curr[x - 1] + prev[x] + next[x])*scale_g - sum * scale_s) >> 10;
//		dptr[x] = tab[val + OFS];
//	}
//}
//
//
//// BM�㷨��ʵ��λ��Դ���� modules/calib3d/src/stereobm.cpp�ļ��У�������Ϊ findStereoCorrespondenceBM
//// Computes the disparity map using block matching algorithm.
//// ����ο���http://opencv.jp/opencv-2.1_org/py/camera_calibration_and_3d_reconstruction.html#findstereocorrespondencebm
//// ����Ҫ�Ա߽����˵�ǻ������ڽ��г�ʼ������Ҫ����һ��winSAD���ڴ�С�Ŀ�Ⱥ�ͼ��������С������ģ������������ѭ���ã���Ҫ���õ��ǻ������������ݣ����ټ�����
//template <typename mType>
//static void
//findStereoCorrespondenceBM(const Mat& left, const Mat& right,
//	Mat& disp, Mat& cost, const StereoBMParams& state,
//	uchar* buf, int _dy0, int _dy1, const int disp_shift)
//{
//	// opencv������ص㣺1.�ռ任ʱ�䣺�����㹻����ڴ棬Ԥ�ȼ�������Ը��õ����ݲ����棬����ֱ�Ӳ��ʹ�ã�
//	// 		       2.�ǳ��õض����ʹ���˸���ָ���������ڴ档
//	// cost������Mat�ͣ�buf��һ��ָ�룬����֪�������뻹�������
//	const int ALIGN = 16;
//	int x, y, d;
//	int wsz = state.SADWindowSize, wsz2 = wsz / 2;	// windows size
//	int dy0 = MIN(_dy0, wsz2 + 1), dy1 = MIN(_dy1, wsz2 + 1); // dy0, dy1 �ǻ����������ĵ㵽���ڵ�һ�к����һ�еľ��룬
//					// ����һ��ʹ��������С�ķ��δ��ڣ���˿�����Ϊdy0 = dy1 = wsz2
//	int ndisp = state.numDisparities;	// �ӲΧ
//	int mindisp = 0;	// default state.minDisparity is 0;
//	int lofs = MAX(ndisp - 1 + mindisp, 0);	// left of start
//	int rofs = -MIN(ndisp - 1 + mindisp, 0);	// right of start
//	int width = left.cols, height = left.rows;
//	int width1 = width - rofs - ndisp + 1;
//	int ftzero = state.preFilterCap; // ������ǰ��Ԥ������x�����sobel�˲�ʱ�Ľض�ֵ��Ĭ��Ϊ31.
//					 // Ԥ����Ľ��������sobel�˲���ֱ�ӽ�����������˽ضϣ�
//					 // �˲����ֵ���С��-preFilterCap����˵˵�������ǿ�����Ϊ0��
//					 // �������preFilterCap����˵������ǿ�����Ϊ2*prefilterCap;
//					 // ����˲�������[-prefilterCap, preFilterCap]֮�䣨�����ʾ����ͬ������Ӧȡ[0, 2*preFilterCap]��
//	int textureThreshold = state.textureThreshold;	// ������ֵ
//	int uniquenessRatio = state.uniquenessRatio;	// ���Ƶ�ı���
//	mType FILTERED = (mType)((mindisp - 1) << disp_shift);	// ƥ��ʧ�ܵ�Ĭ��ֵ
//	// �������������ָ��
//	int *sad, *hsad0, *hsad, *hsad_sub, *htext;	// ndisp�������ָ��
//								// htext ������
//	uchar *cbuf0, *cbuf;	// �����е�ָ��λ��
//	// ����һ��ָ����ָ��ı���
//	// cv::Ptr< T >::Ptr ()
//	// The default constructor creates a null Ptr - one that owns and stores a null pointer. 
//	const uchar* lptr0 = left.ptr() + lofs;
//	const uchar* rptr0 = right.ptr() + rofs;
//	const uchar *lptr, *lptr_sub, *rptr;
//	// Mat����������ָ��Mat.data��uchar����ָ�룬CV_8Uϵ�п���ͨ������ָ��λ�ÿ��ٵض�λ�����е�����Ԫ�ء�
//	// Mat::ptr()�����ָ��ĳ��Ԫ�ص�ָ�룬��ͨ��������ͨ����������Ӧ���ָ�롣
//	mType* dptr = disp.ptr<mType>();	// 
//	int sstep = (int)left.step;
//	int dstep = (int)(disp.step / sizeof(dptr[0]));
//	int cstep = (height + dy0 + dy1)*ndisp;
//	int costbuf = 0;
//	int coststep = cost.data ? (int)(cost.step / sizeof(costbuf)) : 0;
//	const int TABSZ = 256;
//	uchar tab[TABSZ];
//	//��ʼ����ָ�����
//	// cbuf0 -> htext -> hsad0 -> sad -> buf 
//	// ��ֱƫ�ƺ�ˮƽƫ��
//	sad = (int*)alignPtr(buf + sizeof(sad[0]), ALIGN); // ע�⵽sad��ǰ������һ��sizeof(sad[0])��λ�ã��������Ҫ�õ���
//	hsad0 = (int*)alignPtr(sad + ndisp + 1 + dy0 * ndisp, ALIGN); // �������˵һ�䣬opencvÿ��ȷ���������ֽ���ʱ��ֱ��ʹ�ñ���������int, double�����ͣ�
//	// �������������ͱ仯ʱ�������޸Ĵ��롣
//	htext = (int*)alignPtr((int*)(hsad0 + (height + dy1)*ndisp) + wsz2 + 2, ALIGN);
//	cbuf0 = (uchar*)alignPtr((uchar*)(htext + height + wsz2 + 2) + dy0 * ndisp, ALIGN);
//
//	// ����ӳ����������ֱ�����á���֮ǰ��x�����sobel�˲��Ľض�ֵΪ���ģ���������ض�ֵԽԶ��˵������Խǿ��
//	for (x = 0; x < TABSZ; x++)
//		tab[x] = (uchar)std::abs(x - ftzero);
//
//	// initialize buffers
//	// void *memset(void *s, int ch, size_t n);
//	// ��s�е�ǰλ�ú����n���ֽ� ��typedef unsigned int size_t ���� ch �滻������ s 
//	memset((hsad0 - dy0 * ndisp), 0, (height + dy0 + dy1)*ndisp * sizeof(hsad0[0]));
//	memset((htext - wsz2 - 1), 0, (height + wsz + 1) * sizeof(htext[0]));
//
//	// ���ȳ�ʼ��������ͼ x ��[-wsz2 - 1, wsz2), y ��[-dy0, height + dy1) ��Χ�ڵĸ������أ�
//	// ��ͼ�Ӳ�Ϊ[0. ndisp)����֮���SAD. 
//	// ע�����ﲻ���� wsz2 �У������Ǵ�-wsz2 - 1 �п�ʼ������һ�в��ڵ�һ������[-wsz2, wsz2]�У���
//	// ����Ϊ�˺�������ʱ�߼�ͳһ�ʹ���򻯵���Ҫ�������Ϳ����ڴ����һ����������ʱ�ʹ���֮��Ĵ���һ����
//	// �����������ڵĵ�һ�е����� (-wsz2 - 1)��������һ�е����� (wsz2)��
//	for (x = -wsz2 - 1; x < wsz2; x++)
//	{
//		// ͳһ�����ϼ�ȥ������ڳ���ndisp�ľ��롣
//		hsad = hsad0 - dy0 * ndisp; // ��������ѭ��������ڴ�ʾ��ͼ��hsad���ۼӵģ�ÿ�λ���dy0�ͺá�
//		cbuf = cbuf0 + (x + wsz2 + 1)*cstep - dy0 * ndisp; // ��cbuf, lptr, rptr ��Ҫ���ݵ�ǰ�ڲ�ͬx�е���Ҫ���ƶ�ָ��ָ��ǰ��������С�
//		// lptr, rptr �൱�� Mat.at[][x]
//		lptr = lptr0 + std::min(std::max(x, -lofs), width - lofs - 1) - dy0 * sstep; // ǰ���min, max ��Ϊ�˷�ֹ�ڴ�Խ������е��жϡ�
//		rptr = rptr0 + std::min(std::max(x, -rofs), width - rofs - ndisp) - dy0 * sstep;
//
//		// ��SAD���ڵĵ�һ�����ؿ�ʼ��
//		// ѭ�������Ե�ǰ��Ϊ�����ȴ���ǰ�в�ͬ�е����ء�
//		for (y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep)
//		{
//			int lval = lptr[0];
//			d = 0;
//
//			// ���㲻ͬ�Ӳ�d ��SAD��Ҳ����ָ��������Χ��SADֵ
//			for (; d < ndisp; d++)
//			{
//				int diff = std::abs(lval - rptr[d]); // SAD.
//				cbuf[d] = (uchar)diff; // �洢���������и��������������Ӳ��µ�sad������cbuf�Ĵ�СΪwsz * cstep.
//				hsad[d] = (int)(hsad[d] + diff); // �ۼ�ͬһ���ڣ�[-wsz2 - 1, wsz2) ���أ���ͬd�µ�SAD��Ԥ�Ƚ���һ��cost aggregation����
//			}
//			// �ۼƵõ��Ӳ췶Χ�ڵ�����ֵ���ж�����ֵ����ֵ�Ĵ�С
//			htext[y] += tab[lval]; // ����֮ǰ��ӳ���ͳ��һ���ڣ����ڴ�С��ȣ���ͼ���ص�����ȡ�
//				   // ע�⵽y�Ǵ�-dy0��ʼ�ģ���ǰ��buf����ָ��λ�á�hsad0��htext��ʼ��Ϊ0��ʱ���Ѿ����ǵ���һ���ˣ�
//				   // �ر��Ƿ������ָ��ָ����ڴ��С��ʱ�򣬷ֱ𶼷�������һ��ָ�����Ҫ���ϼ�ȥ�Ķ�Ӧ���ڴ��С��
//				   // ���߿����Լ���ȥ��alighPtr��䲿�ֺ�memset���֡�
//		}
//	}
//
//	// initialize the left and right borders of the disparity map
//	// ��ʼ��ͼ�����ұ߽���Ӳ�ֵ
//	for (y = 0; y < height; y++)
//	{
//		for (x = 0; x < lofs; x++)
//			dptr[y*dstep + x] = FILTERED;
//		for (x = lofs + width1; x < width; x++)
//			dptr[y*dstep + x] = FILTERED;
//	}
//	// �ƶ��Ӳ�ͼ��λ��
//	dptr += lofs; // Ȼ��Ϳ���������ʼ���Ĳ����ˡ�
//
//	// ������ѭ�����������ڷ�����ƥ�䡣ע�⵽��ѭ���ܴ󣬰����˺ܶ���ѭ����
//	// cost �Ǹ�ʲô�ģ�
//	for (x = 0; x < width1; x++, dptr++)
//	{
//		int* costptr = cost.data ? cost.ptr<int>() + lofs + x : &costbuf;
//		int x0 = x - wsz2 - 1, x1 = x + wsz2; // ���ڵ���βx���ꡣ
//		// ͬ�ϣ�����ָ��Ӵ��ڵĵ�һ�п�ʼ����-dy0�С�
//		// ����֮ǰ�Ѿ���ʼ��������ˣ�x��0��ʼѭ����
//		// cbuf_sub ��cbuf0 �ĵ�0�п�ʼ��cbuf��cbuf0�����һ�У���һ��ѭ����cbuf_sub�ڵ�1�У�cbuf�ڵ�0�У��Դ����ƣ��洢�˴��ڿ���ڣ�ÿһ�е�SAD.
//		const uchar* cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1))*cstep - dy0 * ndisp;
//		cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1))*cstep - dy0 * ndisp;
//		hsad = hsad0 - dy0 * ndisp;
//		// ������ͬ���أ�lptr_sub ����һ�����ڵ����һ�п�ʼ����x - wsz2 - 1��lptr�ӵ�ǰ���ڵ����һ�п�ʼ����x + wsz2.
//		lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width - 1 - lofs) - dy0 * sstep;
//		lptr = lptr0 + MIN(MAX(x1, -lofs), width - 1 - lofs) - dy0 * sstep;
//		rptr = rptr0 + MIN(MAX(x1, -rofs), width - ndisp - rofs) - dy0 * sstep;
//
//		// ֻ��x1�У�y ��-dy0��height + dy1 ��SAD����֮���µ���Ӧ�ı����С�
//		for (y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp, hsad += ndisp, lptr += sstep, lptr_sub += sstep, rptr += sstep)
//		{
//			int lval = lptr[0];
//			d = 0;
//			// ΪʲôҪ�����ӲΧ
//			for (; d < ndisp; d++)
//			{
//				int diff = std::abs(lval - rptr[d]); // ��ǰ�е�SAD.
//				cbuf[d] = (uchar)diff;
//				hsad[d] = hsad[d] + diff - cbuf_sub[d]; // �ۼ���һ�и������ز�ͬd�µ�SAD����ȥ�������ڵ���һ�ж�Ӧ��SAD.
//			}
//			htext[y] += tab[lval] - tab[lptr_sub[0]]; // ͬ�ϣ�����֮ǰ��ӳ���ͳ��һ���ڣ����ڴ�С��ȣ���ͼ���ص�����ȡ�
//		}
//
//		// fill borders
//		// ����ʲô��˼������Ϊ������
//		for (y = dy1; y <= wsz2; y++)
//			htext[height + y] = htext[height + dy1 - 1];
//		for (y = -wsz2 - 1; y < -dy0; y++)
//			htext[y] = htext[-dy0];
//
//		// initialize sums
//		// ��hsad0�洢�ĵ�-dy0�е����ݳ���2������sad.
//		// sad�����в����ˣ�
//		for (d = 0; d < ndisp; d++)
//			sad[d] = (int)(hsad0[d - ndisp * dy0] * (wsz2 + 2 - dy0));
//
//		// ��hsadָ��hsad0�ĵ�1-dy0�У�ѭ��Ҳ��1-dy0�п�ʼ������ֻ�����ڴ�С�ڵ����ݣ���wsz2 - 1Ϊֹ����
//		// ������wsz2�к�֮ǰ������wsz2�е�ԭ����һ���ġ�
//		hsad = hsad0 + (1 - dy0)*ndisp;
//		for (y = 1 - dy0; y < wsz2; y++, hsad += ndisp)
//		{
//			d = 0;
//
//			// cost aggregation ����
//			// �ۼӲ�ͬ�С�һ�����������ڸ�������ȡ��ͬd ʱ��SAD��
//			for (; d < ndisp; d++)
//				sad[d] = (int)(sad[d] + hsad[d]);
//		}
//		// ѭ���ۼ�һ�����������ڵ�����ֵ��
//		int tsum = 0;
//		for (y = -wsz2 - 1; y < wsz2; y++)
//			tsum += htext[y];
//
//		// finally, start the real processing
//		// ��Ȼ�ٷ�ע��˵���ڲſ�ʼ�����Ĵ�����֮ǰ�Ѿ����˴����Ĵ������ˡ�
//		// minsad,sda ��Сֵ��mind����С�Ӳ�ֵ��
//		for (y = 0; y < height; y++)
//		{
//			int minsad = INT_MAX, mind = -1;
//			hsad = hsad0 + MIN(y + wsz2, height + dy1 - 1)*ndisp; // ��ǰ���ڵ����һ�С�
//			hsad_sub = hsad0 + MAX(y - wsz2 - 1, -dy0)*ndisp; // �ϸ����ڵ����һ�С�
//			d = 0;
//
//			// Ѱ�������Ӳ
//			for (; d < ndisp; d++)
//			{
//				int currsad = sad[d] + hsad[d] - hsad_sub[d]; // ͬ�ϣ��������һ�е�SAD����ȥ������һ�е�SAD.
//								  // ֮ǰ��sad��ֵʱΪ��Ҫ����2Ҳ������ˡ�һ����Ϊ��ʹ�����һ�����ڵ�SAD֮��ʱ��֮��Ĵ�����ͬ��
//								  // ���Լ�����һ�е�SAD��������һ�е�SAD�����Ա������2��ֹ�����һ��������©���˵�һ�С�
//
//				sad[d] = currsad; // ���µ�ǰd�µ�SAD֮�ͣ������´μ���ʹ�á�
//				if (currsad < minsad)
//				{
//					// �õ��Ӳ���Сֵ����С�Ӳ�λ��
//					minsad = currsad;
//					mind = d;
//				}
//			}
//
//			tsum += htext[y + wsz2] - htext[y - wsz2 - 1]; // ͬ����Ҫ��������ֵ��
//			// ���һ�����ظ���������̫�������Ӳ������Ϊ��Ч��
//			if (tsum < textureThreshold)
//			{
//				dptr[y*dstep] = FILTERED;
//				continue;
//			}
//
//			// Ψһ��ƥ�䡣
//			// ����ǰ���ҵ��������Ӳ�mind������SAD minsad������Ӧ��ֵΪminsad * (1 + uniquenessRatio).
//			// Ҫ�����mind ǰ��һ���Ӳ�֮�⣬������Ӳ��SAD���������ֵ�󣬷�����Ϊ�ҵ����Ӳ���Ч��
//			// continue���ֻ��������ѭ����������ֹ����ѭ����ִ�С���break������ǽ�������ѭ�����̣������ж�ִ��ѭ���������Ƿ����
//			if (uniquenessRatio > 0)
//			{
//				int thresh = minsad + (minsad * uniquenessRatio / 100);
//				// ��仰��������ʲô�� �õ�d����Ч�Ļ���d Ӧ�õ��� ndisp
//				for (d = 0; d < ndisp; d++)
//				{
//					// break ������ǰ for��foreach��while��do-while ���� switch �ṹ��ִ�С�
//					if ((d < mind - 1 || d > mind + 1) && sad[d] <= thresh)
//						break;
//				}
//				// �����dС��ndisp,��ʾ uniquenessRatio
//				if (d < ndisp)
//				{
//					// ��������ѭ��,������ѭ����������δִ�е����,���Ž�����һ���Ƿ�ִ��ѭ�����ж�.
//					dptr[y*dstep] = FILTERED;
//					continue;
//				}
//			}
//			// �����d < ndisp,dptr[]��עΪʧ�ܣ�����
//			{
//				// ��󣬾������У�飬����ȷ���˵�ǰ���ص��Ӳ
//				// �ع�֮ǰsadָ����ȷ����ָ��λ�ú�ָ��Ĵ�Сʱ��ǰ������һ��λ�ã��������õ��ˡ�
//				sad[-1] = sad[1];
//				sad[ndisp] = sad[ndisp - 2];
//				// �Ӳ��Ż�
//				// ����������λ�õ����þͺ������ˣ���ֹmindΪ0��ndis-1ʱ������������Խ�硣
//				// p��sad��Сֵ�ĺ�һ��λ�ã�n��ǰһ��λ�ã������dΪ p+n-2d+(p-n),�ж�d�Ƿ�Ϊ0�������Ϊ0�ǶԳƵģ���Ϊ�㣬��һ������ƫ����
//				int p = sad[mind + 1], n = sad[mind - 1];
//				d = p + n - 2 * sad[mind] + std::abs(p - n);
//				//  ע�⵽ǰ�潫dptr��λ�ü�����lofsλ�����������±�Ϊy * dstep��
//				dptr[y*dstep] = (mType)(((ndisp - mind - 1 + mindisp) * 256 + (d != 0 ? (p - n) * 256 / d : 0) + 15) // ��������������ģ��ᷢ��֮ǰ�����Ӳ�dʱ���������Ƿ������ġ�
//														 // ��d=0ʱ����������ͼ����Ӧ���Ǻ���ͼ������ͬ��x���꣬
//														 // ����ʵ֮ǰ������rptr�ǣ���ʱ��ͼ���ص�x����Ϊx-(ndisp-1)��
//														 // �������������Ӳ�Ҫ��ת������Ϊndisp-mind-1��
//														 // ����15����ΪopencvĬ���������Ϊ16λ����������Ϊ�˻���������Ӳ�Ҫ����16��
//														 // ����ӵ�һ������������ͳ����ضϵ�һ��������
//														 // ����Ϊ�ζ���һ��(p-n)/d����Ҳ��̫����Ӧ��������������SAD�ı仯�ʵ�һ��������ϣ�����˿���ָ����:)
//					>> (DISPARITY_SHIFT_32S - disp_shift));
//				costptr[y*coststep] = sad[mind]; // ���opencvĬ�ϵõ����Ӳ�ֵ��Ҫ����16������ǰ�����256������������4λ��
//			}
//		} // y 
//	}// x
//}
//
//
