#include "test.h"

using namespace std;
using namespace cv;

void myCopyMakeBorder_H_32F(Mat& img, const int& left, const int& right)
{
	const float* sptr = (float*)img.ptr(0) + left;
	float* dptr = (float*)img.ptr(0);
	int dstep = img.cols;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < left; j++)
		{
			memcpy(dptr, sptr, sizeof(float) * 1);
			dptr++;
		}

		dptr += img.cols - (left + right);
		sptr += img.cols - (left + right + 1);

		for (int j = 0; j < right; j++)
		{
			memcpy(dptr, sptr, sizeof(float) * 1);
			dptr++;
		}

		sptr += (left + right + 1);
	}
}

void myCopyMakeBorder_H_8U(Mat& img, const int& left, const int& right)
{
	const uchar* sptr = (uchar*)img.ptr(0) + left;
	uchar* dptr = (uchar*)img.ptr(0);
	int dstep = img.cols;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < left; j++)
		{
			memcpy(dptr, sptr, sizeof(uchar) * 1);
			dptr++;
		}

		dptr += img.cols - (left + right);
		sptr += img.cols - (left + right + 1);

		for (int j = 0; j < right; j++)
		{
			memcpy(dptr, sptr, sizeof(uchar) * 1);
			dptr++;
		}

		sptr += (left + right + 1);
	}
}

void myCopyMakeBorder_V_32F(Mat& img, const int& top, const int& bottom)
{
	const float* sptr = (float*)img.ptr(top);
	float* dptr = (float*)img.ptr(0);
	int dstep = img.cols;

	for (int i = 0; i < top; i++)
	{
		memcpy(dptr, sptr, sizeof(float) * dstep);
		dptr += dstep;
	}

	{
		dptr += (img.rows - top - bottom) * dstep;
		sptr += (img.rows - top - bottom - 1) * dstep;
	}

	for (int i = 0; i < bottom; i++)
	{
		memcpy(dptr, sptr, sizeof(float) * dstep);
		dptr += dstep;
	}
}

void myCopyMakeBorder_V_8U(Mat& img, const int& top, const int& bottom)
{
	const uchar* sptr = (uchar*)img.ptr(top);
	uchar* dptr = (uchar*)img.ptr(0);
	int dstep = img.cols;

	for (int i = 0; i < top; i++)
	{
		memcpy(dptr, sptr, sizeof(uchar) * dstep);
		dptr += dstep;
	}

	{
		dptr += (img.rows - top - bottom) * dstep;
		sptr += (img.rows - top - bottom - 1) * dstep;
	}

	for (int i = 0; i < bottom; i++)
	{
		memcpy(dptr, sptr, sizeof(uchar) * dstep);
		dptr += dstep;
	}
}

double calcMAE(const Mat& src1_, const Mat& src2_)
{
	CV_Assert(src1_.size() == src2_.size() && src1_.channels() == src2_.channels());
	int ch = src1_.channels();

	Mat src1, src2;
	src1_.convertTo(src1, CV_64F);
	src2_.convertTo(src2, CV_64F);

	Mat diff;
	absdiff(src1, src2, diff);

	Scalar s = sum(diff);

	double sae = 0;
	if (ch == 3)
		sae = s.val[0] + s.val[1] + s.val[2];
	else
		sae = s.val[0];

	double  mae = sae / (double)(ch * src1.total());

	return mae;
}

double calcMSE(const Mat& src1_, const Mat& src2_)
{
	CV_Assert(src1_.size() == src2_.size() && src1_.channels() == src2_.channels());
	int ch = src1_.channels();

	Mat src1, src2;
	src1_.convertTo(src1, CV_64F);
	src2_.convertTo(src2, CV_64F);

	Mat diff;
	absdiff(src1, src2, diff);
	diff = diff.mul(diff);

	Scalar s = sum(diff);

	double sse = 0;
	if (ch == 3)
		sse = s.val[0] + s.val[1] + s.val[2];
	else
		sse = s.val[0];

	double  mse = sse / (double)(ch * src1.total());

	return mse;
}

double calcPSNR(const Mat& src1_, const Mat& src2_)
{
	CV_Assert(src1_.size() == src2_.size() && src1_.channels() == src2_.channels());
	const int ch = src1_.channels();

	Mat src1, src2;
	src1_.convertTo(src1, CV_64F);
	src2_.convertTo(src2, CV_64F);

	Mat diff;
	absdiff(src1, src2, diff);
	diff = diff.mul(diff);

	Scalar s = sum(diff);

	double sse = 0;
	if (ch == 3)
	{
		sse = s.val[0] + s.val[1] + s.val[2];
	}
	else
	{
		sse = s.val[0];
	}

	if (sse <= 1e-10)
	{
		return INFINITY;
	}
	else
	{
		const double  mse = sse / (double)(ch * src1.total());
		return 10.0 * log10((255.0 * 255.0) / mse);
	}
}

void CalcTime::start()
{
	pre = getTickCount();
}

void CalcTime::restart()
{
	start();
}

void CalcTime::lap(const string message)
{
	string v = message + format(" %f", getTime());
	switch (timeMode)
	{
	case TIME_NSEC:
		v += " NSEC";
		break;
	case TIME_SEC:
		v += " SEC";
		break;
	case TIME_MIN:
		v += " MIN";
		break;
	case TIME_HOUR:
		v += " HOUR";
		break;

	case TIME_MSEC:
	default:
		v += " msec";
		break;
	}

	lap_mes.push_back(v);
	restart();
}


string CalcTime::getTimeString()
{
	string ret;
	getTime();

	int mode = timeMode;
	if (timeMode == TIME_AUTO)
	{
		mode = autoMode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		ret = mes + format(": %f nsec", cTime);
		break;
	case TIME_SEC:
		ret = mes + format(": %f sec", cTime);
		break;
	case TIME_MIN:
		ret = mes + format(": %f minute", cTime);
		break;
	case TIME_HOUR:
		ret = mes + format(": %f hour", cTime);
		break;
	case TIME_MSEC:
	default:
		ret = mes + format(": %f msec", cTime);
		break;
	}

	return ret;
}

void CalcTime::show()
{
	getTime();

	int mode = timeMode;
	if (timeMode == TIME_AUTO)
	{
		mode = autoMode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cout << mes << ": " << cTime << " nsec" << endl;
		break;
	case TIME_SEC:
		cout << mes << ": " << cTime << " sec" << endl;
		break;
	case TIME_MIN:
		cout << mes << ": " << cTime << " minute" << endl;
		break;
	case TIME_HOUR:
		cout << mes << ": " << cTime << " hour" << endl;
		break;
	case TIME_MSEC:
	default:
		cout << mes << ": " << cTime << " msec" << endl;
		break;
	}
}

void CalcTime::show(const string mes)
{
	getTime();
	int mode = timeMode;
	if (timeMode == TIME_AUTO)
	{
		mode = autoMode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cout << mes << ": " << cTime << " nsec" << endl;
		break;
	case TIME_SEC:
		cout << mes << ": " << cTime << " sec" << endl;
		break;
	case TIME_MIN:
		cout << mes << ": " << cTime << " minute" << endl;
		break;
	case TIME_HOUR:
		cout << mes << ": " << cTime << " hour" << endl;
		break;
	case TIME_DAY:
		cout << mes << ": " << cTime << " day" << endl;
	case TIME_MSEC:
		cout << mes << ": " << cTime << " msec" << endl;
		break;
	default:
		cout << mes << ": error" << endl;
		break;
	}
}

int CalcTime::autoTimeMode() const
{
	if (cTime > 60.0*60.0*24.0)
	{
		return TIME_DAY;
	}
	else if (cTime > 60.0*60.0)
	{
		return TIME_HOUR;
	}
	else if (cTime > 60.0)
	{
		return TIME_MIN;
	}
	else if (cTime > 1.0)
	{
		return TIME_SEC;
	}
	else if (cTime > 1.0 / 1000.0)
	{
		return TIME_MSEC;
	}
	else
	{

		return TIME_NSEC;
	}
}

double CalcTime::getTime()
{
	cTime = (getTickCount() - pre) / (getTickFrequency());

	int mode = timeMode;
	if (mode == TIME_AUTO)
	{
		mode = autoTimeMode();
		autoMode = mode;
	}

	switch (mode)
	{
	case TIME_NSEC:
		cTime *= 1000000.0;
		break;
	case TIME_SEC:
		cTime *= 1.0;
		break;
	case TIME_MIN:
		cTime /= (60.0);
		break;
	case TIME_HOUR:
		cTime /= (60 * 60);
		break;
	case TIME_DAY:
		cTime /= (60 * 60 * 24);
		break;
	case TIME_MSEC:
	default:
		cTime *= 1000.0;
		break;
	}
	return cTime;
}

void CalcTime::setMessage(const string& src)
{
	mes = src;
}

void CalcTime::setMode(const int mode)
{
	timeMode = mode;
}

void CalcTime::init(const string message, const int mode, const bool isShow)
{
	_isShow = isShow;
	timeMode = mode;

	setMessage(message);
	start();
}

CalcTime::CalcTime()
{
	const string t = "time ";
	init(t, TIME_AUTO, true);
}

CalcTime::CalcTime(const char* message, const int mode, const bool isShow)
{
	const string m = message;
	init(m, mode, isShow);
}

CalcTime::CalcTime(const string message, const int mode, const bool isShow)
{
	init(message, mode, isShow);
}

CalcTime::~CalcTime()
{
	getTime();
	if (_isShow)	show();
	if (lap_mes.size() != 0)
	{
		for (int i = 0; i < lap_mes.size(); i++)
		{
			cout << lap_mes[i] << endl;
		}
	}
}

// avx
void printRegister(const __m256i m, const int bdepth)
{
	if (bdepth == 8)
	{
		uchar* p = (uchar*)&m;
		for (int i = 0; i < 32; i++)
		{
			cout << (int)p[i];
			if (i != 31)
				cout << ", ";
		}
		cout << endl;
	}
	else if (bdepth == 16)
	{
		short* p = (short*)&m;
		for (int i = 0; i < 16; i++)
		{
			cout << p[i];
			if (i != 15)
				cout << ", ";
		}
		cout << endl;
	}
	else if (bdepth == 32)
	{
		int* p = (int*)&m;
		for (int i = 0; i < 8; i++)
		{
			cout << p[i];
			if (i != 7)
				cout << ", ";
		}
		cout << endl;
	}
}

void printRegister(const __m256 m)
{
	float* p = (float*)&m;
	for (int i = 0; i < 8; i++)
	{
		cout << p[i];
		if (i != 7)
			cout << ", ";
	}
	cout << endl;
}

void printRegister(const __m256d m)
{
	double* p = (double*)&m;
	for (int i = 0; i < 4; i++)
	{
		cout << p[i];
		if (i != 3)
			cout << ", ";
	}
	cout << endl;
}

// sse
void printRegister(const __m128i m, const int bdepth)
{
	if (bdepth == 8)
	{
		uchar* p = (uchar*)&m;
		for (int i = 0; i < 16; i++)
		{
			cout << (int)p[i];
			if (i != 15)
				cout << ", ";
		}
		cout << endl;
	}
	else if (bdepth == 16)
	{
		short* p = (short*)&m;
		for (int i = 0; i < 8; i++)
		{
			cout << p[i];
			if (i != 7)
				cout << ", ";
		}
		cout << endl;
	}
	else if (bdepth == 32)
	{
		int* p = (int*)&m;
		for (int i = 0; i < 4; i++)
		{
			cout << p[i];
			if (i != 3)
				cout << ", ";
		}
		cout << endl;
	}
}

void printRegister(const __m128 m)
{
	float* p = (float*)&m;
	for (int i = 0; i < 4; i++)
	{
		cout << p[i];
		if (i != 3)
			cout << ", ";
	}
	cout << endl;
}

void printRegister(const __m128d m)
{
	double* p = (double*)&m;
	for (int i = 0; i < 2; i++)
	{
		cout << p[i];
		if (i != 1)
			cout << ", ";
	}
	cout << endl;
}

void showMatInfo(const InputArray src_, const string name)
{
	Mat src = src_.getMat();

	cout << name << ":" << endl;
	if (src.empty())
	{
		cout << "empty" << endl;
		return;
	}
	cout << "size   : " << src.size() << endl;
	cout << "channel: " << src.channels() << endl;
	cout << "depth  : ";
	if (src.depth() == CV_8U)cout << "8U" << endl;
	else if (src.depth() == CV_16S)cout << "16S" << endl;
	else if (src.depth() == CV_16U)cout << "16U" << endl;
	else if (src.depth() == CV_32S)cout << "32S" << endl;
	else if (src.depth() == CV_32F)cout << "32F" << endl;
	else if (src.depth() == CV_64F)cout << "64F" << endl;

	if (src.channels() == 1)
	{
		Scalar v = mean(src);
		cout << "mean  : " << v.val[0] << endl;
		double minv, maxv;
		minMaxLoc(src, &minv, &maxv);
		cout << "minmax: " << minv << "," << maxv << endl;
	}
	else if (src.channels() == 3)
	{
		Scalar v = mean(src);
		cout << "mean  : " << v.val[0] << "," << v.val[1] << "," << v.val[2] << endl;

		vector<Mat> vv;
		split(src, vv);
		double minv, maxv;
		minMaxLoc(vv[0], &minv, &maxv);
		cout << "minmax0: " << minv << "," << maxv << endl;
		minMaxLoc(vv[1], &minv, &maxv);
		cout << "minmax1: " << minv << "," << maxv << endl;
		minMaxLoc(vv[2], &minv, &maxv);
		cout << "minmax2: " << minv << "," << maxv << endl;
	}
}

Stat::Stat() { ; }
Stat::~Stat() { ; }

int Stat::num_data() const
{
	return data.size();
}

double Stat::getMin()
{
	double minv = DBL_MAX;
	for (int i = 0; i < data.size(); i++)
	{
		minv = min(minv, data[i]);
	}
	return minv;
}

double Stat::getMax()
{
	double maxv = DBL_MIN;
	for (int i = 0; i < data.size(); i++)
	{
		maxv = max(maxv, data[i]);
	}
	return maxv;
}

double Stat::getMean()
{
	double sum = 0.0;
	for (int i = 0; i < data.size(); i++)
	{
		sum += data[i];
	}
	return sum / (double)data.size();
}

double Stat::getStd()
{
	double std = 0.0;
	double mean = getMean();
	for (int i = 0; i < data.size(); i++)
	{
		std += (mean - data[i])*(mean - data[i]);
	}
	return sqrt(std / (double)data.size());
}

double Stat::getMedian()
{
	if (data.size() == 0) return 0.0;
	vector<double> v;
	vector<double> s;
	for (int i = 0; i < data.size(); i++)
	{
		s.push_back(data[i]);
	}
	cv::sort(s, v, cv::SORT_ASCENDING);
	return v[data.size() / 2];
}

void Stat::push_back(const double val)
{
	data.push_back(val);
}

void Stat::clear()
{
	data.clear();
}

void Stat::show()
{
	cout << "mean " << getMean() << endl;
	cout << "min  " << getMin() << endl;
	cout << "med  " << getMedian() << endl;
	cout << "max  " << getMax() << endl;
	cout << "std  " << getStd() << endl;
}

void ConsoleImage::init(const Size size, const string wname)
{
	isLineNumber = false;
	windowName = wname;
	show = Mat::zeros(size, CV_8UC3);
	clear();
}
ConsoleImage::ConsoleImage()
{
	init(Size(640, 480), "console");
}
ConsoleImage::ConsoleImage(const Size size, const string wname)
{
	init(size, wname);
}
ConsoleImage::~ConsoleImage()
{
	printData();
}
void ConsoleImage::setIsLineNumber(const bool isLine)
{
	isLineNumber = isLine;
}

bool ConsoleImage::getIsLineNumber() const
{
	return isLineNumber;
}
void ConsoleImage::printData()
{
	for (int i = 0; i < (int)strings.size(); i++)
	{
		cout << strings[i] << endl;
	}
}
void ConsoleImage::clear()
{
	count = 0;
	show.setTo(0);
	strings.clear();
}
void ConsoleImage::flush(const bool isClear)
{
	imshow(windowName, show);
	if (isClear)clear();
}
void ConsoleImage::operator()(const string src)
{
	if (isLineNumber)strings.push_back(format("%2d ", count) + src);
	else strings.push_back(src);

	cv::putText(show, src, Point(20, 20 + count * 20), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}
void ConsoleImage::operator()(const char *format, ...)
{
	char buff[255];

	va_list ap;
	va_start(ap, format);
	vsprintf(buff, format, ap);
	va_end(ap);

	string a = buff;

	if (isLineNumber)strings.push_back(cv::format("%2d ", count) + a);
	else strings.push_back(a);

	cv::putText(show, buff, Point(20, 20 + count * 20), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}

void ConsoleImage::operator()(const cv::Scalar color, const char *format, ...)
{
	char buff[255];

	va_list ap;
	va_start(ap, format);
	vsprintf(buff, format, ap);
	va_end(ap);

	string a = buff;
	if (isLineNumber)strings.push_back(cv::format("%2d ", count) + a);
	else strings.push_back(a);
	cv::putText(show, buff, Point(20, 20 + count * 20), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(255, 255, 255), 1);
	count++;
}