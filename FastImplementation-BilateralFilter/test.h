#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

// experimental function
namespace grf
{
	void exTest_GRF_AVX512();
	void exTest_GRF_AVX();
	void exTest_GRF_SSE();

	void ex_impls_GRF_AVX512(const double sigma_range, const int r, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_GRF_AVX(const double sigma_range, const int r, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_GRF_SSE(const double sigma_range, const int r, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_GRF_Scalar(const double sigma_range, const int r, const std::string fname, const int datatype, const unsigned int loop);
}

namespace bf
{
	void exTest_BF_AVX512();
	void exTest_BF_AVX();
	void exTest_BF_SSE();

	void ex_impls_BF_AVX512(const double sigma_range, const double sigma_space, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_BF_AVX(const double sigma_range, const double sigma_space, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_BF_SSE(const double sigma_range, const double sigma_space, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_BF_Scalar(const double sigma_range, const double sigma_space, const std::string fname, const int datatype, const unsigned int loop);
}

namespace nlmf
{
	void exTest_NLMF_AVX512();
	void exTest_NLMF_AVX();
	void exTest_NLMF_SSE();

	void ex_impls_NLMF_Scalar(const cv::Size templateWindowSize, const cv::Size searchWindowSize, const double h, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_NLMF_AVX512(const cv::Size templateWindowSize, const cv::Size searchWindowSize, const double h, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_NLMF_AVX(const cv::Size templateWindowSize, const cv::Size searchWindowSize, const double h, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_NLMF_SSE(const cv::Size templateWindowSize, const cv::Size searchWindowSize, const double h, const std::string fname, const int datatype, const unsigned int loop);
}

namespace blnlmf
{
	void exTest_BNLMF_AVX512();
	void exTest_BNLMF_AVX();
	void exTest_BNLMF_SSE();

	void ex_impls_BNLMF_Scalar(const cv::Size templateWindowSize, const cv::Size searchWindowSize, const double h, const double sigma_space, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_BNLMF_AVX512(const cv::Size templateWindowSize, const cv::Size searchWindowSize, const double h, const double sigma_space, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_BNLMF_AVX(const cv::Size templateWindowSize, const cv::Size searchWindowSize, const double h, const double sigma_space, const std::string fname, const int datatype, const unsigned int loop);
	void ex_impls_BNLMF_SSE(const cv::Size templateWindowSize, const cv::Size searchWindowSize, const double h, const double sigma_space, const std::string fname, const int datatype, const unsigned int loop);
}


void benchmark_filters();

// for debug
void printRegister(const __m256i m, const int bdepth);
void printRegister(const __m256 m);
void printRegister(const __m256d m);
void printRegister(const __m128i m, const int bdepth);
void printRegister(const __m128 m);
void printRegister(const __m128d m);

// expand image to only horizontal direction
void myCopyMakeBorder_H_32F(cv::Mat& img, const int& left, const int& right);
void myCopyMakeBorder_H_8U(cv::Mat& img, const int& left, const int& right);

// expand image to only vertical direction
void myCopyMakeBorder_V_32F(cv::Mat& img, const int& top, const int& bottom);
void myCopyMakeBorder_V_8U(cv::Mat& img, const int& top, const int& bottom);

// calculate mean absolute error (MAE)
double calcMAE(const cv::Mat& src1_, const cv::Mat& src2_);

// calculate mean squared error (MSE)
double calcMSE(const cv::Mat& src1_, const cv::Mat& src2_);

// calculate peak signal-to-noise ratio (PSNR)
double calcPSNR(const cv::Mat& src1, const cv::Mat& src2);

enum
{
	TIME_AUTO = 0,
	TIME_NSEC,
	TIME_MSEC,
	TIME_SEC,
	TIME_MIN,
	TIME_HOUR,
	TIME_DAY
};

class CalcTime
{
	int64 pre;
	std::string mes;

	int timeMode;

	double cTime;
	bool _isShow;

	int autoMode;
	int autoTimeMode() const;
	std::vector<std::string> lap_mes;
public:

	void start();
	void setMode(const int mode);
	void setMessage(const std::string& src);
	void restart();
	double getTime();
	void show();
	void show(const std::string message);
	void lap(const std::string message);
	void init(const std::string message, const int mode, const bool isShow);

	CalcTime(const std::string message, const int mode = TIME_AUTO, const bool isShow = true);
	CalcTime(const char* message, const int mode = TIME_AUTO, const bool isShow = true);
	CalcTime();

	~CalcTime();
	std::string getTimeString();
};

class Stat
{
public:
	std::vector<double> data;
	Stat();
	~Stat();
	int num_data() const;
	double getMin();
	double getMax();
	double getMean();
	double getStd();
	double getMedian();

	void push_back(const double val);

	void clear();
	void show();
};

class ConsoleImage
{
private:
	int count;
	std::string windowName;
	std::vector<std::string> strings;
	bool isLineNumber;
public:
	void setIsLineNumber(bool isLine = true);
	bool getIsLineNumber() const;
	cv::Mat show;

	void init(const cv::Size size, const std::string wname);
	ConsoleImage();
	ConsoleImage(const cv::Size size, const std::string wname = "console");
	~ConsoleImage();

	void printData();
	void clear();

	void operator()(const std::string src);
	void operator()(const char *format, ...);
	void operator()(const cv::Scalar color, const char *format, ...);

	void flush(const bool isClear = true);
};