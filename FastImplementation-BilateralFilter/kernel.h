#pragma once
#include <opencv2/opencv.hpp>

namespace kernel
{
	bool checkDenormal(const float v);

	void gaussianFilter(const cv::Mat src, cv::Mat& dest, cv::Mat& denormalMap, const int r, const float sigma_space);
	std::vector<float> getGaussianKernel_float(const cv::Mat src, cv::Mat& kernel, const cv::Point pos, const int r, const float sigma_space);
	void gaussian_kernel_test(const cv::Mat src);

	void gaussianRangeFilter(const cv::Mat src, cv::Mat& dest, cv::Mat& denormalMap, const int r, const float sigma_range);
	std::vector<float> getGaussianRangeKernel_float(const cv::Mat src, cv::Mat& kernel, const cv::Point pos, const int r, const float sigma_range);
	void gaussianRange_kernel_test(const cv::Mat src);

	void bilateralFilter(const cv::Mat src, cv::Mat& dest, cv::Mat& denormalMap, const int r, const float sigma_range, const float sigma_space);
	std::vector<float> getBilateralKernel_float(const cv::Mat src, cv::Mat& kernel, const cv::Point pos, const int r, const float sigma_range, const float sigma_space);
	void bilateral_kernel_test(const cv::Mat src);

	void nonLocalMeansFilter(const cv::Mat src, cv::Mat& dest, cv::Mat& denormalMap, const cv::Size templateWindowSize, const cv::Size searchWindowSize, const float h);
	std::vector<float> getNonLocalMeansKernel_float(const cv::Mat src, cv::Mat& kernel, const cv::Point pos, const cv::Size templateWindow, const cv::Size searchWindow, const float h);
	void nonLocalMeans_kernel_test(const cv::Mat src);


	void bilateralNonLocalMeansFilter(const cv::Mat src, cv::Mat& dest, cv::Mat& denormalMap, const cv::Size templateWindowSize, const cv::Size searchWindowSize, const float h, const float sigma_sapce);
	std::vector<float> getBilateralNonLocalMeansKernel_float(const cv::Mat src, cv::Mat& kernel, const cv::Point pos, const cv::Size templateWindow, const cv::Size searchWindow, const float h, const float sigma_sapce);
	void bilateralNonLocalMeans_kernel_test(const cv::Mat src);
}
