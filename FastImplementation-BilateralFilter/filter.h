#pragma once

#pragma warning(disable:4309)

#include <iomanip>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#define sqr(x) ((x) * (x))
#define sign(x) ((x > 0) - (x < 0))

// filter core
void setSpaceKernel(int* space_ofs, int& maxk, const int radiusH, const int radiusV, const int imstep, const bool isRectangle);
//double
void setSpaceKernel(double* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle, const bool isSpaceWeight);
void setSpaceKernel_expArgClip(double* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle, const double expClipVal, const double bias);
void setSpaceKernel_denormalSuppression(double* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle);
//float
void setSpaceKernel(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const float gauss_space_coeff, const int imstep, const bool isRectangle, const bool isSpaceWeight);
void setSpaceKernel_expArgClip(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const float gauss_space_coeff, const int imstep, const bool isRectangle, const float expClipVal, const float bias);
void setSpaceKernel_denormalSuppression(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const float gauss_space_coeff, const int imstep, const bool isRectangle);


void showMatInfo(const cv::InputArray src, const std::string name = "Mat: ");

// kernel shape
enum
{
	FILTER_RECTANGLE = 0,
	FILTER_CIRCLE,			//TODO: not support
	FILTER_SEPARABLE,			//TODO: not support
};

//Gaussian range flter
namespace grf
{
	enum WEIGHT_MODE
	{
		WEIGHT_VECTOR_EXP = 0,
		WEIGHT_RANGE_LUT_SET,			
		WEIGHT_RANGE_QUANTIZATION_LUT_SET,
		WEIGHT_RANGE_LUT_GATHER,
		WEIGHT_RANGE_QUANTIZATION_LUT_GATHER,
	};

	// GRF base implementation
	void gaussianRangeFilter_Scalar(cv::Mat& src, cv::Mat& dest, const int radius, const double sigma_range, const int borderType, const bool isParallel = true);
	// GRF SSE implementation
	void gaussianRangeFilter_SSE(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, const double sigma_range, const int kernel_type = FILTER_RECTANGLE, const int borderType = cv::BORDER_REPLICATE, const grf::WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
	// GRF AVX2 implementation
	void gaussianRangeFilter_AVX(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, const double sigma_range, const int kernel_type = FILTER_RECTANGLE, const int borderType = cv::BORDER_REPLICATE, const grf::WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
	// GRF AVX512 implementation
	void gaussianRangeFilter_AVX512(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, const double sigma_range, const int kernel_type = FILTER_RECTANGLE, const int borderType = cv::BORDER_REPLICATE, const grf::WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
}

//bilateral flter
namespace bf
{
	enum WEIGHT_MODE
	{
		WEIGHT_VECTOR_EXP = 0,
		WEIGHT_VECTOR_EXP_WITH_SPACE_LUT,
		WEIGHT_RANGE_LUT_SET,					
		WEIGHT_RANGE_QUANTIZATION_LUT_SET,
		WEIGHT_MERGED_QUANTIZATION_LUT_SET,
		WEIGHT_RANGE_LUT_GATHER,
		WEIGHT_RANGE_QUANTIZATION_LUT_GATHER,
		WEIGHT_MERGER_QUANTIZATION_LUT_GATHER,
		WEIGHT_RANGE_QUANTIZATION_LUT_GATHER_KAHAN,//only supported in 32F (grayscale image)
	};

	// BF base implementation
	void bilateralFilter_Scalar(cv::Mat& src, cv::Mat& dest, const int radius, const double sigma_range, const double sigma_space, const int borderType, const int parallelNum = -1);
	// BF SSE implementation
	void bilateralFilter_SSE(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, const double sigma_range, const double sigma_space, const int kernel_type = FILTER_RECTANGLE, const int borderType = cv::BORDER_REPLICATE, const bf::WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
	// BF AVX2 implementation
	void bilateralFilter_AVX(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, const double sigma_range, const double sigma_space, const int kernel_type = FILTER_RECTANGLE, const int borderType = cv::BORDER_REPLICATE, const bf::WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
	// BF AVX512 implementation
	void bilateralFilter_AVX512(cv::InputArray src, cv::OutputArray dest, cv::Size kernelSize, const double sigma_range, const double sigma_space, const int kernel_type = FILTER_RECTANGLE, const int borderType = cv::BORDER_REPLICATE, const bf::WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
}

//non local means filter
namespace nlmf
{
	enum WEIGHT_MODE
	{
		WEIGHT_VECTOR_EXP = 0,
		WEIGHT_RANGE_LUT_SET,
		WEIGHT_RANGE_QUANTIZATION_LUT_SETxN,
		WEIGHT_RANGE_QUANTIZATION_LUT_SETx1,
		WEIGHT_RANGE_LUT_GATHER,
		WEIGHT_RANGE_QUANTIZATION_LUT_GATHERxN,
		WEIGHT_RANGE_QUANTIZATION_LUT_GATHERx1,
	};

	void nonLocalMeansFilter_Scalar(const cv::Mat& src, cv::Mat& dest, const cv::Size templeteWindowSize, const cv::Size searchWindowSize, const double h, const int borderType, const bool isParallel = true);
	void nonLocalMeansFilter_SSE(cv::InputArray src, cv::OutputArray dest, const cv::Size templeteWindowSize, const cv::Size searchWindowSize, const double h, const int borderType, const WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
	void nonLocalMeansFilter_AVX(cv::InputArray src, cv::OutputArray dest, const cv::Size templeteWindowSize, const cv::Size searchWindowSize, const double h, const int borderType, const WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
	void nonLocalMeansFilter_AVX512(cv::InputArray src, cv::OutputArray dest, const cv::Size templeteWindowSize, const cv::Size searchWindowSize, const double h, const int borderType, const WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
}

//bilateral non local means filter
namespace blnlmf
{
	enum WEIGHT_MODE
	{
		WEIGHT_VECTOR_EXP = 0,
		WEIGHT_VECTOR_EXP_WITH_SPACE_LUT,
		WEIGHT_RANGE_LUT_SET,
		WEIGHT_RANGE_QUANTIZATION_LUT_SETxN,
		WEIGHT_RANGE_QUANTIZATION_LUT_SETx1,
		WEIGHT_MERGED_QUANTIZATION_LUT_SETx1,
		WEIGHT_RANGE_LUT_GATHER,
		WEIGHT_RANGE_QUANTIZATION_LUT_GATHERxN,
		WEIGHT_RANGE_QUANTIZATION_LUT_GATHERx1,
		WEIGHT_MERGED_QUANTIZATION_LUT_GATHERx1,
	};

	void bilateralNonLocalMeansFilter_Scalar(const cv::Mat& src, cv::Mat& dest, const cv::Size templeteWindowSize, const cv::Size searchWindowSize, const double h, const double sigma_space, const int borderType, const bool isParallel = true);
	void bilateralNonLocalMeansFilter_SSE(cv::InputArray src, cv::OutputArray dest, const cv::Size templeteWindowSize, const cv::Size searchWindowSize, const double h, const double sigma_space, const int borderType, const WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
	void bilateralNonLocalMeansFilter_AVX(cv::InputArray src, cv::OutputArray dest, const cv::Size templeteWindowSize, const cv::Size searchWindowSize, const double h, const double sigma_space, const int borderType, const WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
	void bilateralNonLocalMeansFilter_AVX512(cv::InputArray src, cv::OutputArray dest, const cv::Size templeteWindowSize, const cv::Size searchWindowSize, const double h, const double sigma_space, const int borderType, const WEIGHT_MODE weightingMethod = WEIGHT_VECTOR_EXP);
}
