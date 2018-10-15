#include "test.h"
#include "filter.h"
#include "kernel.h"
#include "fmath.hpp"

#define CV_LIB_PREFIX comment(lib, "opencv_"

#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib")
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib")
#endif

#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX

#pragma CV_LIBRARY(core)
#pragma CV_LIBRARY(highgui)
#pragma CV_LIBRARY(imgcodecs)
#pragma CV_LIBRARY(imgproc)

using namespace std;
using namespace cv;


int main(const int argc, char **argv)
{
	//denormal status
	if (false)
	{
		//Mat src = imread("img/lena.png");
		Mat src = imread("img/kodak/kodim11.png");
		//kernel::gaussian_kernel_test(src);
		//kernel::gaussianRange_kernel_test(src);
		//destroyAllWindows();
		//waitKey(1);
		//kernel::bilateral_kernel_test(src);
		//destroyAllWindows();
		//waitKey(1);
		//kernel::nonLocalMeans_kernel_test(src);
		//destroyAllWindows();
		//waitKey(1);
		kernel::bilateralNonLocalMeans_kernel_test(src);
		//destroyAllWindows();
		//waitKey(1);
		return 0;
	}

	//Gaussian range filter
	if (false)
	{
		if (checkHardwareSupport(CV_CPU_AVX_512F))
		{
			grf::exTest_GRF_AVX512();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		if (checkHardwareSupport(CV_CPU_AVX2) || checkHardwareSupport(CV_CPU_AVX))
		{
			grf::exTest_GRF_AVX();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		if (checkHardwareSupport(CV_CPU_SSE4_1))
		{
			grf::exTest_GRF_SSE();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		return 0;
	}
	//bilateral filter
	//if (false)
	{
		if (checkHardwareSupport(CV_CPU_AVX_512F))
		{
			bf::exTest_BF_AVX512();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		if (checkHardwareSupport(CV_CPU_AVX2) || checkHardwareSupport(CV_CPU_AVX))
		{
			bf::exTest_BF_AVX();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		if (checkHardwareSupport(CV_CPU_SSE4_1))
		{
			bf::exTest_BF_SSE();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		return 0;
	}
	//non local means filter
	if (false)
	{
		if (checkHardwareSupport(CV_CPU_AVX_512F))
		{
			nlmf::exTest_NLMF_AVX512();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		if (checkHardwareSupport(CV_CPU_AVX2) || checkHardwareSupport(CV_CPU_AVX))
		{
			nlmf::exTest_NLMF_AVX();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		if (checkHardwareSupport(CV_CPU_SSE4_1))
		{
			nlmf::exTest_NLMF_SSE();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		return 0;
	}
	//bilateral local means filter
	if (false)
	{
		if (checkHardwareSupport(CV_CPU_AVX_512F))
		{
			blnlmf::exTest_BNLMF_AVX512();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		if (checkHardwareSupport(CV_CPU_AVX2) || checkHardwareSupport(CV_CPU_AVX))
		{
			blnlmf::exTest_BNLMF_AVX();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		if (checkHardwareSupport(CV_CPU_SSE4_1))
		{
			blnlmf::exTest_BNLMF_SSE();
			waitKey(5);
			destroyAllWindows();
			waitKey(5);
		}
		return 0;
	}
}
