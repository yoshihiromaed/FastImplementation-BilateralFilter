#include "filter.h"
#include "test.h"

using namespace std;
using namespace cv;

namespace bf
{
	void exTest_BF_AVX()
	{
		string fname;
		fname = "lena.png";
		//fname = "lena256.png";
		//fname = "lena1024.png";
		//fname = "flower.png";
		//fname = "kodak/kodim04.png";

		const string wname = "image BF AVX/AVX2";
		namedWindow(wname);

		int ss = 10;
		int sr = 16;

		int data = 0;
		createTrackbar("data", wname, &data, 24);
		int impl = 7;
		createTrackbar("implementation", wname, &impl, 7);
		int datatype = 1;
		createTrackbar("8U, 32F, 64F", wname, &datatype, 2);
		int isGray = 1;
		createTrackbar("isGray", wname, &isGray, 1);

		createTrackbar("ss", wname, &ss, 32);
		createTrackbar("sr", wname, &sr, 128);
		int sigma = 5;  createTrackbar("sigma", wname, &sigma, 128);

		Mat src;
		if (isGray == 1)
		{
			src = imread(fname, 0);
		}
		else
		{
			src = imread(fname, 1);
		}

		Mat dest, ref;
		src.copyTo(dest);
		int r = cvRound(3 * ss);
		int d = 2 * r + 1;
#if CV_AVX || CV_AVX2
		//bilateralFilter_Scalar(src, ref, r, sr, ss, BORDER_REPLICATE);
		bilateralFilter_AVX(src, ref, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_VECTOR_EXP);
#endif

		int key = 0;
		Stat st;
		ConsoleImage ci(Size(640, 480), "BF AVX/AVX2");
		
		Mat noise(Size(512,512), CV_32F);
		cv::randn(noise, cv::Scalar(0.f), cv::Scalar(sigma));
		while (true)
		{
			
			//fname = format("kodak/kodim%02d.png",data);
			if (data == 0) fname = "img/lena.png";
			if (data == 1) fname = "img/lena256.png";
			if (data == 2) fname = "img/lena128.png";
			if (data == 3) fname = "img/lena64.png";
			if (data == 4) fname = "img/lena32.png";
			if (data == 5) fname = "img/lena1024.png";

			
			r = cvRound(3 * ss);
			d = 2 * r + 1;
			if (isGray == 1)
			{
				src = imread(fname, 0);
			}
			else
			{
				src = imread(fname, 1);
			}

			if (datatype == 0)
			{
				src.convertTo(src, CV_8U);
			}
			else if (datatype == 1)
			{
				src.convertTo(src, CV_32F);
			}
			else
			{
				src.convertTo(src, CV_64F);
			}


			
			
			Mat refsrc = src.clone();
			src = src + noise;
			if(key=='u')
				cv::randn(noise, cv::Scalar(0.f), cv::Scalar(sigma));	
#if CV_AVX || CV_AVX2
			//bilateralFilter_Scalar(src, ref, r, sr, ss, BORDER_REPLICATE);
			bilateralFilter_AVX(src, ref, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_VECTOR_EXP);
#endif

			dest = Mat::zeros(src.size(), src.type());
			// test SIMD implementation
			switch (impl)
			{
			case WEIGHT_VECTOR_EXP:
			{
				const string title = " EXP";
				ci((datatype == 0 ? "8U" : (datatype == 1 ? "32F" : "64F")) + title);
#if CV_AVX || CV_AVX2
				if (CV_AVX == 1 && CV_AVX2 == 0 && src.depth() == CV_8U)
				{
					ci("invalid impl");
				}
				else
				{
					CalcTime t;
					// normal bilateral filter
					bilateralFilter_AVX(src, dest, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_VECTOR_EXP);
					st.push_back(t.getTime());
					ci(t.getTimeString());
				}
#else
				ci("invalid impl");
#endif
				break;
			}
			case WEIGHT_VECTOR_EXP_WITH_SPACE_LUT:
			{
				const string title = " EXP with space LUT";
				ci((datatype == 0 ? "8U" : (datatype == 1 ? "32F" : "64F")) + title);
#if CV_AVX || CV_AVX2
				if (CV_AVX == 1 && CV_AVX2 == 0 && src.depth() == CV_8U)
				{
					ci("invalid impl");
				}
				else
				{
					CalcTime t;
					// normal bilateral filter
					bilateralFilter_AVX(src, dest, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_VECTOR_EXP_WITH_SPACE_LUT);
					st.push_back(t.getTime());
					ci(t.getTimeString());
				}
#else
				ci("invalid impl");
#endif
				break;
			}
			case WEIGHT_RANGE_LUT_SET:
			{
				const string title = " LUT set";
				ci((datatype == 0 ? "8U" : (datatype == 1 ? "32F" : "64F")) + title);
#if CV_AVX || CV_AVX2
				if (CV_AVX == 1 && CV_AVX2 == 0 && src.depth() == CV_8U)
				{
					ci("invalid impl");
				}
				else
				{
					CalcTime t;
					// normal bilateral filter
					bilateralFilter_AVX(src, dest, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_RANGE_LUT_SET);
					st.push_back(t.getTime());
					ci(t.getTimeString());
				}
#else
				ci("invalid impl");
#endif
				break;
			}
			case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
			{
				const string title = " quantization range LUT set";
				ci((datatype == 0 ? "8U" : (datatype == 1 ? "32F" : "64F")) + title);
#if CV_AVX || CV_AVX2
				if (CV_AVX == 1 && CV_AVX2 == 0 && src.depth() == CV_8U)
				{
					ci("invalid impl");
				}
				else
				{
					CalcTime t;
					// normal bilateral filter
					bilateralFilter_AVX(src, dest, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_RANGE_QUANTIZATION_LUT_SET);
					st.push_back(t.getTime());
					ci(t.getTimeString());
				}
#else
				ci("invalid impl");
#endif
				break;
			}
			case WEIGHT_MERGED_QUANTIZATION_LUT_SET:
			{
				const string title = " quantization LUT set";
				ci((datatype == 0 ? "8U" : (datatype == 1 ? "32F" : "64F")) + title);
#if CV_AVX || CV_AVX2
				if (CV_AVX == 1 && CV_AVX2 == 0 && src.depth() == CV_8U)
				{
					ci("invalid impl");
				}
				else
				{
					CalcTime t;
					// normal bilateral filter
					bilateralFilter_AVX(src, dest, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_MERGED_QUANTIZATION_LUT_SET);
					st.push_back(t.getTime());
					ci(t.getTimeString());
				}
#else
				ci("invalid impl");
#endif
				break;
			}
			case WEIGHT_RANGE_LUT_GATHER:
			{
				const string title = " LUT gather";
				ci((datatype == 0 ? "8U" : (datatype == 1 ? "32F" : "64F")) + title);
#if CV_AVX2
				CalcTime t;
				// normal bilateral filter
				bilateralFilter_AVX(src, dest, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_RANGE_LUT_GATHER);
				st.push_back(t.getTime());
				ci(t.getTimeString());
#else
				ci("invalid impl");
#endif
				break;
			}
			case WEIGHT_RANGE_QUANTIZATION_LUT_GATHER:
			{
				const string title = " quantization range LUT gather";
				ci((datatype == 0 ? "8U" : (datatype == 1 ? "32F" : "64F")) + title);
#if CV_AVX2
				CalcTime t;
				// normal bilateral filter
				bilateralFilter_AVX(src, dest, Size(d, d), (double)sr, (double)ss, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_RANGE_QUANTIZATION_LUT_GATHER);
				st.push_back(t.getTime());
				ci(t.getTimeString());
#else
				ci("invalid impl");
#endif
				break;
			}
			case WEIGHT_MERGER_QUANTIZATION_LUT_GATHER:
			{
				const string title = " quantization LUT gather";
				ci((datatype == 0 ? "8U" : (datatype == 1 ? "32F" : "64F")) + title);
#if CV_AVX2
				CalcTime t;
				// normal bilateral filter
				bilateralFilter_AVX(src, dest, Size(d, d), (double)sr, (double)ss*0.1, FILTER_RECTANGLE, BORDER_REPLICATE, WEIGHT_MERGER_QUANTIZATION_LUT_GATHER);
				st.push_back(t.getTime());
				ci(t.getTimeString());
#else
				ci("invalid impl");
#endif
				break;
			}
			default:
			{
				ci("invalid impl");
				break;
			}
			}

			ci(format("median: %f ", st.getMedian()));
			ci(format("num   : %d ", st.num_data()));
			if (!dest.empty() && !ref.empty())
			{
				ci(format("PSNR: %f ", calcPSNR(dest, refsrc)));
				ci(format("PSNR: %f ", calcPSNR(dest, ref)));
			}
			ci.flush();


			Mat show;
			dest.convertTo(show, CV_8U);
			imshow(wname, show);

			Mat show_ref;
			ref.convertTo(show_ref, CV_8U);
			imshow("ref", show_ref);

			const Mat show_diff(10 * (show - show_ref));
			imshow("diff", show_diff);


			key = waitKey(1);
			if (key == 'q')
			{
				break;
			}
			if (key == 'r')
			{
				system("clear");
				st.clear();
			}
		}
	}
}
