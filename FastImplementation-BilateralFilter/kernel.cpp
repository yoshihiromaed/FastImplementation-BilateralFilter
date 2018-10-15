#include "kernel.h"
#include "test.h"
#include "filter.h"
#include <cmath>
#include "color.h"

using namespace cv;
using namespace std;


namespace kernel
{
	bool checkDenormal(const float v)
	{
		return fpclassify(v) == FP_SUBNORMAL;
	}

	/////////////////////////////
	// gaussian filter
	/////////////////////////////
	void gaussianFilter(const Mat src, Mat& dest, Mat& denormalMap, const int r, const float sigma_space)
	{
		Mat src_;
		copyMakeBorder(src, src_, r, r, r, r, BORDER_REPLICATE);
		denormalMap = Mat::zeros(src.size(), CV_8UC1);

		const int width = 2 * r + 1;
		const float coef = -1.f / (2 * sigma_space*sigma_space);

		if (src_.channels() == 1)
		{
			dest = Mat::zeros(src.size(), CV_32FC1);

#pragma omp parallel for
			for (int y = 0; y < src.rows; y++)
			{
				uchar* sptr = src_.ptr<uchar>(y + r) + r;
				float* dptr = dest.ptr<float>(y);
				uchar* denormal_ptr = denormalMap.ptr<uchar>(y);

				for (int x = 0; x < src.cols; x++)
				{
					uchar* sptrx = sptr + x;
					float* dptrx = dptr + x;
					uchar* denormal_ptrx = denormal_ptr + x;

					int count = 0;
					float sum = 0.f;
					float wsum = 0.f;
					for (int j = 0; j < width; j++)
					{
						for (int i = 0; i < width; i++)
						{
							const int x_ = i - r;
							const int y_ = j - r;
							const float d = float(x_ * x_ + y_ * y_);
							const float w = exp(float(d * coef));

							if (checkDenormal(w))
							{
								count++;
							}

							sum += w * sptrx[y_*src_.cols + x_];
							wsum += w;
						}
					}

					//store
					*dptrx = sum / wsum;
					*denormal_ptrx = count;
				}
			}
		}
		else if (src_.channels() == 3)
		{
			dest = Mat::zeros(src.size(), CV_32FC3);

#pragma omp parallel for
			for (int y = 0; y < src.rows; y++)
			{
				uchar* sptr = src_.ptr<uchar>(y + r) + r * 3;
				float* dptr = dest.ptr<float>(y);
				uchar* denormal_ptr = denormalMap.ptr<uchar>(y);

				for (int x = 0; x < src.cols; x++)
				{
					uchar* sptrx = sptr + x * 3;
					float* dptrx = dptr + x * 3;
					uchar* denormal_ptrx = denormal_ptr + x;

					int count = 0;
					float sum_b = 0.f;
					float sum_g = 0.f;
					float sum_r = 0.f;
					float wsum = 0.f;

					for (int j = 0; j < width; j++)
					{
						for (int i = 0; i < width; i++)
						{
							const int x_ = i - r;
							const int y_ = j - r;
							const float d = float(x_ * x_ + y_ * y_);
							const float w = exp(d * coef);

							if (checkDenormal(w))
							{
								count++;
							}

							sum_b += w * sptrx[(y_*src_.cols + x_) * 3 + 0];
							sum_g += w * sptrx[(y_*src_.cols + x_) * 3 + 1];
							sum_r += w * sptrx[(y_*src_.cols + x_) * 3 + 2];
							wsum += w;
						}
					}

					//store
					*(dptrx + 0) = sum_b / wsum;
					*(dptrx + 1) = sum_g / wsum;
					*(dptrx + 2) = sum_r / wsum;
					*denormal_ptrx = count;
				}
			}
		}
	}

	vector<float> getGaussianKernel_float(const Mat src, Mat& kernel, const Point pos, const int r, const float sigma_space)
	{
		Mat src_;
		copyMakeBorder(src, src_, r, r, r, r, BORDER_REPLICATE);

		const int width = 2 * r + 1;
		const float coef = -1.f / (2 * sigma_space*sigma_space);
		kernel = Mat(width, width, CV_32FC1);

		vector<float> ret;
		if (src_.channels() == 1)
		{
			float sum = 0.f;
			float wsum = 0.f;
			for (int j = 0; j < width; j++)
			{
				float* kp = kernel.ptr<float>(j);
				for (int i = 0; i < width; i++)
				{
					const int x = i - r;
					const int y = j - r;
					const float d = float(x * x + y * y);
					const float w = exp(float(d * coef));

					sum += w * src_.at<uchar>(pos.y + y + r, pos.x + x + r);

					wsum += w;
					*kp = w;
					kp++;
				}
			}
			ret.push_back(sum / wsum);
		}
		else if (src_.channels() == 3)
		{
			float sum_b = 0.f;
			float sum_g = 0.f;
			float sum_r = 0.f;
			float wsum = 0.f;

			for (int j = 0; j < width; j++)
			{
				float* kp = kernel.ptr<float>(j);
				for (int i = 0; i < width; i++)
				{
					const int x = i - r;
					const int y = j - r;
					const float d = float(x * x + y * y);
					const float w = exp(float(d * coef));

					Vec3b val = src_.at<Vec3b>(pos.y + y + r, pos.x + x + r);
					sum_b += w * val[0];
					sum_g += w * val[1];
					sum_r += w * val[2];

					wsum += w;
					*kp = w;
					kp++;
				}
			}
			ret.push_back(sum_b / wsum);
			ret.push_back(sum_g / wsum);
			ret.push_back(sum_r / wsum);
		}
		return ret;
	}

	void gaussian_kernel_test(const Mat src)
	{
		imshow("src", src);
		const string wname = "gaussian kernel";
		namedWindow(wname);
		int sigma_space = 1;
		int x = 0;
		int y = 0;
		createTrackbar("sigma_space", wname, &sigma_space, 500);
		createTrackbar("y", wname, &y, src.rows - 1);
		createTrackbar("x", wname, &x, src.cols - 1);

		ConsoleImage ci(Size(720, 480), "gaussian kernel test");
		while (true)
		{
			const float sigma_space_ = sigma_space / 10.f;
			const int r_ = sigma_space_ * 3.f;
			Mat gaussianKernel_32f;

			vector<float> ret = getGaussianKernel_float(src, gaussianKernel_32f, Point(x, y), r_, sigma_space_);
			Mat gaussianKernel_8u;
			gaussianKernel_32f.convertTo(gaussianKernel_8u, CV_8UC1, 255);
			imshow(wname, gaussianKernel_8u);

			Mat denormal = Mat::zeros(gaussianKernel_32f.size(), CV_8UC1);
			gaussianKernel_32f.forEach<float>([&](float& p, const int* position) -> void {
				if (checkDenormal(p))
				{
					uchar* val = (uchar*)denormal.ptr<uchar>(position[0], position[1]);
					*val = 255;
				}
			});
			const int count = int(sum(denormal)[0]) / 255;
			imshow("denormal kernel", denormal);

			Mat dest;
			Mat dest_32f;
			Mat denormalMap;
			gaussianFilter(src, dest_32f, denormalMap, r_, sigma_space_);
			dest_32f.convertTo(dest, CV_8U);

			//Mat dest_;
			//GaussianBlur(src, dest_, Size(2 * r_ + 1, 2 * r_ + 1), sigma_space_, sigma_space_, BORDER_REPLICATE);
			//cout << PSNR(dest, dest_) << endl;

			line(dest, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			line(dest, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("dest", dest);

			Mat denormalMap_line = denormalMap.clone();
			line(denormalMap_line, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			line(denormalMap_line, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("denormal image", denormalMap_line * 100);


			ci("r        : %d", r_);
			ci("sigma    : %f", sigma_space_);
			if (src.channels() == 1)
			{
				const int kernelNum = (2 * r_ + 1)*(2 * r_ + 1);
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("num denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("num denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			else
			{
				const int kernelNum = (2 * r_ + 1)*(2 * r_ + 1);
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("num denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("num denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			ci.flush();

			const int key = waitKey(1);
			if (key == 'q')
			{
				break;
			}
		}
	}


	/////////////////////////////
	// Gaussian range filter
	/////////////////////////////
	void gaussianRangeFilter(const Mat src, Mat& dest, Mat& denormalMap, const int r, const float sigma_range)
	{
		Mat src_;
		copyMakeBorder(src, src_, r, r, r, r, BORDER_REPLICATE);
		denormalMap = Mat::zeros(src.size(), CV_8UC1);

		const int width = 2 * r + 1;
		const float range_coef = -1.f / (2 * sigma_range*sigma_range);

		if (src_.channels() == 1)
		{
			dest = Mat::zeros(src.size(), CV_32FC1);

			for (int y = 0; y < src.rows; y++)
			{
				uchar* sptr = src_.ptr<uchar>(y + r) + r;
				float* dptr = dest.ptr<float>(y);
				uchar* denormal_ptr = denormalMap.ptr<uchar>(y);

				for (int x = 0; x < src.cols; x++)
				{
					uchar* sptrx = sptr + x;
					float* dptrx = dptr + x;
					uchar* denormal_ptrx = denormal_ptr + x;

					int count = 0;
					float sum = 0.f;
					float wsum = 0.f;
					for (int j = 0; j < width; j++)
					{
						for (int i = 0; i < width; i++)
						{
							const int x_ = i - r;
							const int y_ = j - r;

							const float c = abs(float(*sptrx) - float(sptrx[y_*src_.cols + x_]));
							const float w = exp(c * range_coef);

							if (checkDenormal(w))
							{
								count++;
							}

							sum += w * sptrx[y_*src_.cols + x_];
							wsum += w;
						}
					}

					//store
					*dptrx = sum / wsum;
					*denormal_ptrx = count;
				}

			}
		}
		else if (src_.channels() == 3)
		{
			dest = Mat::zeros(src.size(), CV_32FC3);

#pragma omp parallel for
			for (int y = 0; y < src.rows; y++)
			{
				uchar* sptr = src_.ptr<uchar>(y + r) + r * 3;
				float* dptr = dest.ptr<float>(y);
				uchar* denormal_ptr = denormalMap.ptr<uchar>(y);

				for (int x = 0; x < src.cols; x++)
				{
					uchar* sptrx = sptr + x * 3;
					float* dptrx = dptr + x * 3;
					uchar* denormal_ptrx = denormal_ptr + x;

					int count = 0;
					float sum_b = 0.f;
					float sum_g = 0.f;
					float sum_r = 0.f;
					float wsum = 0.f;

					for (int j = 0; j < width; j++)
					{
						for (int i = 0; i < width; i++)
						{
							const int x_ = i - r;
							const int y_ = j - r;

							const float c =
								(float(sptrx[0]) - float(sptrx[(y_*src_.cols + x_) * 3 + 0]))*(float(sptrx[0]) - float(sptrx[(y_*src_.cols + x_) * 3 + 0])) +
								(float(sptrx[1]) - float(sptrx[(y_*src_.cols + x_) * 3 + 1]))*(float(sptrx[1]) - float(sptrx[(y_*src_.cols + x_) * 3 + 1])) +
								(float(sptrx[2]) - float(sptrx[(y_*src_.cols + x_) * 3 + 2]))*(float(sptrx[2]) - float(sptrx[(y_*src_.cols + x_) * 3 + 2]));

							float a = c * range_coef;
							const float w = exp(a);

							if (checkDenormal(w))
							{
								count++;
							}

							sum_b += w * sptrx[(y_*src_.cols + x_) * 3 + 0];
							sum_g += w * sptrx[(y_*src_.cols + x_) * 3 + 1];
							sum_r += w * sptrx[(y_*src_.cols + x_) * 3 + 2];
							wsum += w;
						}
					}

					//store
					*(dptrx + 0) = sum_b / wsum;
					*(dptrx + 1) = sum_g / wsum;
					*(dptrx + 2) = sum_r / wsum;
					*denormal_ptrx = count;
				}
			}
		}
	}

	vector<float> getGaussianRangeKernel_float(const Mat src, Mat& kernel, const Point pos, const int r, const float sigma_range)
	{
		Mat src_;
		copyMakeBorder(src, src_, r, r, r, r, BORDER_REPLICATE);

		const int width = 2 * r + 1;
		const float range_coef = -1.f / (2 * sigma_range*sigma_range);
		kernel = Mat(width, width, CV_32FC1);

		vector<float> ret;
		if (src_.channels() == 1)
		{
			const uchar tval = src_.at<uchar>(pos.y + r, pos.x + r);
			float sum = 0.f;
			float wsum = 0.f;

			for (int j = 0; j < width; j++)
			{
				float* kp = kernel.ptr<float>(j);
				for (int i = 0; i < width; i++)
				{
					const int x = i - r;
					const int y = j - r;

					const uchar rval = src_.at<uchar>(pos.y + y + r, pos.x + x + r);

					const float c = abs(float(tval) - float(rval));

					const float w = exp(c * range_coef);
					sum += w * rval;

					wsum += w;
					*kp = w;
					kp++;
				}
			}
			ret.push_back(sum / wsum);
		}
		else if (src_.channels() == 3)
		{
			const Vec3b tval = src_.at<Vec3b>(pos.y + r, pos.x + r);
			float sum_b = 0.f;
			float sum_g = 0.f;
			float sum_r = 0.f;
			float wsum = 0.f;

			for (int j = 0; j < width; j++)
			{
				float* kp = kernel.ptr<float>(j);
				for (int i = 0; i < width; i++)
				{
					const int x = i - r;
					const int y = j - r;

					const Vec3b rval = src_.at<Vec3b>(pos.y + y + r, pos.x + x + r);

					const float c =
						(float(tval[0]) - float(rval[0]))*(float(tval[0]) - float(rval[0])) +
						(float(tval[1]) - float(rval[1]))*(float(tval[1]) - float(rval[1])) +
						(float(tval[2]) - float(rval[2]))*(float(tval[2]) - float(rval[2]));

					const float w = exp(c * range_coef);

					sum_b += w * rval[0];
					sum_g += w * rval[1];
					sum_r += w * rval[2];

					wsum += w;
					*kp = w;
					kp++;
				}
			}
			ret.push_back(sum_b / wsum);
			ret.push_back(sum_g / wsum);
			ret.push_back(sum_r / wsum);
		}
		return ret;
	}

	void gaussianRange_kernel_test(const Mat src)
	{
		imshow("src", src);
		const string wname = "gaussian range kernel";
		namedWindow(wname);
		int sigma_range = 1;
		int r = 1;
		int x = 0;
		int y = 0;
		createTrackbar("sigma_range", wname, &sigma_range, 500);
		createTrackbar("r", wname, &r, 500);
		createTrackbar("x", wname, &x, src.cols - 1);
		createTrackbar("y", wname, &y, src.rows - 1);

		ConsoleImage ci(Size(720, 480), "gaussian range kernel test");
		while (true)
		{
			Mat gaussianKernel_32f;
			const float sigma_range_ = sigma_range / 10.f;

			vector<float> ret = getGaussianRangeKernel_float(src, gaussianKernel_32f, Point(x, y), r, sigma_range_);
			Mat gaussianKernel_8u;
			gaussianKernel_32f.convertTo(gaussianKernel_8u, CV_8UC1, 255);
			imshow(wname, gaussianKernel_8u);

			Mat denormal = Mat::zeros(gaussianKernel_32f.size(), CV_8UC1);
			gaussianKernel_32f.forEach<float>([&](float& p, const int* position) -> void {
				if (checkDenormal(p))
				{
					uchar* val = static_cast<uchar*>(denormal.ptr<uchar>(position[0], position[1]));
					*val = 255;
				}
			});
			const int count = int(sum(denormal)[0]) / 255;
			imshow("denormal kernel", denormal);

			Mat dest;
			Mat dest_32f;
			Mat denormalMap;

			gaussianRangeFilter(src, dest_32f, denormalMap, r, sigma_range_);
			dest_32f.convertTo(dest, CV_8U);

			line(dest, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			line(dest, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("dest", dest);

			Mat denormalMap_line = denormalMap.clone();
			//line(denormalMap_line, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			//line(denormalMap_line, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("denormal image", denormalMap_line);

			ci(format("r           : %d", r));
			ci(format("sigma_range : %f", sigma_range_));
			if (src.channels() == 1)
			{
				const int kernelNum = (2 * r + 1)*(2 * r + 1);
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("num denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("num denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			else
			{
				const int kernelNum = (2 * r + 1)*(2 * r + 1);
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			ci.flush();

			const int key = waitKey(1);
			if (key == 'q')
			{
				ofstream ofs("gaussian_range.csv");
				for (int j = 0; j < denormalMap_line.rows; j++)
				{
					uchar* dp = denormalMap_line.ptr<uchar>(j);
					for (int i = 0; i < denormalMap_line.cols; i++)
					{
						ofs << int(*dp) << " ";
						dp++;
					}
					ofs << endl;
				}
				ofs.close();
				break;
			}
		}
	}


	/////////////////////////////
	// bilateral filter
	/////////////////////////////
	void bilateralFilter(const Mat src, Mat& dest, Mat& denormalMap, const int r, const float sigma_range, const float sigma_space)
	{
		Mat src_;
		copyMakeBorder(src, src_, r, r, r, r, BORDER_REPLICATE);
		denormalMap = Mat::zeros(src.size(), CV_8UC1);

		const int width = 2 * r + 1;
		const float range_coef = -1.f / (2 * sigma_range*sigma_range);
		const float space_coef = -1.f / (2 * sigma_space*sigma_space);

		if (src_.channels() == 1)
		{
			dest = Mat::zeros(src.size(), CV_32FC1);

			for (int y = 0; y < src.rows; y++)
			{
				uchar* sptr = src_.ptr<uchar>(y + r) + r;
				float* dptr = dest.ptr<float>(y);
				uchar* denormal_ptr = denormalMap.ptr<uchar>(y);

				for (int x = 0; x < src.cols; x++)
				{
					uchar* sptrx = sptr + x;
					float* dptrx = dptr + x;
					uchar* denormal_ptrx = denormal_ptr + x;

					int count = 0;
					float sum = 0.f;
					float wsum = 0.f;
					for (int j = 0; j < width; j++)
					{
						for (int i = 0; i < width; i++)
						{
							const int x_ = i - r;
							const int y_ = j - r;

							const float c = abs(float(*sptrx) - float(sptrx[y_*src_.cols + x_]));
							const float d = float(x_ * x_ + y_ * y_);

							const float w = exp(d * space_coef + c * range_coef);

							if (checkDenormal(w))
							{
								count++;
							}

							sum += w * sptrx[y_*src_.cols + x_];
							wsum += w;
						}
					}

					//store
					*dptrx = sum / wsum;
					*denormal_ptrx = count;
				}

			}
		}
		else if (src_.channels() == 3)
		{
			dest = Mat::zeros(src.size(), CV_32FC3);

#pragma omp parallel for
			for (int y = 0; y < src.rows; y++)
			{
				uchar* sptr = src_.ptr<uchar>(y + r) + r * 3;
				float* dptr = dest.ptr<float>(y);
				uchar* denormal_ptr = denormalMap.ptr<uchar>(y);

				for (int x = 0; x < src.cols; x++)
				{
					uchar* sptrx = sptr + x * 3;
					float* dptrx = dptr + x * 3;
					uchar* denormal_ptrx = denormal_ptr + x;

					int count = 0;
					float sum_b = 0.f;
					float sum_g = 0.f;
					float sum_r = 0.f;
					float wsum = 0.f;

					for (int j = 0; j < width; j++)
					{
						for (int i = 0; i < width; i++)
						{
							const int x_ = i - r;
							const int y_ = j - r;

							const float c =
								(float(sptrx[0]) - float(sptrx[(y_*src_.cols + x_) * 3 + 0]))*(float(sptrx[0]) - float(sptrx[(y_*src_.cols + x_) * 3 + 0])) +
								(float(sptrx[1]) - float(sptrx[(y_*src_.cols + x_) * 3 + 1]))*(float(sptrx[1]) - float(sptrx[(y_*src_.cols + x_) * 3 + 1])) +
								(float(sptrx[2]) - float(sptrx[(y_*src_.cols + x_) * 3 + 2]))*(float(sptrx[2]) - float(sptrx[(y_*src_.cols + x_) * 3 + 2]));
							const float d = float(x_ * x_ + y_ * y_);

							float a = d * space_coef + c * range_coef;
							//a = max(a, EXP_DENORMAL_THERESHOLD);
							const float w = exp(a);

							if (checkDenormal(w))
							{
								count++;
							}

							sum_b += w * sptrx[(y_*src_.cols + x_) * 3 + 0];
							sum_g += w * sptrx[(y_*src_.cols + x_) * 3 + 1];
							sum_r += w * sptrx[(y_*src_.cols + x_) * 3 + 2];
							wsum += w;
						}
					}

					//store
					*(dptrx + 0) = sum_b / wsum;
					*(dptrx + 1) = sum_g / wsum;
					*(dptrx + 2) = sum_r / wsum;
					*denormal_ptrx = count;
				}
			}
		}
	}

	vector<float> getBilateralKernel_float(const Mat src, Mat& kernel, const Point pos, const int r, const float sigma_range, const float sigma_space)
	{
		Mat src_;
		copyMakeBorder(src, src_, r, r, r, r, BORDER_REPLICATE);

		const int width = 2 * r + 1;
		const float range_coef = -1.f / (2 * sigma_range*sigma_range);
		const float space_coef = -1.f / (2 * sigma_space*sigma_space);
		kernel = Mat(width, width, CV_32FC1);

		vector<float> ret;
		if (src_.channels() == 1)
		{
			const uchar tval = src_.at<uchar>(pos.y + r, pos.x + r);
			float sum = 0.f;
			float wsum = 0.f;

			for (int j = 0; j < width; j++)
			{
				float* kp = kernel.ptr<float>(j);
				for (int i = 0; i < width; i++)
				{
					const int x = i - r;
					const int y = j - r;

					const uchar rval = src_.at<uchar>(pos.y + y + r, pos.x + x + r);

					const float c = abs(float(tval) - float(rval));
					const float d = float(x * x + y * y);

					const float w = exp(d * space_coef + c * range_coef);
					sum += w * rval;

					wsum += w;
					*kp = w;
					kp++;
				}
			}
			ret.push_back(sum / wsum);
		}
		else if (src_.channels() == 3)
		{
			const Vec3b tval = src_.at<Vec3b>(pos.y + r, pos.x + r);
			float sum_b = 0.f;
			float sum_g = 0.f;
			float sum_r = 0.f;
			float wsum = 0.f;

			for (int j = 0; j < width; j++)
			{
				float* kp = kernel.ptr<float>(j);
				for (int i = 0; i < width; i++)
				{
					const int x = i - r;
					const int y = j - r;

					const Vec3b rval = src_.at<Vec3b>(pos.y + y + r, pos.x + x + r);

					const float c =
						(float(tval[0]) - float(rval[0]))*(float(tval[0]) - float(rval[0])) +
						(float(tval[1]) - float(rval[1]))*(float(tval[1]) - float(rval[1])) +
						(float(tval[2]) - float(rval[2]))*(float(tval[2]) - float(rval[2]));
					const float d = float(x * x + y * y);

					const float w = exp(d * space_coef + c * range_coef);

					sum_b += w * rval[0];
					sum_g += w * rval[1];
					sum_r += w * rval[2];

					wsum += w;
					*kp = w;
					kp++;
				}
			}
			ret.push_back(sum_b / wsum);
			ret.push_back(sum_g / wsum);
			ret.push_back(sum_r / wsum);
		}
		return ret;
	}

	void bilateral_kernel_test(const Mat src)
	{
		imshow("src", src);
		const string wname = "bilateral kernel";
		namedWindow(wname);
		int sigma_range = 1;
		int sigma_space = 1;
		int x = 0;
		int y = 0;
		createTrackbar("sigma_range", wname, &sigma_range, 500);
		createTrackbar("sigma_space", wname, &sigma_space, 500);
		createTrackbar("x", wname, &x, src.cols - 1);
		createTrackbar("y", wname, &y, src.rows - 1);

		ConsoleImage ci(Size(720, 480), "bilateral kernel test");
		while (true)
		{
			Mat bilateralKernel_32f;
			const float sigma_range_ = sigma_range / 10.f;
			const float sigma_space_ = sigma_space / 10.f;
			const int r_ = 3 * sigma_space_;

			vector<float> ret = getBilateralKernel_float(src, bilateralKernel_32f, Point(x, y), r_, sigma_range_, sigma_space_);
			Mat gaussianKernel_8u;
			bilateralKernel_32f.convertTo(gaussianKernel_8u, CV_8UC1, 255);
			imshow(wname, gaussianKernel_8u);

			Mat denormal = Mat::zeros(bilateralKernel_32f.size(), CV_8UC1);
			bilateralKernel_32f.forEach<float>([&](float& p, const int* position) -> void {
				if (checkDenormal(p))
				{
					uchar* val = static_cast<uchar*>(denormal.ptr<uchar>(position[0], position[1]));
					*val = 255;
				}
			});
			const int count = int(sum(denormal)[0]) / 255;
			imshow("denormal kernel", denormal);

			Mat dest;
			Mat dest_32f;
			Mat denormalMap;

			bilateralFilter(src, dest_32f, denormalMap, r_, sigma_range_, sigma_space_);
			dest_32f.convertTo(dest, CV_8U);

			line(dest, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			line(dest, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("dest", dest);

			Mat denormalMap_line = denormalMap.clone();
			//line(denormalMap_line, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			//line(denormalMap_line, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("denormal image", denormalMap_line);

			ci(format("r           : %d", r_));
			ci(format("sigma_range : %f", sigma_range_));
			ci(format("sigma_space : %f", sigma_space_));
			if (src.channels() == 1)
			{
				const int kernelNum = (2 * r_ + 1)*(2 * r_ + 1);
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("num denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("num denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			else
			{
				const int kernelNum = (2 * r_ + 1)*(2 * r_ + 1);
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			ci.flush();

			const int key = waitKey(1);
			if (key == 'q')
			{
				ofstream ofs("bilateral.csv");
				for (int j = 0; j < denormalMap_line.rows; j++)
				{
					uchar* dp = denormalMap_line.ptr<uchar>(j);
					for (int i = 0; i < denormalMap_line.cols; i++)
					{
						ofs << int(*dp) << " ";
						dp++;
					}
					ofs << endl;
				}
				ofs.close();
				break;
			}
		}
	}


	/////////////////////////////
	// non local means filter
	/////////////////////////////
	void nonLocalMeansFilter(const Mat src, Mat& dest, Mat& denormalMap, const Size templateWindowSize, const Size searchWindowSize, const float h)
	{
		denormalMap = Mat::zeros(src.size(), CV_8UC1);

		const int cn = src.channels();
		const int templateH = templateWindowSize.width >> 1;
		const int templateV = templateWindowSize.height >> 1;
		const int searchH = searchWindowSize.width >> 1;
		const int searchV = searchWindowSize.height >> 1;
		const int bbx = templateH + searchH;
		const int bby = templateV + searchV;

		//create large size image for bounding box;
		Mat im;
		if (cn == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx, BORDER_REPLICATE);
			dest = Mat::zeros(src.size(), CV_32FC1);
		}
		else if (cn == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx, BORDER_REPLICATE);
			cvtColorBGR2PLANE(temp, im);
			dest = Mat::zeros(src.size(), CV_32FC3);
		}

		vector<int> template_offset(templateWindowSize.area());
		int* template_ofs = &template_offset[0];
		vector<int> search_offset(searchWindowSize.area());
		int* search_ofs = &search_offset[0];

		int template_maxk = 0;
		setSpaceKernel(template_ofs, template_maxk, templateH, templateV, im.cols, true);
		int search_maxk = 0;
		setSpaceKernel(search_ofs, search_maxk, searchH, searchV, im.cols, true);


		const int tr_x = templateWindowSize.width >> 1;
		const int sr_x = searchWindowSize.width >> 1;
		const int tr_y = templateWindowSize.height >> 1;
		const int sr_y = searchWindowSize.height >> 1;
		const float gauss_color_coeff = -(1.0 / (h*h));

		if (dest.channels() == 3)
		{
			const int colorstep = im.size().area() / 3;
			const int colorstep2 = colorstep * 2;
#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				float* d = dest.ptr<float>(j);
				uchar* denormal_ptr = denormalMap.ptr<uchar>(j);
				const uchar* sptr_ = im.ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				for (int i = 0; i < src.cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const uchar* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<uchar> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					int dcount = 0;
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - (float)tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep) - (float)tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep2) - (float)tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff);
						if (checkDenormal(w))
						{
							dcount++;
						}
						sumb += *(s)* w;
						sumg += *(s + colorstep) * w;
						sumr += *(s + colorstep2) * w;
						sumw += w;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
					denormal_ptr[0] = dcount;
					denormal_ptr++;
				}
			}
		}
		else if (dest.channels() == 1)
		{
#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				float* d = dest.ptr<float>(j);
				const uchar* sptr_ = im.ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				uchar* denormal_ptr = denormalMap.ptr<uchar>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float sumb = 0;
					float sumw = 0;

					const uchar* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<uchar> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					int dcount = 0;
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - (float)tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff);
						if (checkDenormal(w))
						{
							dcount++;
						}
						sumb += *(s)* w;
						sumw += w;
					}
					d[0] = sumb / sumw;
					d++;
					denormal_ptr[0] = dcount;
					denormal_ptr++;
				}
			}
		}
	}

	vector<float> getNonLocalMeansKernel_float(const Mat src, Mat& kernel, const Point pos, const Size templateWindowSize, const Size searchWindowSize, const float h)
	{
		kernel = Mat(searchWindowSize, CV_32FC1);

		const int cn = src.channels();
		const int templateH = templateWindowSize.width >> 1;
		const int templateV = templateWindowSize.height >> 1;
		const int searchH = searchWindowSize.width >> 1;
		const int searchV = searchWindowSize.height >> 1;
		const int bbx = templateH + searchH;
		const int bby = templateV + searchV;


		//create large size image for bounding box;
		Mat im;
		if (cn == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx, BORDER_REPLICATE);
		}
		else if (cn == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx, BORDER_REPLICATE);
			cvtColorBGR2PLANE(temp, im);
		}

		vector<int> template_offset(templateWindowSize.area());
		int* template_ofs = &template_offset[0];
		vector<int> search_offset(searchWindowSize.area());
		int* search_ofs = &search_offset[0];

		int template_maxk = 0;
		setSpaceKernel(template_ofs, template_maxk, templateH, templateV, im.cols, true);
		int search_maxk = 0;
		vector<float> search_weight(searchWindowSize.area());
		setSpaceKernel(search_ofs, search_maxk, searchH, searchV, im.cols, true);


		const int tr_x = templateWindowSize.width >> 1;
		const int sr_x = searchWindowSize.width >> 1;
		const int tr_y = templateWindowSize.height >> 1;
		const int sr_y = searchWindowSize.height >> 1;
		const float gauss_color_coeff = -(1.0 / (h*h));

		vector<float> ret;
		if (src.channels() == 3)
		{
			const int colorstep = im.size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const int j = pos.y;
			{
				const uchar* sptr_ = im.ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				const int i = pos.x;
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const uchar* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<uchar> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					float* kp = kernel.ptr<float>(j);
					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - (float)tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep) - (float)tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep2) - (float)tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff);
						sumb += *(s)* w;
						sumg += *(s + colorstep) * w;
						sumr += *(s + colorstep2) * w;
						sumw += w;
						*kp = w;
						kp++;
					}
					ret.push_back(sumb / sumw);
					ret.push_back(sumg / sumw);
					ret.push_back(sumr / sumw);
				}
			}
		}
		else if (src.channels() == 1)
		{
			const int j = pos.y;
			{
				const uchar* sptr_ = im.ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				const int i = pos.x;
				{
					float sumb = 0;
					float sumw = 0;

					const uchar* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<uchar> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					float* kp = kernel.ptr<float>(j);
					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - (float)tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff);
						sumb += *(s)* w;
						sumw += w;
						*kp = w;
						kp++;
					}
					ret.push_back(sumb / sumw);
				}
			}
		}
		return ret;
	}

	void nonLocalMeans_kernel_test(const Mat src)
	{
		imshow("src", src);
		const string wname = "non local means kernel";
		namedWindow(wname);
		int templateWindowR = 1;
		int seachWindowR = 2;
		int h = 10;
		int x = 0;
		int y = 0;
		createTrackbar("templateWindowSize", wname, &templateWindowR, 500);
		createTrackbar("seachWindowSize", wname, &seachWindowR, 500);
		createTrackbar("h", wname, &h, 100);
		createTrackbar("x", wname, &x, src.cols - 1);
		createTrackbar("y", wname, &y, src.rows - 1);

		ConsoleImage ci(Size(720, 480), "non local means kernel test");
		while (true)
		{
			Mat nonLocalMeansKernel_32f;
			const Size tW(2 * templateWindowR + 1, 2 * templateWindowR + 1);
			const Size sW(2 * seachWindowR + 1, 2 * seachWindowR + 1);

			const float _h = sqrt(2)*h;
			vector<float> ret = getNonLocalMeansKernel_float(src, nonLocalMeansKernel_32f, Point(x, y), tW, sW, _h);
			Mat gaussianKernel_8u;
			nonLocalMeansKernel_32f.convertTo(gaussianKernel_8u, CV_8UC1, 255);
			imshow(wname, gaussianKernel_8u);

			Mat denormal = Mat::zeros(nonLocalMeansKernel_32f.size(), CV_8UC1);
			nonLocalMeansKernel_32f.forEach<float>([&](float& p, const int* position) -> void {
				if (checkDenormal(p))
				{
					uchar* val = static_cast<uchar*>(denormal.ptr<uchar>(position[0], position[1]));
					*val = 255;
				}
			});
			const int count = int(sum(denormal)[0]) / 255;
			imshow("denormal kernel", denormal);

			Mat dest;
			Mat dest_32f;
			Mat denormalMap;

			nonLocalMeansFilter(src, dest_32f, denormalMap, tW, sW, _h);
			dest_32f.convertTo(dest, CV_8U);

			line(dest, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			line(dest, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("dest", dest);

			Mat denormalMap_line = denormalMap.clone();
			//line(denormalMap_line, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			//line(denormalMap_line, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("denormal image", denormalMap_line * 100);

			ci(format("templateWindow : (%d, %d)", tW.width, tW.height));
			ci(format("searchWindow : (%d, %d)", sW.width, sW.height));
			ci(format("h : %f", _h));
			if (src.channels() == 1)
			{
				const int kernelNum = sW.area();
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("num denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("num denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			else
			{
				const int kernelNum = sW.area();
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			ci.flush();

			const int key = waitKey(1);
			if (key == 'q')
			{
				ofstream ofs("nonlocalmeans.csv");
				for (int j = 0; j < denormalMap_line.rows; j++)
				{
					uchar* dp = denormalMap_line.ptr<uchar>(j);
					for (int i = 0; i < denormalMap_line.cols; i++)
					{
						ofs << int(*dp) << " ";
						dp++;
					}
					ofs << endl;
				}
				ofs.close();
				break;
			}
		}
	}

	/////////////////////////////
	// bilateral non local means filter
	/////////////////////////////
	void bilateralNonLocalMeansFilter(const Mat src, Mat& dest, Mat& denormalMap, const Size templateWindowSize, const Size searchWindowSize, const float h, const float sigma_space)
	{
		denormalMap = Mat::zeros(src.size(), CV_8UC1);

		const int cn = src.channels();
		const int templateH = templateWindowSize.width >> 1;
		const int templateV = templateWindowSize.height >> 1;
		const int searchH = searchWindowSize.width >> 1;
		const int searchV = searchWindowSize.height >> 1;
		const int bbx = templateH + searchH;
		const int bby = templateV + searchV;

		//create large size image for bounding box;
		Mat im;
		if (cn == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx, BORDER_REPLICATE);
			dest = Mat::zeros(src.size(), CV_32FC1);
		}
		else if (cn == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx, BORDER_REPLICATE);
			cvtColorBGR2PLANE(temp, im);
			dest = Mat::zeros(src.size(), CV_32FC3);
		}

		vector<int> template_offset(templateWindowSize.area());
		int* template_ofs = &template_offset[0];
		vector<int> search_offset(searchWindowSize.area());
		int* search_ofs = &search_offset[0];

		int template_maxk = 0;
		setSpaceKernel(template_ofs, template_maxk, templateH, templateV, im.cols, true);
		int search_maxk = 0;
		const float gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
		vector<float> search_weight(searchWindowSize.area());
		float* space_weight = &search_weight[0];
		setSpaceKernel(space_weight, search_ofs, search_maxk, searchH, searchV, gauss_space_coeff, im.cols, true, false);


		const int tr_x = templateWindowSize.width >> 1;
		const int sr_x = searchWindowSize.width >> 1;
		const int tr_y = templateWindowSize.height >> 1;
		const int sr_y = searchWindowSize.height >> 1;
		const float gauss_color_coeff = -(1.0 / (h*h));

		if (dest.channels() == 3)
		{
			const int colorstep = im.size().area() / 3;
			const int colorstep2 = colorstep * 2;
#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				float* d = dest.ptr<float>(j);
				uchar* denormal_ptr = denormalMap.ptr<uchar>(j);
				const uchar* sptr_ = im.ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				for (int i = 0; i < src.cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const uchar* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<uchar> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					const float* spw = space_weight;
					int dcount = 0;
					for (int l = 0; l < search_maxk; l++, spw++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - (float)tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep) - (float)tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep2) - (float)tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff + *spw * gauss_space_coeff);
						if (checkDenormal(w))
						{
							dcount++;
						}
						sumb += *(s)* w;
						sumg += *(s + colorstep) * w;
						sumr += *(s + colorstep2) * w;
						sumw += w;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
					denormal_ptr[0] = dcount;
					denormal_ptr++;
				}
			}
		}
		else if (dest.channels() == 1)
		{
#pragma omp parallel for
			for (int j = 0; j < src.rows; j++)
			{
				float* d = dest.ptr<float>(j);
				const uchar* sptr_ = im.ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				uchar* denormal_ptr = denormalMap.ptr<uchar>(j);
				for (int i = 0; i < src.cols; i++)
				{
					float sumb = 0;
					float sumw = 0;

					const uchar* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<uchar> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					const float* spw = space_weight;
					int dcount = 0;
					for (int l = 0; l < search_maxk; l++, spw++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - (float)tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff + *spw * gauss_space_coeff);
						if (checkDenormal(w))
						{
							dcount++;
						}
						sumb += *(s)* w;
						sumw += w;
					}
					d[0] = sumb / sumw;
					d++;
					denormal_ptr[0] = dcount;
					denormal_ptr++;
				}
			}
		}
	}

	vector<float> getBilateralNonLocalMeansKernel_float(const Mat src, Mat& kernel, const Point pos, const Size templateWindowSize, const Size searchWindowSize, const float h, const float sigma_space)
	{
		kernel = Mat(searchWindowSize, CV_32FC1);

		const int cn = src.channels();
		const int templateH = templateWindowSize.width >> 1;
		const int templateV = templateWindowSize.height >> 1;
		const int searchH = searchWindowSize.width >> 1;
		const int searchV = searchWindowSize.height >> 1;
		const int bbx = templateH + searchH;
		const int bby = templateV + searchV;


		//create large size image for bounding box;
		Mat im;
		if (cn == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx, BORDER_REPLICATE);
		}
		else if (cn == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx, BORDER_REPLICATE);
			cvtColorBGR2PLANE(temp, im);
		}

		vector<int> template_offset(templateWindowSize.area());
		int* template_ofs = &template_offset[0];
		vector<int> search_offset(searchWindowSize.area());
		int* search_ofs = &search_offset[0];

		int template_maxk = 0;
		setSpaceKernel(template_ofs, template_maxk, templateH, templateV, im.cols, true);
		int search_maxk = 0;
		const float gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
		vector<float> search_weight(searchWindowSize.area());
		float* space_weight = &search_weight[0];
		setSpaceKernel(space_weight, search_ofs, search_maxk, searchH, searchV, gauss_space_coeff, im.cols, true, false);


		const int tr_x = templateWindowSize.width >> 1;
		const int sr_x = searchWindowSize.width >> 1;
		const int tr_y = templateWindowSize.height >> 1;
		const int sr_y = searchWindowSize.height >> 1;
		const float gauss_color_coeff = -(1.0 / (h*h));

		vector<float> ret;
		if (src.channels() == 3)
		{
			const int colorstep = im.size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const int j = pos.y;
			{
				const uchar* sptr_ = im.ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				const int i = pos.x;
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const uchar* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<uchar> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					float* kp = kernel.ptr<float>(j);
					//search loop
					const int* s_ofs = &search_ofs[0];
					const float* spw = space_weight;
					for (int l = 0; l < search_maxk; l++, spw++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - (float)tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep) - (float)tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep2) - (float)tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff + *spw * gauss_space_coeff);
						sumb += *(s)* w;
						sumg += *(s + colorstep) * w;
						sumr += *(s + colorstep2) * w;
						sumw += w;
						*kp = w;
						kp++;
					}
					ret.push_back(sumb / sumw);
					ret.push_back(sumg / sumw);
					ret.push_back(sumr / sumw);
				}
			}
		}
		else if (src.channels() == 1)
		{
			const int j = pos.y;
			{
				const uchar* sptr_ = im.ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				const int i = pos.x;
				{
					float sumb = 0;
					float sumw = 0;

					const uchar* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<uchar> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					float* kp = kernel.ptr<float>(j);
					//search loop
					const int* s_ofs = &search_ofs[0];
					const float* spw = space_weight;
					int dcount = 0;
					for (int l = 0; l < search_maxk; l++, spw++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - (float)tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff + *spw * gauss_space_coeff);
						sumb += *(s)* w;
						sumw += w;
						*kp = w;
						kp++;
					}
					ret.push_back(sumb / sumw);
				}
			}
		}
		return ret;
	}

	void bilateralNonLocalMeans_kernel_test(const Mat src)
	{
		imshow("src", src);
		const string wname = "bilateral non local means kernel";
		namedWindow(wname);
		int templateWindowR = 1;
		int seachWindowR = 2;
		int h = 10;
		int sigma_space = 3;
		int x = 0;
		int y = 0;
		createTrackbar("templateWindowSize", wname, &templateWindowR, 500);
		createTrackbar("seachWindowSize", wname, &seachWindowR, 500);
		createTrackbar("h", wname, &h, 100);
		createTrackbar("sigma_space", wname, &sigma_space, 100);
		createTrackbar("x", wname, &x, src.cols - 1);
		createTrackbar("y", wname, &y, src.rows - 1);

		ConsoleImage ci(Size(720, 480), "bilatral non-local means kernel test");
		while (true)
		{
			Mat nonLocalMeansKernel_32f;
			const Size tW(2 * templateWindowR + 1, 2 * templateWindowR + 1);
			const Size sW(2 * seachWindowR + 1, 2 * seachWindowR + 1);

			const float _h = sqrt(2)*h;
			vector<float> ret = getBilateralNonLocalMeansKernel_float(src, nonLocalMeansKernel_32f, Point(x, y), tW, sW, _h, (float)sigma_space);
			Mat gaussianKernel_8u;
			nonLocalMeansKernel_32f.convertTo(gaussianKernel_8u, CV_8UC1, 255);
			imshow(wname, gaussianKernel_8u);

			Mat denormal = Mat::zeros(nonLocalMeansKernel_32f.size(), CV_8UC1);
			nonLocalMeansKernel_32f.forEach<float>([&](float& p, const int* position) -> void {
				if (checkDenormal(p))
				{
					uchar* val = static_cast<uchar*>(denormal.ptr<uchar>(position[0], position[1]));
					*val = 255;
				}
			});
			const int count = int(sum(denormal)[0]) / 255;
			imshow("denormal kernel", denormal);

			Mat dest;
			Mat dest_32f;
			Mat denormalMap;

			bilateralNonLocalMeansFilter(src, dest_32f, denormalMap, tW, sW, _h, (float)sigma_space);
			dest_32f.convertTo(dest, CV_8U);

			line(dest, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			line(dest, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("dest", dest);

			Mat denormalMap_line = denormalMap.clone();
			//line(denormalMap_line, Point(x, 0), Point(x, dest.rows), Scalar(255, 255, 255));
			//line(denormalMap_line, Point(0, y), Point(dest.cols, y), Scalar(255, 255, 255));
			imshow("denormal image", denormalMap_line * 100);

			ci(format("templateWindow : (%d, %d)", tW.width, tW.height));
			ci(format("searchWindow : (%d, %d)", sW.width, sW.height));
			ci(format("h : %f", (float)_h));
			ci(format("sigma_space : %f", (float)sigma_space));
			if (src.channels() == 1)
			{
				const int kernelNum = sW.area();
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("num denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("num denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			else
			{
				const int kernelNum = sW.area();
				const int imageSize = src.size().area();
				const int imageDenormal = int(sum(denormalMap)[0]);
				ci("denormal in kernel : %d/%d = %f", count, kernelNum, count / float(kernelNum));
				ci("denormal in image  : %d/(%d * %d) = %f", imageDenormal, imageSize, kernelNum, imageDenormal / float(imageSize*kernelNum));
			}
			ci.flush();

			const int key = waitKey(1);
			if (key == 'q')
			{
				ofstream ofs("bilatralnonlocalmeans.csv");
				for (int j = 0; j < denormalMap_line.rows; j++)
				{
					uchar* dp = denormalMap_line.ptr<uchar>(j);
					for (int i = 0; i < denormalMap_line.cols; i++)
					{
						ofs << int(*dp) << " ";
						dp++;
					}
					ofs << endl;
				}
				ofs.close();
				break;
			}
		}
	}
}
