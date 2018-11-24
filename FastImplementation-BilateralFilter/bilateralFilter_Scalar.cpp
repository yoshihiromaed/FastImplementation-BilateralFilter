#include "filter.h"

using namespace std;
using namespace cv;


class BilateralFilterInvorker_8u : public cv::ParallelLoopBody
{
public:
	BilateralFilterInvorker_8u(const Mat src, Mat& _dest, const int _radius, const float _sigma_range, const float _sigma_space, const int borderType, const int parallel = -1) :
		dest(&_dest), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
		if (_radius == 0)
		{
			src.copyTo(_dest);
			return;
		}
		radius = _radius;

		size = src.size();

		if (dest->empty() || src.type() != dest->type() || src.size() != dest->size())
		{
			*dest = Mat::zeros(src.size(), src.type());
		}

		const float gauss_space_coeff = -0.5f / (sigma_space*sigma_space);

		cn = src.channels();

		const int d = 2 * radius + 1;
		copyMakeBorder(src, temp, radius, radius, radius, radius, borderType);

		vector<float> _space_weight(d*d);
		vector<int> _space_ofs(d*d);
		space_weight = &_space_weight[0];
		space_ofs = &_space_ofs[0];

		maxk = 0;
		// initialize space-related bilateral filter coefficients
		for (int i = -radius; i <= radius; i++)
		{
			for (int j = -radius; j <= radius; j++)
			{
				const float r = (float)i*i + (float)j*j;
				space_weight[maxk] = r * gauss_space_coeff;
				space_ofs[maxk++] = (int)(i*temp.cols*cn + j * cn);
			}
		}

		parallel_for_(Range(0, size.height), *this, parallel);
	}

	void operator()(const Range& range) const override
	{
		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);

		for (int i = range.start; i < range.end; i++)
		{
			const uchar* sptr = temp.ptr<uchar>(i + radius) + radius * cn;
			uchar* dptr = dest->ptr<uchar>(i);

			if (cn == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					float sum = 0.f;
					float wsum = 0.f;
					const float val0 = sptr[j];
					for (int k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];

						float aw = space_weight[k] + (val - val0)*(val - val0)*gauss_range_coeff;
#if __BF_PREVENTION__
						aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
						float w = exp(aw);
#if __BF_POSTVENTION__
						w = max(w, FLT_MIN);
#endif
						sum += val * w;
						wsum += w;
					}
					dptr[j] = cvRound(sum / wsum);
				}
			}
			else if (cn == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					float sum_b = 0.f;
					float sum_g = 0.f;
					float sum_r = 0.f;
					float wsum = 0.f;
					const float b0 = sptr[j + 0];
					const float g0 = sptr[j + 1];
					const float r0 = sptr[j + 2];
					for (int k = 0; k < maxk; k++)
					{
						const uchar* sptr_k = sptr + j + space_ofs[k];
						const float b = sptr_k[0];
						const float g = sptr_k[1];
						const float r = sptr_k[2];

						float aw = space_weight[k] + ((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)) *gauss_range_coeff;
#if __BF_PREVENTION__
						aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
						float w = exp(aw);
#if __BF_POSTVENTION__
						w = max(w, FLT_MIN);
#endif

						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[j] = cvRound(sum_b / wsum);
					dptr[j + 1] = cvRound(sum_g / wsum);
					dptr[j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	Mat temp;

	Mat *dest;
	int radius, maxk, cn;
	Size size;
	int *space_ofs;
	float *space_weight;
	float sigma_range, sigma_space;
};

class BilateralFilterInvorker_32f : public cv::ParallelLoopBody
{
public:
	BilateralFilterInvorker_32f(const Mat src, Mat& _dest, const int _radius, const float _sigma_range, const float _sigma_space, const int borderType, const int parallel = -1) :
		dest(&_dest), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
		if (_radius == 0)
		{
			src.copyTo(_dest);
			return;
		}
		radius = _radius;

		size = src.size();

		if (dest->empty() || src.type() != dest->type() || src.size() != dest->size())
		{
			*dest = Mat::zeros(src.size(), src.type());
		}

		const float gauss_space_coeff = -0.5f / (sigma_space*sigma_space);

		cn = src.channels();

		const int d = 2 * radius + 1;
		copyMakeBorder(src, temp, radius, radius, radius, radius, borderType);

		vector<float> _space_weight(d*d);
		vector<int> _space_ofs(d*d);
		space_weight = &_space_weight[0];
		space_ofs = &_space_ofs[0];

		maxk = 0;
		// initialize space-related bilateral filter coefficients
		for (int i = -radius; i <= radius; i++)
		{
			for (int j = -radius; j <= radius; j++)
			{
				const float r = (float)i*i + (float)j*j;
				space_weight[maxk] = r * gauss_space_coeff;
				space_ofs[maxk++] = (int)(i*temp.cols*cn + j * cn);
			}
		}

		parallel_for_(Range(0, size.height), *this, parallel);
	}

	void operator()(const Range& range) const override
	{
		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);

		for (int i = range.start; i < range.end; i++)
		{
			const float* sptr = temp.ptr<float>(i + radius) + radius * cn;
			float* dptr = dest->ptr<float>(i);

			if (cn == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					float sum = 0.f;
					float wsum = 0.f;
					const float val0 = sptr[j];
					for (int k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];

						float aw = space_weight[k] + (val - val0)*(val - val0) *gauss_range_coeff;
#if __BF_PREVENTION__
						aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
						float w = exp(aw);
#if __BF_POSTVENTION__
						w = max(aw, FLT_MIN);
#endif

						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
			else if (cn == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					float sum_b = 0.f;
					float sum_g = 0.f;
					float sum_r = 0.f;
					float wsum = 0.f;
					const float b0 = sptr[j + 0];
					const float g0 = sptr[j + 1];
					const float r0 = sptr[j + 2];
					for (int k = 0; k < maxk; k++)
					{
						const float* sptr_k = sptr + j + space_ofs[k];
						const float b = sptr_k[0];
						const float g = sptr_k[1];
						const float r = sptr_k[2];

						float aw = space_weight[k] + ((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)) *gauss_range_coeff;
#if __BF_PREVENTION__
						aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
						float w = exp(aw);
#if __BF_POSTVENTION__
						w = max(aw, FLT_MIN);
#endif

						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[j + 0] = sum_b / wsum;
					dptr[j + 1] = sum_g / wsum;
					dptr[j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	Mat temp;

	Mat *dest;
	int radius, maxk, cn;
	Size size;
	int *space_ofs;
	float *space_weight;
	float sigma_range, sigma_space;
};

class BilateralFilterInvorker_64f : public cv::ParallelLoopBody
{
public:
	BilateralFilterInvorker_64f(const Mat src, Mat& _dest, const int _radius, const double _sigma_range, const double _sigma_space, const int borderType, const int parallel = -1) :
		dest(&_dest), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
		if (_radius == 0)
		{
			src.copyTo(_dest);
			return;
		}
		radius = _radius;

		size = src.size();

		if (dest->empty() || src.type() != dest->type() || src.size() != dest->size())
		{
			*dest = Mat::zeros(src.size(), src.type());
		}

		const double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		cn = src.channels();

		const int d = 2 * radius + 1;
		copyMakeBorder(src, temp, radius, radius, radius, radius, borderType);

		vector<double> _space_weight(d*d);
		vector<int> _space_ofs(d*d);
		space_weight = &_space_weight[0];
		space_ofs = &_space_ofs[0];

		maxk = 0;
		// initialize space-related bilateral filter coefficients
		for (int i = -radius; i <= radius; i++)
		{
			for (int j = -radius; j <= radius; j++)
			{
				const double r = (double)i*i + (double)j*j;
				space_weight[maxk] = r * gauss_space_coeff;
				space_ofs[maxk++] = (int)(i*temp.cols*cn + j * cn);
			}
		}

		parallel_for_(Range(0, size.height), *this, parallel);
	}

	void operator()(const Range& range) const override
	{
		const double gauss_range_coeff = -0.5f / (sigma_range*sigma_range);

		for (int i = range.start; i < range.end; i++)
		{
			const double* sptr = temp.ptr<double>(i + radius) + radius * cn;
			double* dptr = dest->ptr<double>(i);

			if (cn == 1)
			{
				for (int j = 0; j < size.width; j++)
				{
					double sum = 0.f;
					double wsum = 0.f;
					const double val0 = sptr[j];
					for (int k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];

						double aw = space_weight[k] + (val - val0)*(val - val0)*gauss_range_coeff;
#if __BF_PREVENTION__
						aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
						const double w = exp(aw);
#if __BF_POSTVENTION__
						w = max(w, DBL_MIN);
#endif

						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
			else if (cn == 3)
			{
				for (int j = 0; j < size.width * 3; j += 3)
				{
					double sum_b = 0.f;
					double sum_g = 0.f;
					double sum_r = 0.f;
					double wsum = 0.f;
					const double b0 = sptr[j + 0];
					const double g0 = sptr[j + 1];
					const double r0 = sptr[j + 2];
					for (int k = 0; k < maxk; k++)
					{
						const double* sptr_k = sptr + j + space_ofs[k];
						const double b = sptr_k[0];
						const double g = sptr_k[1];
						const double r = sptr_k[2];

						double aw = space_weight[k] + ((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)) *gauss_range_coeff;
#if __BF_PREVENTION__
						aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
						double w = exp(aw);
#if __BF_POSTVENTION__
						w = max(w, DBL_MIN);
#endif

						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[j + 0] = sum_b / wsum;
					dptr[j + 1] = sum_g / wsum;
					dptr[j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	Mat temp;

	Mat *dest;
	int radius, maxk, cn;
	Size size;
	int *space_ofs;
	double *space_weight;
	double sigma_range, sigma_space;
};

namespace bf
{
	void bilateralFilter_Scalar(Mat& src, Mat& dest, const int radius, const double sigma_range, const double sigma_space, const int borderType, const int parallelNum)
	{
		if (src.type() == CV_MAKE_TYPE(CV_8U, src.channels()))
		{
			BilateralFilterInvorker_8u(src, dest, radius, sigma_range, sigma_space, borderType, parallelNum);
		}
		else if (src.type() == CV_MAKE_TYPE(CV_32F, src.channels()))
		{
			BilateralFilterInvorker_32f(src, dest, radius, sigma_range, sigma_space, borderType, parallelNum);
		}
		else if (src.type() == CV_MAKE_TYPE(CV_64F, src.channels()))
		{
			BilateralFilterInvorker_64f(src, dest, radius, sigma_range, sigma_space, borderType, parallelNum);
		}
	}
}
