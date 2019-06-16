#include "cpu_env.h"
#include "color.h"
#include "filter.h"
#include "fmath.hpp"
#include "simd_util.h"

using namespace std;
using namespace cv;


/*************************************************
	using exp function
*************************************************/
class BilateralFilter_EXP_64f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_EXP_64f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double*_space_weight, const double _sigma_range, const double _sigma_space) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const double gauss_range_coeff = -0.5 / (sigma_range*sigma_range);
		const double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			double* sptr = (double*)temp->ptr<double>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d mcw = _mm512_set1_pd(gauss_range_coeff);
					const __m512d msw = _mm512_set1_pd(gauss_space_coeff);
#if __BF_PREVENTION__
					static const __m512d exp_arg_min = _mm512_set1_pd(EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrj = sptr + j;
						const __m512d sval0 = _mm512_load_pd(sptrj);

						__m512d tval = _mm512_setzero_pd();
						__m512d wval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d sref = _mm512_loadu_pd((sptrj + *ofs));

							__m512d diff = _mm512_sub_pd(sval0, sref);
							__m512d difft = _mm512_mul_pd(diff, diff);

							const __m512d _sw = _mm512_mul_pd(_mm512_set1_pd(*spw), msw);
							__m512d aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_pd(difft, mcw, _sw);
#else
								_mm512_add_pd(_mm512_mul_pd(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_pd(aw, exp_arg_min);
#endif
							__m512d _w =
#if __USE_INTEL_EXP__
								_mm512_exp_pd(aw);
#else
								//fmath::exp_pd512(aw); //TODO: exp_pd
								_mm512_set1_pd(1);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_pd(sref, _w, tval);
#else
							sref = _mm512_mul_pd(sref, _w);
							tval = _mm512_add_pd(tval, sref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						tval = _mm512_div_pd(tval, wval);

						_mm512_stream_pd((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double val0 = sptr[j];
					double sum = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];
						const double w = space_weight[k] * exp(gauss_range_coeff*((val - val0)*(val - val0)));
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d mcw = _mm512_set1_pd(gauss_range_coeff);
					const __m512d msw = _mm512_set1_pd(gauss_space_coeff);
#if __BF_PREVENTION__
					static const __m512d exp_arg_min = _mm512_set1_pd(EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrrj = sptrr + j;
						const double* sptrgj = sptrg + j;
						const double* sptrbj = sptrb + j;

						const __m512d bval0 = _mm512_load_pd((sptrbj));
						const __m512d gval0 = _mm512_load_pd((sptrgj));
						const __m512d rval0 = _mm512_load_pd((sptrrj));

						__m512d wval = _mm512_setzero_pd();
						__m512d bval = _mm512_setzero_pd();
						__m512d gval = _mm512_setzero_pd();
						__m512d rval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d bref = _mm512_loadu_pd((sptrbj + *ofs));
							__m512d gref = _mm512_loadu_pd((sptrgj + *ofs));
							__m512d rref = _mm512_loadu_pd((sptrrj + *ofs));

							__m512d bdiff = _mm512_sub_pd(bval0, bref);
							__m512d gdiff = _mm512_sub_pd(gval0, gref);
							__m512d rdiff = _mm512_sub_pd(rval0, rref);

							__m512d  difft = _mm512_mul_pd(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm512_fmadd_pd(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_pd(gdiff, gdiff);
							difft = _mm512_add_pd(difft, gdiff);
							rdiff = _mm512_mul_pd(rdiff, rdiff);
							difft = _mm512_add_pd(difft, rdiff);
#endif
							const __m512d _sw = _mm512_mul_pd(_mm512_set1_pd(*spw), msw);
							__m512d aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_pd(difft, mcw, _sw);
#else
								_mm512_add_pd(_mm512_mul_pd(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_pd(aw, exp_arg_min);
#endif
							__m512d _w =
#if __USE_INTEL_EXP__
								_mm512_exp_pd(aw);
#else
								//fmath::exp_pd512(aw); //TODO: epx_pd
								_mm512_set1_pd(1);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_pd(bref, _w, bval);
							gval = _mm512_fmadd_pd(gref, _w, gval);
							rval = _mm512_fmadd_pd(rref, _w, rval);
#else
							bref = _mm512_mul_pd(bref, _w);
							bval = _mm512_add_pd(bval, bref);
							gref = _mm512_mul_pd(gref, _w);
							gval = _mm512_add_pd(gval, gref);
							rref = _mm512_mul_pd(rref, _w);
							rval = _mm512_add_pd(rval, rref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						bval = _mm512_div_pd(bval, wval);
						gval = _mm512_div_pd(gval, wval);
						rval = _mm512_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm512_stream_pd_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double* sptrrj = sptrr + j;
					const double* sptrgj = sptrg + j;
					const double* sptrbj = sptrb + j;

					const double r0 = sptrrj[0];
					const double g0 = sptrgj[0];
					const double b0 = sptrbj[0];

					double sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const double w = space_weight[k] * exp(gauss_range_coeff*((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *space_weight;
	double sigma_range, sigma_space;
};

class BilateralFilter_EXP_32f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_EXP_32f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float _sigma_range, const float _sigma_space) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);
		const float gauss_space_coeff = -0.5f / (sigma_space*sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcw = _mm512_set1_ps(gauss_range_coeff);
					const __m512 msw = _mm512_set1_ps(gauss_space_coeff);
#if __BF_PREVENTION__
					static const __m512 exp_arg_min = _mm512_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrj = sptr + j;
						const __m512 sval0 = _mm512_load_ps(sptrj);

						__m512 tval = _mm512_setzero_ps();
						__m512 wval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 sref = _mm512_loadu_ps((sptrj + *ofs));

							__m512 diff = _mm512_sub_ps(sval0, sref);
							__m512 difft = _mm512_mul_ps(diff, diff);

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), msw);
							__m512 aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 _w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_ps(sref, _w, tval);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval = _mm512_add_ps(tval, sref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						tval = _mm512_div_ps(tval, wval);

						_mm512_stream_ps((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * exp(gauss_range_coeff*((val - val0)*(val - val0)));
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcw = _mm512_set1_ps(gauss_range_coeff);
					const __m512 msw = _mm512_set1_ps(gauss_space_coeff);
#if __BF_PREVENTION__
					static const __m512 exp_arg_min = _mm512_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;

						const __m512 bval0 = _mm512_load_ps((sptrbj));
						const __m512 gval0 = _mm512_load_ps((sptrgj));
						const __m512 rval0 = _mm512_load_ps((sptrrj));

						__m512 wval = _mm512_setzero_ps();
						__m512 bval = _mm512_setzero_ps();
						__m512 gval = _mm512_setzero_ps();
						__m512 rval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 bref = _mm512_loadu_ps((sptrbj + *ofs));
							__m512 gref = _mm512_loadu_ps((sptrgj + *ofs));
							__m512 rref = _mm512_loadu_ps((sptrrj + *ofs));

							__m512 bdiff = _mm512_sub_ps(bval0, bref);
							__m512 gdiff = _mm512_sub_ps(gval0, gref);
							__m512 rdiff = _mm512_sub_ps(rval0, rref);

							__m512  difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), msw);
							__m512 aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 _w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_ps(bref, _w, bval);
							gval = _mm512_fmadd_ps(gref, _w, gval);
							rval = _mm512_fmadd_ps(rref, _w, rval);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval = _mm512_add_ps(bval, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval = _mm512_add_ps(gval, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval = _mm512_add_ps(rval, rref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						bval = _mm512_div_ps(bval, wval);
						gval = _mm512_div_ps(gval, wval);
						rval = _mm512_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm512_stream_ps_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float* sptrrj = sptrr + j;
					const float* sptrgj = sptrg + j;
					const float* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * exp(gauss_range_coeff*((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight;
	float sigma_range, sigma_space;
};

class BilateralFilter_EXP_8u_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_EXP_8u_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float _sigma_range, const float _sigma_space) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);
		const float gauss_space_coeff = -0.5f / (sigma_space*sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F) && checkHardwareSupport(CV_CPU_AVX_512BW);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 64 * (radiusH / 64 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcw = _mm512_set1_ps(gauss_range_coeff);
					const __m512 msw = _mm512_set1_ps(gauss_space_coeff);
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_PREVENTION__
					static const __m512 exp_arg_min = _mm512_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrj = sptr + j;
						const __m512i sval0_64elem = _mm512_load_si512((__m512i*)sptrj);

						__m512 tval1 = _mm512_setzero_ps();
						__m512 wval1 = _mm512_setzero_ps();
						__m512 tval2 = _mm512_setzero_ps();
						__m512 wval2 = _mm512_setzero_ps();
						__m512 tval3 = _mm512_setzero_ps();
						__m512 wval3 = _mm512_setzero_ps();
						__m512 tval4 = _mm512_setzero_ps();
						__m512 wval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i sref_64elem = _mm512_loadu_si512((__m512i*)(sptrj + *ofs));
							const __m512i diff_64elem = _mm512_add_epi8(_mm512_subs_epu8(sval0_64elem, sref_64elem), _mm512_subs_epu8(sref_64elem, sval0_64elem));

							// lower bit
							__m512i sref_32elem = _mm512_unpacklo_epi8(sref_64elem, zero);
							__m512i diff_32elem = _mm512_unpacklo_epi8(diff_64elem, zero);

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), msw);
							__m512 diff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(diff_32elem, zero));
							__m512 difft = _mm512_mul_ps(diff, diff);

							__m512 aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 _w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							__m512 sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm512_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval1 = _mm512_add_ps(tval1, sref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							diff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(diff_32elem, zero));
							difft = _mm512_mul_ps(diff, diff);

							aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm512_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval2 = _mm512_add_ps(tval2, sref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// higher bit
							sref_32elem = _mm512_unpackhi_epi8(sref_64elem, zero);
							diff_32elem = _mm512_unpackhi_epi8(diff_64elem, zero);

							diff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(diff_32elem, zero));
							difft = _mm512_mul_ps(diff, diff);

							aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(float_min, _w);
#endif

							sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm512_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval3 = _mm512_add_ps(tval3, sref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							diff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(diff_32elem, zero));
							difft = _mm512_mul_ps(diff, diff);

							aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(float_min, _w);
#endif

							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm512_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval4 = _mm512_add_ps(tval4, sref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						tval1 = _mm512_div_ps(tval1, wval1);
						tval2 = _mm512_div_ps(tval2, wval2);
						tval3 = _mm512_div_ps(tval3, wval3);
						tval4 = _mm512_div_ps(tval4, wval4);
						_mm512_stream_si512((__m512i*)(dptr + j), _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(tval1), _mm512_cvtps_epi32(tval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(tval3), _mm512_cvtps_epi32(tval4))));
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * exp(gauss_range_coeff*((val - val0)*(val - val0)));
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (uchar)(sum / wsum);
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 64 * (radiusH / 64 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 64 * (radiusH / 64 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 64 * (radiusH / 64 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcw = _mm512_set1_ps(gauss_range_coeff);
					const __m512 msw = _mm512_set1_ps(gauss_space_coeff);
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_PREVENTION__
					static const __m512 exp_arg_min = _mm512_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m512i bval0_64elem = _mm512_load_si512((__m512i*)(sptrbj));
						const __m512i gval0_64elem = _mm512_load_si512((__m512i*)(sptrgj));
						const __m512i rval0_64elem = _mm512_load_si512((__m512i*)(sptrrj));

						__m512 wval1 = _mm512_setzero_ps();
						__m512 bval1 = _mm512_setzero_ps();
						__m512 gval1 = _mm512_setzero_ps();
						__m512 rval1 = _mm512_setzero_ps();

						__m512 wval2 = _mm512_setzero_ps();
						__m512 bval2 = _mm512_setzero_ps();
						__m512 gval2 = _mm512_setzero_ps();
						__m512 rval2 = _mm512_setzero_ps();

						__m512 wval3 = _mm512_setzero_ps();
						__m512 bval3 = _mm512_setzero_ps();
						__m512 gval3 = _mm512_setzero_ps();
						__m512 rval3 = _mm512_setzero_ps();

						__m512 wval4 = _mm512_setzero_ps();
						__m512 bval4 = _mm512_setzero_ps();
						__m512 gval4 = _mm512_setzero_ps();
						__m512 rval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							// 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, || 
							//16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, || 
							//32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, || 
							//48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
							__m512i bref_64elem = _mm512_loadu_si512((__m512i*)(sptrbj + *ofs));
							__m512i gref_64elem = _mm512_loadu_si512((__m512i*)(sptrgj + *ofs));
							__m512i rref_64elem = _mm512_loadu_si512((__m512i*)(sptrrj + *ofs));

							__m512i bdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(bval0_64elem, bref_64elem), _mm512_subs_epu8(bref_64elem, bval0_64elem));
							__m512i gdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(gval0_64elem, gref_64elem), _mm512_subs_epu8(gref_64elem, gval0_64elem));
							__m512i rdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(rval0_64elem, rref_64elem), _mm512_subs_epu8(rref_64elem, rval0_64elem));


							// lower bit
							// 0,  1,  2,  3,  4,  5,  6,  7, || 
							//16, 17, 18, 19, 20, 21, 22, 23, || 
							//32, 33, 34, 35, 36, 37, 38, 39, || 
							//48, 49, 50, 51, 52, 53, 54, 55 
							__m512i bref_32elem = _mm512_unpacklo_epi8(bref_64elem, zero);
							__m512i gref_32elem = _mm512_unpacklo_epi8(gref_64elem, zero);
							__m512i rref_32elem = _mm512_unpacklo_epi8(rref_64elem, zero);
							__m512i bdiff_32elem = _mm512_unpacklo_epi8(bdiff_64elem, zero);
							__m512i gdiff_32elem = _mm512_unpacklo_epi8(gdiff_64elem, zero);
							__m512i rdiff_32elem = _mm512_unpacklo_epi8(rdiff_64elem, zero);

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), msw);

							// 0,  1,  2,  3, || 
							//16, 17, 18, 19, || 
							//32, 33, 34, 35, || 
							//48, 49, 50, 51 
							__m512 bdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bdiff_32elem, zero));
							__m512 gdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gdiff_32elem, zero));
							__m512 rdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rdiff_32elem, zero));

							__m512 difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, gdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							__m512 aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 _w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							__m512 bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							__m512 gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							__m512 rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm512_fmadd_ps(bref, _w, bval1);
							gval1 = _mm512_fmadd_ps(gref, _w, gval1);
							rval1 = _mm512_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm512_mul_ps(bref, _w);
							gref = _mm512_mul_ps(gref, _w);
							rref = _mm512_mul_ps(rref, _w);
							bval1 = _mm512_add_ps(bval1, bref);
							gval1 = _mm512_add_ps(gval1, gref);
							rval1 = _mm512_add_ps(rval1, rref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							// 4,  5,  6,  7, || 
							//20, 21, 22, 23, || 
							//36, 37, 38, 39, || 
							//52, 53, 54, 55 
							bdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bdiff_32elem, zero));
							gdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gdiff_32elem, zero));
							rdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rdiff_32elem, zero));

							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, gdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm512_fmadd_ps(bref, _w, bval2);
							gval2 = _mm512_fmadd_ps(gref, _w, gval2);
							rval2 = _mm512_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm512_mul_ps(bref, _w);
							gref = _mm512_mul_ps(gref, _w);
							rref = _mm512_mul_ps(rref, _w);
							bval2 = _mm512_add_ps(bval2, bref);
							gval2 = _mm512_add_ps(gval2, gref);
							rval2 = _mm512_add_ps(rval2, rref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);

							// high
							// 8,  9, 10, 11, 12, 13, 14, 15, || 
							//24, 25, 26, 27, 28, 29, 30, 31, || 
							//40, 41, 42, 43, 44, 45, 46, 47, || 
							//56, 57, 58, 59, 60, 61, 62, 63
							bref_32elem = _mm512_unpackhi_epi8(bref_64elem, zero);
							gref_32elem = _mm512_unpackhi_epi8(gref_64elem, zero);
							rref_32elem = _mm512_unpackhi_epi8(rref_64elem, zero);
							bdiff_32elem = _mm512_unpackhi_epi8(bdiff_64elem, zero);
							gdiff_32elem = _mm512_unpackhi_epi8(gdiff_64elem, zero);
							rdiff_32elem = _mm512_unpackhi_epi8(rdiff_64elem, zero);


							// 8,  9, 10, 11, || 
							//24, 25, 26, 27, || 
							//40, 41, 42, 43, || 
							//56, 57, 58, 59,
							bdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bdiff_32elem, zero));
							gdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gdiff_32elem, zero));
							rdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rdiff_32elem, zero));
							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, gdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif

							aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm512_fmadd_ps(bref, _w, bval3);
							gval3 = _mm512_fmadd_ps(gref, _w, gval3);
							rval3 = _mm512_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm512_mul_ps(bref, _w);
							gref = _mm512_mul_ps(gref, _w);
							rref = _mm512_mul_ps(rref, _w);
							bval3 = _mm512_add_ps(bval3, bref);
							gval3 = _mm512_add_ps(gval3, gref);
							rval3 = _mm512_add_ps(rval3, rref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							//12, 13, 14, 15, || 
							//28, 29, 30, 31, || 
							//44, 45, 46, 47, || 
							//60, 61, 62, 63
							bdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bdiff_32elem, zero));
							gdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gdiff_32elem, zero));
							rdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rdiff_32elem, zero));

							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, gdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							aw =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(difft, mcw, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(difft, mcw), _sw);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm512_fmadd_ps(bref, _w, bval4);
							gval4 = _mm512_fmadd_ps(gref, _w, gval4);
							rval4 = _mm512_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm512_mul_ps(bref, _w);
							gref = _mm512_mul_ps(gref, _w);
							rref = _mm512_mul_ps(rref, _w);
							bval4 = _mm512_add_ps(bval4, bref);
							gval4 = _mm512_add_ps(gval4, gref);
							rval4 = _mm512_add_ps(rval4, rref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}

						bval1 = _mm512_div_ps(bval1, wval1);
						bval2 = _mm512_div_ps(bval2, wval2);
						bval3 = _mm512_div_ps(bval3, wval3);
						bval4 = _mm512_div_ps(bval4, wval4);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(bval1), _mm512_cvtps_epi32(bval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(bval3), _mm512_cvtps_epi32(bval4)));
						gval1 = _mm512_div_ps(gval1, wval1);
						gval2 = _mm512_div_ps(gval2, wval2);
						gval3 = _mm512_div_ps(gval3, wval3);
						gval4 = _mm512_div_ps(gval4, wval4);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(gval1), _mm512_cvtps_epi32(gval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(gval3), _mm512_cvtps_epi32(gval4)));
						rval1 = _mm512_div_ps(rval1, wval1);
						rval2 = _mm512_div_ps(rval2, wval2);
						rval3 = _mm512_div_ps(rval3, wval3);
						rval4 = _mm512_div_ps(rval4, wval4);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(rval1), _mm512_cvtps_epi32(rval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(rval3), _mm512_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm512_stream_epi8_color(dptrc, a, b, c);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr + j;
					const uchar* sptrgj = sptrg + j;
					const uchar* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * exp(gauss_range_coeff*((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight;
	float sigma_range, sigma_space;
};


/*************************************************
	using exp function with space LUT
*************************************************/
class BilateralFilter_EXP_With_SpaceLUT_64f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_EXP_With_SpaceLUT_64f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double *_space_weight, const double _sigma_range, const double exp_clip_val = -100000) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), sigma_range(_sigma_range), exp_clip_val(exp_clip_val)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const double gauss_range_coeff = -0.5 / (sigma_range*sigma_range);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			const double* sptr = temp->ptr<double>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d mcw = _mm512_set1_pd(gauss_range_coeff);
#if __BF_PREVENTION__
					const __m512d exp_arg_min = _mm512_set1_pd(exp_clip_val);
#endif
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrj = sptr + j;
						const __m512d sval0 = _mm512_load_pd(sptrj);

						__m512d tval = _mm512_setzero_pd();
						__m512d wval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d sref = _mm512_loadu_pd((sptrj + *ofs));

							__m512d diff = _mm512_sub_pd(sval0, sref);
							__m512d difft = _mm512_mul_pd(diff, diff);

							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d aw = _mm512_mul_pd(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_pd(aw, exp_arg_min);
#endif
							__m512d _rw =
#if __USE_INTEL_EXP__
								_mm512_exp_pd(aw);
#else
								//fmath::exp_ps512(aw); //TODO: exp_pd
								_mm512_set1_pd(1);
#endif
#if __BF_POSTVENTION__
							_rw = _mm512_max_pd(_rw, double_min);
#endif
							__m512d _w = _mm512_mul_pd(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_pd(sref, _w, tval);
#else
							sref = _mm512_mul_pd(sref, _w);
							tval = _mm512_add_pd(tval, sref);
#endif

							wval = _mm512_add_pd(wval, _w);
						}
						tval = _mm512_div_pd(tval, wval);

						_mm512_stream_pd((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double val0 = sptr[j];
					double sum = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];
						const double w = space_weight[k] * exp(gauss_range_coeff*((val - val0)*(val - val0)));
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d mcw = _mm512_set1_pd(gauss_range_coeff);
#if __BF_PREVENTION__
					const __m512d exp_arg_min = _mm512_set1_pd(exp_clip_val);
#endif
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrrj = sptrr + j;
						const double* sptrgj = sptrg + j;
						const double* sptrbj = sptrb + j;

						const __m512d bval0 = _mm512_load_pd((sptrbj));
						const __m512d gval0 = _mm512_load_pd((sptrgj));
						const __m512d rval0 = _mm512_load_pd((sptrrj));

						__m512d wval = _mm512_setzero_pd();
						__m512d bval = _mm512_setzero_pd();
						__m512d gval = _mm512_setzero_pd();
						__m512d rval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d bref = _mm512_loadu_pd((sptrbj + *ofs));
							__m512d gref = _mm512_loadu_pd((sptrgj + *ofs));
							__m512d rref = _mm512_loadu_pd((sptrrj + *ofs));

							__m512d bdiff = _mm512_sub_pd(bval0, bref);
							__m512d gdiff = _mm512_sub_pd(gval0, gref);
							__m512d rdiff = _mm512_sub_pd(rval0, rref);

							__m512d  difft = _mm512_mul_pd(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm512_fmadd_pd(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_pd(gdiff, gdiff);
							difft = _mm512_add_pd(difft, gdiff);
							rdiff = _mm512_mul_pd(rdiff, rdiff);
							difft = _mm512_add_pd(difft, rdiff);
#endif
							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d aw = _mm512_mul_pd(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_pd(aw, exp_arg_min);
#endif
							__m512d _rw =
#if __USE_INTEL_EXP__
								_mm512_exp_pd(aw);
#else
								//fmath::exp_ps512(aw); //TODO: exp_pd
								_mm512_set1_pd(1);
#endif
#if __BF_POSTVENTION__
							_rw = _mm512_max_pd(_rw, double_min);
#endif
							__m512d _w = _mm512_mul_pd(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_pd(bref, _w, bval);
							gval = _mm512_fmadd_pd(gref, _w, gval);
							rval = _mm512_fmadd_pd(rref, _w, rval);
#else
							bref = _mm512_mul_pd(bref, _w);
							bval = _mm512_add_pd(bval, bref);
							gref = _mm512_mul_pd(gref, _w);
							gval = _mm512_add_pd(gval, gref);
							rref = _mm512_mul_pd(rref, _w);
							rval = _mm512_add_pd(rval, rref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						bval = _mm512_div_pd(bval, wval);
						gval = _mm512_div_pd(gval, wval);
						rval = _mm512_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm512_stream_pd_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double* sptrrj = sptrr + j;
					const double* sptrgj = sptrg + j;
					const double* sptrbj = sptrb + j;

					const double r0 = sptrrj[0];
					const double g0 = sptrgj[0];
					const double b0 = sptrbj[0];

					double sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const double w = space_weight[k] * exp(gauss_range_coeff*((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double* space_weight;
	const double sigma_range;
	const double exp_clip_val;
};

class BilateralFilter_EXP_With_SpaceLUT_32f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_EXP_With_SpaceLUT_32f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float _sigma_range, const float exp_clip_val = -200) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), sigma_range(_sigma_range), exp_clip_val(exp_clip_val)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);


#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcw = _mm512_set1_ps(gauss_range_coeff);
#if __BF_PREVENTION__
					const __m512 exp_arg_min = _mm512_set1_ps(exp_clip_val);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrj = sptr + j;
						const __m512 sval0 = _mm512_load_ps(sptrj);

						__m512 tval = _mm512_setzero_ps();
						__m512 wval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 sref = _mm512_loadu_ps((sptrj + *ofs));

							__m512 diff = _mm512_sub_ps(sval0, sref);
							__m512 difft = _mm512_mul_ps(diff, diff);

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 _rw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_rw = _mm512_max_ps(_rw, float_min);
#endif

							__m512 _w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_ps(sref, _w, tval);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval = _mm512_add_ps(tval, sref);
#endif

							wval = _mm512_add_ps(wval, _w);
						}
						tval = _mm512_div_ps(tval, wval);

						_mm512_stream_ps((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * exp(gauss_range_coeff*((val - val0)*(val - val0)));
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcw = _mm512_set1_ps(gauss_range_coeff);
#if __BF_PREVENTION__
					const __m512 exp_arg_min = _mm512_set1_ps(exp_clip_val);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;

						const __m512 bval0 = _mm512_load_ps((sptrbj));
						const __m512 gval0 = _mm512_load_ps((sptrgj));
						const __m512 rval0 = _mm512_load_ps((sptrrj));

						__m512 wval = _mm512_setzero_ps();
						__m512 bval = _mm512_setzero_ps();
						__m512 gval = _mm512_setzero_ps();
						__m512 rval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 bref = _mm512_loadu_ps((sptrbj + *ofs));
							__m512 gref = _mm512_loadu_ps((sptrgj + *ofs));
							__m512 rref = _mm512_loadu_ps((sptrrj + *ofs));

							__m512 bdiff = _mm512_sub_ps(bval0, bref);
							__m512 gdiff = _mm512_sub_ps(gval0, gref);
							__m512 rdiff = _mm512_sub_ps(rval0, rref);

							__m512  difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 _rw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_rw = _mm512_max_ps(_rw, float_min);
#endif
							__m512 _w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_ps(bref, _w, bval);
							gval = _mm512_fmadd_ps(gref, _w, gval);
							rval = _mm512_fmadd_ps(rref, _w, rval);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval = _mm512_add_ps(bval, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval = _mm512_add_ps(gval, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval = _mm512_add_ps(rval, rref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						bval = _mm512_div_ps(bval, wval);
						gval = _mm512_div_ps(gval, wval);
						rval = _mm512_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm512_stream_ps_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float* sptrrj = sptrr + j;
					const float* sptrgj = sptrg + j;
					const float* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * exp(gauss_range_coeff*((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight;
	const float sigma_range;
	const float exp_clip_val;
};

class BilateralFilter_EXP_With_SpaceLUT_8u_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_EXP_With_SpaceLUT_8u_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float _sigma_range, const float exp_clip_val = -200) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), sigma_range(_sigma_range), exp_clip_val(exp_clip_val)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F) && checkHardwareSupport(CV_CPU_AVX_512BW);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 64 * (radiusH / 64 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcw = _mm512_set1_ps(gauss_range_coeff);
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_PREVENTION__
					const __m512 exp_arg_min = _mm512_set1_ps(exp_clip_val);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrj = sptr + j;
						const __m512i sval0_64elem = _mm512_load_si512((__m512i*)sptrj);

						__m512 tval1 = _mm512_setzero_ps();
						__m512 wval1 = _mm512_setzero_ps();
						__m512 tval2 = _mm512_setzero_ps();
						__m512 wval2 = _mm512_setzero_ps();
						__m512 tval3 = _mm512_setzero_ps();
						__m512 wval3 = _mm512_setzero_ps();
						__m512 tval4 = _mm512_setzero_ps();
						__m512 wval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i sref_64elem = _mm512_loadu_si512((__m512i*)(sptrj + *ofs));
							const __m512i diff_64elem = _mm512_add_epi8(_mm512_subs_epu8(sval0_64elem, sref_64elem), _mm512_subs_epu8(sref_64elem, sval0_64elem));

							// lower bit
							__m512i sref_32elem = _mm512_unpacklo_epi8(sref_64elem, zero);
							__m512i diff_32elem = _mm512_unpacklo_epi8(diff_64elem, zero);

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 diff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(diff_32elem, zero));
							__m512 difft = _mm512_mul_ps(diff, diff);

							__m512 aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 _rw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_rw = _mm512_max_ps(_rw, float_min);
#endif
							__m512 _w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							__m512 sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm512_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval1 = _mm512_add_ps(tval1, sref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							diff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(diff_32elem, zero));
							difft = _mm512_mul_ps(diff, diff);

							aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_rw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_rw = _mm512_max_ps(_rw, float_min);
#endif
							_w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm512_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval2 = _mm512_add_ps(tval2, sref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// higher bit
							sref_32elem = _mm512_unpackhi_epi8(sref_64elem, zero);
							diff_32elem = _mm512_unpackhi_epi8(diff_64elem, zero);

							diff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(diff_32elem, zero));
							difft = _mm512_mul_ps(diff, diff);

							aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_rw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_rw = _mm512_max_ps(_rw, float_min);
#endif
							_w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(float_min, _w);
#endif

							sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm512_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval3 = _mm512_add_ps(tval3, sref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							diff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(diff_32elem, zero));
							difft = _mm512_mul_ps(diff, diff);

							aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_rw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_rw = _mm512_max_ps(_rw, float_min);
#endif
							_w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(float_min, _w);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm512_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval4 = _mm512_add_ps(tval4, sref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						tval1 = _mm512_div_ps(tval1, wval1);
						tval2 = _mm512_div_ps(tval2, wval2);
						tval3 = _mm512_div_ps(tval3, wval3);
						tval4 = _mm512_div_ps(tval4, wval4);
						_mm512_stream_si512((__m512i*)(dptr + j), _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(tval1), _mm512_cvtps_epi32(tval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(tval3), _mm512_cvtps_epi32(tval4))));
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * exp(gauss_range_coeff*((val - val0)*(val - val0)));
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (uchar)(sum / wsum);
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 64 * (radiusH / 64 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 64 * (radiusH / 64 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 64 * (radiusH / 64 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
					const __m512 mcw = _mm512_set1_ps(gauss_range_coeff);
#if __BF_PREVENTION__
					const __m512 exp_arg_min = _mm512_set1_ps(exp_clip_val);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m512i bval0_64elem = _mm512_load_si512((__m512i*)(sptrbj));
						const __m512i gval0_64elem = _mm512_load_si512((__m512i*)(sptrgj));
						const __m512i rval0_64elem = _mm512_load_si512((__m512i*)(sptrrj));

						__m512 wval1 = _mm512_setzero_ps();
						__m512 bval1 = _mm512_setzero_ps();
						__m512 gval1 = _mm512_setzero_ps();
						__m512 rval1 = _mm512_setzero_ps();

						__m512 wval2 = _mm512_setzero_ps();
						__m512 bval2 = _mm512_setzero_ps();
						__m512 gval2 = _mm512_setzero_ps();
						__m512 rval2 = _mm512_setzero_ps();

						__m512 wval3 = _mm512_setzero_ps();
						__m512 bval3 = _mm512_setzero_ps();
						__m512 gval3 = _mm512_setzero_ps();
						__m512 rval3 = _mm512_setzero_ps();

						__m512 wval4 = _mm512_setzero_ps();
						__m512 bval4 = _mm512_setzero_ps();
						__m512 gval4 = _mm512_setzero_ps();
						__m512 rval4 = _mm512_setzero_ps();


						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							// 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, || 
							//16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, || 
							//32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, || 
							//48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
							__m512i bref_64elem = _mm512_loadu_si512((__m512i*)(sptrbj + *ofs));
							__m512i gref_64elem = _mm512_loadu_si512((__m512i*)(sptrgj + *ofs));
							__m512i rref_64elem = _mm512_loadu_si512((__m512i*)(sptrrj + *ofs));

							__m512i bdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(bval0_64elem, bref_64elem), _mm512_subs_epu8(bref_64elem, bval0_64elem));
							__m512i gdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(gval0_64elem, gref_64elem), _mm512_subs_epu8(gref_64elem, gval0_64elem));
							__m512i rdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(rval0_64elem, rref_64elem), _mm512_subs_epu8(rref_64elem, rval0_64elem));


							// lower bit
							// 0,  1,  2,  3,  4,  5,  6,  7, || 
							//16, 17, 18, 19, 20, 21, 22, 23, || 
							//32, 33, 34, 35, 36, 37, 38, 39, || 
							//48, 49, 50, 51, 52, 53, 54, 55 
							__m512i bref_32elem = _mm512_unpacklo_epi8(bref_64elem, zero);
							__m512i gref_32elem = _mm512_unpacklo_epi8(gref_64elem, zero);
							__m512i rref_32elem = _mm512_unpacklo_epi8(rref_64elem, zero);
							__m512i bdiff_32elem = _mm512_unpacklo_epi8(bdiff_64elem, zero);
							__m512i gdiff_32elem = _mm512_unpacklo_epi8(gdiff_64elem, zero);
							__m512i rdiff_32elem = _mm512_unpacklo_epi8(rdiff_64elem, zero);

							const __m512 _sw = _mm512_set1_ps(*spw);

							// 0,  1,  2,  3, || 
							//16, 17, 18, 19, || 
							//32, 33, 34, 35, || 
							//48, 49, 50, 51 
							__m512 bdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bdiff_32elem, zero));
							__m512 gdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gdiff_32elem, zero));
							__m512 rdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rdiff_32elem, zero));

							__m512 difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, gdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							__m512 aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 _cw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_cw = _mm512_max_ps(_cw, float_min);
#endif
							__m512 _w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							__m512 bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							__m512 gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							__m512 rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm512_fmadd_ps(bref, _w, bval1);
							gval1 = _mm512_fmadd_ps(gref, _w, gval1);
							rval1 = _mm512_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm512_mul_ps(bref, _w);
							gref = _mm512_mul_ps(gref, _w);
							rref = _mm512_mul_ps(rref, _w);
							bval1 = _mm512_add_ps(bval1, bref);
							gval1 = _mm512_add_ps(gval1, gref);
							rval1 = _mm512_add_ps(rval1, rref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							// 4,  5,  6,  7, || 
							//20, 21, 22, 23, || 
							//36, 37, 38, 39, || 
							//52, 53, 54, 55 
							bdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bdiff_32elem, zero));
							gdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gdiff_32elem, zero));
							rdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rdiff_32elem, zero));

							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, gdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_cw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_cw = _mm512_max_ps(_cw, float_min);
#endif
							_w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm512_fmadd_ps(bref, _w, bval2);
							gval2 = _mm512_fmadd_ps(gref, _w, gval2);
							rval2 = _mm512_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm512_mul_ps(bref, _w);
							gref = _mm512_mul_ps(gref, _w);
							rref = _mm512_mul_ps(rref, _w);
							bval2 = _mm512_add_ps(bval2, bref);
							gval2 = _mm512_add_ps(gval2, gref);
							rval2 = _mm512_add_ps(rval2, rref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);

							// high
							// 8,  9, 10, 11, 12, 13, 14, 15, || 
							//24, 25, 26, 27, 28, 29, 30, 31, || 
							//40, 41, 42, 43, 44, 45, 46, 47, || 
							//56, 57, 58, 59, 60, 61, 62, 63
							bref_32elem = _mm512_unpackhi_epi8(bref_64elem, zero);
							gref_32elem = _mm512_unpackhi_epi8(gref_64elem, zero);
							rref_32elem = _mm512_unpackhi_epi8(rref_64elem, zero);
							bdiff_32elem = _mm512_unpackhi_epi8(bdiff_64elem, zero);
							gdiff_32elem = _mm512_unpackhi_epi8(gdiff_64elem, zero);
							rdiff_32elem = _mm512_unpackhi_epi8(rdiff_64elem, zero);


							// 8,  9, 10, 11, || 
							//24, 25, 26, 27, || 
							//40, 41, 42, 43, || 
							//56, 57, 58, 59,
							bdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bdiff_32elem, zero));
							gdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gdiff_32elem, zero));
							rdiff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rdiff_32elem, zero));
							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, gdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif

							aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_cw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_cw = _mm512_max_ps(_cw, float_min);
#endif
							_w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm512_fmadd_ps(bref, _w, bval3);
							gval3 = _mm512_fmadd_ps(gref, _w, gval3);
							rval3 = _mm512_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm512_mul_ps(bref, _w);
							gref = _mm512_mul_ps(gref, _w);
							rref = _mm512_mul_ps(rref, _w);
							bval3 = _mm512_add_ps(bval3, bref);
							gval3 = _mm512_add_ps(gval3, gref);
							rval3 = _mm512_add_ps(rval3, rref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							//12, 13, 14, 15, || 
							//28, 29, 30, 31, || 
							//44, 45, 46, 47, || 
							//60, 61, 62, 63
							bdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bdiff_32elem, zero));
							gdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gdiff_32elem, zero));
							rdiff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rdiff_32elem, zero));

							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, gdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							aw = _mm512_mul_ps(difft, mcw);
#if __BF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							_cw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __BF_POSTVENTION__
							_cw = _mm512_max_ps(_cw, float_min);
#endif
							_w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm512_fmadd_ps(bref, _w, bval4);
							gval4 = _mm512_fmadd_ps(gref, _w, gval4);
							rval4 = _mm512_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm512_mul_ps(bref, _w);
							gref = _mm512_mul_ps(gref, _w);
							rref = _mm512_mul_ps(rref, _w);
							bval4 = _mm512_add_ps(bval4, bref);
							gval4 = _mm512_add_ps(gval4, gref);
							rval4 = _mm512_add_ps(rval4, rref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}

						bval1 = _mm512_div_ps(bval1, wval1);
						bval2 = _mm512_div_ps(bval2, wval2);
						bval3 = _mm512_div_ps(bval3, wval3);
						bval4 = _mm512_div_ps(bval4, wval4);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(bval1), _mm512_cvtps_epi32(bval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(bval3), _mm512_cvtps_epi32(bval4)));
						gval1 = _mm512_div_ps(gval1, wval1);
						gval2 = _mm512_div_ps(gval2, wval2);
						gval3 = _mm512_div_ps(gval3, wval3);
						gval4 = _mm512_div_ps(gval4, wval4);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(gval1), _mm512_cvtps_epi32(gval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(gval3), _mm512_cvtps_epi32(gval4)));
						rval1 = _mm512_div_ps(rval1, wval1);
						rval2 = _mm512_div_ps(rval2, wval2);
						rval3 = _mm512_div_ps(rval3, wval3);
						rval4 = _mm512_div_ps(rval4, wval4);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(rval1), _mm512_cvtps_epi32(rval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(rval3), _mm512_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm512_stream_epi8_color(dptrc, a, b, c);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr + j;
					const uchar* sptrgj = sptrg + j;
					const uchar* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * exp(gauss_range_coeff*((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight;
	const float sigma_range;
	const float exp_clip_val;
};


/*************************************************
	using LUT with "set instruction" x 3
*************************************************/
class BilateralFilter_LUT_Setx3_64f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_LUT_Setx3_64f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double *_space_weight, const double *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		static const long long CV_DECL_ALIGNED(64) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff,
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff
		};
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			long long CV_DECL_ALIGNED(64) buf[8];

			const double* sptr = temp->ptr<double>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(FLT_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrj = sptr + j;
						const __m512d sval0 = _mm512_load_pd(sptrj);

						__m512d tval = _mm512_setzero_pd();
						__m512d wval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d sref = _mm512_loadu_pd((sptrj + *ofs));
							_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(sval0, sref), *(const __m512d*)v64f_absmask)));

							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d _w = _mm512_mul_pd(
								_sw,
								_mm512_set_pd(
									range_weight[buf[7]],
									range_weight[buf[6]],
									range_weight[buf[5]],
									range_weight[buf[4]],
									range_weight[buf[3]],
									range_weight[buf[2]],
									range_weight[buf[1]],
									range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_pd(sref, _w, tval);
#else
							sref = _mm512_mul_pd(sref, _w);
							tval = _mm512_add_pd(tval, sref);
#endif

							wval = _mm512_add_pd(wval, _w);
						}
						tval = _mm512_div_pd(tval, wval);
						_mm512_stream_pd((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double val0 = sptr[j];
					double sum = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];
						const double w = space_weight[k] * range_weight[(uchar)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			long long CV_DECL_ALIGNED(64) rbuf[8];
			long long CV_DECL_ALIGNED(64) gbuf[8];
			long long CV_DECL_ALIGNED(64) bbuf[8];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m512d bval0 = _mm512_load_pd(sptrbj);
						const __m512d gval0 = _mm512_load_pd(sptrgj);
						const __m512d rval0 = _mm512_load_pd(sptrrj);

						__m512d wval = _mm512_setzero_pd();
						__m512d bval = _mm512_setzero_pd();
						__m512d gval = _mm512_setzero_pd();
						__m512d rval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d bref = _mm512_loadu_pd(sptrbj + *ofs);
							__m512d gref = _mm512_loadu_pd(sptrgj + *ofs);
							__m512d rref = _mm512_loadu_pd(sptrrj + *ofs);
							_mm512_store_si512((__m512i*)bbuf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(bval0, bref), *(const __m512d*)v64f_absmask)));
							_mm512_store_si512((__m512i*)gbuf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(gval0, gref), *(const __m512d*)v64f_absmask)));
							_mm512_store_si512((__m512i*)rbuf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(rval0, rref), *(const __m512d*)v64f_absmask)));

							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d _w = _mm512_mul_pd(
								_sw,
								_mm512_set_pd(
									range_weight[bbuf[7]],
									range_weight[bbuf[6]],
									range_weight[bbuf[5]],
									range_weight[bbuf[4]],
									range_weight[bbuf[3]],
									range_weight[bbuf[2]],
									range_weight[bbuf[1]],
									range_weight[bbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
							_w = _mm512_mul_pd(
								_w,
								_mm512_set_pd(
									range_weight[gbuf[7]],
									range_weight[gbuf[6]],
									range_weight[gbuf[5]],
									range_weight[gbuf[4]],
									range_weight[gbuf[3]],
									range_weight[gbuf[2]],
									range_weight[gbuf[1]],
									range_weight[gbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
							_w = _mm512_mul_pd(
								_w,
								_mm512_set_pd(
									range_weight[rbuf[7]],
									range_weight[rbuf[6]],
									range_weight[rbuf[5]],
									range_weight[rbuf[4]],
									range_weight[rbuf[3]],
									range_weight[rbuf[2]],
									range_weight[rbuf[1]],
									range_weight[rbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_pd(bref, _w, bval);
							gval = _mm512_fmadd_pd(gref, _w, gval);
							rval = _mm512_fmadd_pd(rref, _w, rval);
#else
							bref = _mm512_mul_pd(bref, _w);
							bval = _mm512_add_pd(bval, bref);
							gref = _mm512_mul_pd(gref, _w);
							gval = _mm512_add_pd(gval, gref);
							rref = _mm512_mul_pd(rref, _w);
							rval = _mm512_add_pd(rval, rref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						bval = _mm512_div_pd(bval, wval);
						gval = _mm512_div_pd(gval, wval);
						rval = _mm512_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm512_stream_pd_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double* sptrrj = sptrr + j;
					const double* sptrgj = sptrg + j;
					const double* sptrbj = sptrb + j;

					const double r0 = sptrrj[0];
					const double g0 = sptrgj[0];
					const double b0 = sptrbj[0];

					double sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const double w = space_weight[k] * range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)abs(r - r0)];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *space_weight, *range_weight;
};

class BilateralFilter_LUT_Setx3_32f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_LUT_Setx3_32f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(64) buf[16];

			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrj = sptr + j;
						const __m512 sval0 = _mm512_load_ps(sptrj);

						__m512 tval = _mm512_setzero_ps();
						__m512 wval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 sref = _mm512_loadu_ps((sptrj + *ofs));
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(sval0, sref), *(const __m512*)v32f_absmask)));

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[15]],
									range_weight[buf[14]],
									range_weight[buf[13]],
									range_weight[buf[12]],
									range_weight[buf[11]],
									range_weight[buf[10]],
									range_weight[buf[9]],
									range_weight[buf[8]],
									range_weight[buf[7]],
									range_weight[buf[6]],
									range_weight[buf[5]],
									range_weight[buf[4]],
									range_weight[buf[3]],
									range_weight[buf[2]],
									range_weight[buf[1]],
									range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_ps(sref, _w, tval);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval = _mm512_add_ps(tval, sref);
#endif

							wval = _mm512_add_ps(wval, _w);
						}
						tval = _mm512_div_ps(tval, wval);
						_mm512_stream_ps((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[(uchar)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(64) rbuf[16];
			int CV_DECL_ALIGNED(64) gbuf[16];
			int CV_DECL_ALIGNED(64) bbuf[16];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m512 bval0 = _mm512_load_ps(sptrbj);
						const __m512 gval0 = _mm512_load_ps(sptrgj);
						const __m512 rval0 = _mm512_load_ps(sptrrj);

						__m512 wval = _mm512_setzero_ps();
						__m512 bval = _mm512_setzero_ps();
						__m512 gval = _mm512_setzero_ps();
						__m512 rval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 bref = _mm512_load_ps(sptrbj + *ofs);
							__m512 gref = _mm512_load_ps(sptrgj + *ofs);
							__m512 rref = _mm512_load_ps(sptrrj + *ofs);
							_mm512_store_si512((__m512i*)bbuf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(bval0, bref), *(const __m512*)v32f_absmask)));
							_mm512_store_si512((__m512i*)gbuf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(gval0, gref), *(const __m512*)v32f_absmask)));
							_mm512_store_si512((__m512i*)rbuf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(rval0, rref), *(const __m512*)v32f_absmask)));

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[bbuf[15]],
									range_weight[bbuf[14]],
									range_weight[bbuf[13]],
									range_weight[bbuf[12]],
									range_weight[bbuf[11]],
									range_weight[bbuf[10]],
									range_weight[bbuf[9]],
									range_weight[bbuf[8]],
									range_weight[bbuf[7]],
									range_weight[bbuf[6]],
									range_weight[bbuf[5]],
									range_weight[bbuf[4]],
									range_weight[bbuf[3]],
									range_weight[bbuf[2]],
									range_weight[bbuf[1]],
									range_weight[bbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[gbuf[15]],
									range_weight[gbuf[14]],
									range_weight[gbuf[13]],
									range_weight[gbuf[12]],
									range_weight[gbuf[11]],
									range_weight[gbuf[10]],
									range_weight[gbuf[9]],
									range_weight[gbuf[8]],
									range_weight[gbuf[7]],
									range_weight[gbuf[6]],
									range_weight[gbuf[5]],
									range_weight[gbuf[4]],
									range_weight[gbuf[3]],
									range_weight[gbuf[2]],
									range_weight[gbuf[1]],
									range_weight[gbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[rbuf[15]],
									range_weight[rbuf[14]],
									range_weight[rbuf[13]],
									range_weight[rbuf[12]],
									range_weight[rbuf[11]],
									range_weight[rbuf[10]],
									range_weight[rbuf[9]],
									range_weight[rbuf[8]],
									range_weight[rbuf[7]],
									range_weight[rbuf[6]],
									range_weight[rbuf[5]],
									range_weight[rbuf[4]],
									range_weight[rbuf[3]],
									range_weight[rbuf[2]],
									range_weight[rbuf[1]],
									range_weight[rbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_ps(bref, _w, bval);
							gval = _mm512_fmadd_ps(gref, _w, gval);
							rval = _mm512_fmadd_ps(rref, _w, rval);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval = _mm512_add_ps(bval, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval = _mm512_add_ps(gval, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval = _mm512_add_ps(rval, rref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						bval = _mm512_div_ps(bval, wval);
						gval = _mm512_div_ps(gval, wval);
						rval = _mm512_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm512_stream_ps_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float* sptrrj = sptrr + j;
					const float* sptrgj = sptrg + j;
					const float* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)abs(r - r0)];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
};

class BilateralFilter_LUT_Setx3_8u_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_LUT_Setx3_8u_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F) && checkHardwareSupport(CV_CPU_AVX_512BW);
#endif
		if (cn == 1)
		{
			uchar CV_DECL_ALIGNED(64) buf[64];

			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 64 * (radiusH / 64 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrj = sptr + j;
						const __m512i sval0_64elem = _mm512_load_si512((__m512i*)sptrj);

						__m512 tval1 = _mm512_setzero_ps();
						__m512 wval1 = _mm512_setzero_ps();
						__m512 tval2 = _mm512_setzero_ps();
						__m512 wval2 = _mm512_setzero_ps();
						__m512 tval3 = _mm512_setzero_ps();
						__m512 wval3 = _mm512_setzero_ps();
						__m512 tval4 = _mm512_setzero_ps();
						__m512 wval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512i sref_64elem = _mm512_loadu_si512((__m512i*)(sptrj + *ofs));
							_mm512_store_si512((__m512i*)buf, _mm512_add_epi8(_mm512_subs_epu8(sval0_64elem, sref_64elem), _mm512_subs_epu8(sref_64elem, sval0_64elem)));

							// lower bit
							__m512i sref_32elem = _mm512_unpacklo_epi8(sref_64elem, zero);

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[51]], range_weight[buf[50]], range_weight[buf[49]], range_weight[buf[48]],
									range_weight[buf[35]], range_weight[buf[34]], range_weight[buf[33]], range_weight[buf[32]],
									range_weight[buf[19]], range_weight[buf[18]], range_weight[buf[17]], range_weight[buf[16]],
									range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							__m512 sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm512_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval1 = _mm512_add_ps(tval1, sref);
#endif

							wval1 = _mm512_add_ps(wval1, _w);

							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[55]], range_weight[buf[54]], range_weight[buf[53]], range_weight[buf[52]],
									range_weight[buf[39]], range_weight[buf[38]], range_weight[buf[37]], range_weight[buf[36]],
									range_weight[buf[23]], range_weight[buf[22]], range_weight[buf[21]], range_weight[buf[20]],
									range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm512_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval2 = _mm512_add_ps(tval2, sref);
#endif

							wval2 = _mm512_add_ps(wval2, _w);

							// higher bit
							sref_32elem = _mm512_unpackhi_epi8(sref_64elem, zero);

							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[59]], range_weight[buf[58]], range_weight[buf[57]], range_weight[buf[56]],
									range_weight[buf[43]], range_weight[buf[42]], range_weight[buf[41]], range_weight[buf[40]],
									range_weight[buf[27]], range_weight[buf[26]], range_weight[buf[25]], range_weight[buf[24]],
									range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm512_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval3 = _mm512_add_ps(tval3, sref);
#endif

							wval3 = _mm512_add_ps(wval3, _w);

							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[63]], range_weight[buf[62]], range_weight[buf[61]], range_weight[buf[60]],
									range_weight[buf[47]], range_weight[buf[46]], range_weight[buf[45]], range_weight[buf[44]],
									range_weight[buf[31]], range_weight[buf[30]], range_weight[buf[29]], range_weight[buf[28]],
									range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(float_min, _w);
#endif

							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm512_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval4 = _mm512_add_ps(tval4, sref);
#endif

							wval4 = _mm512_add_ps(wval4, _w);
						}
						tval1 = _mm512_div_ps(tval1, wval1);
						tval2 = _mm512_div_ps(tval2, wval2);
						tval3 = _mm512_div_ps(tval3, wval3);
						tval4 = _mm512_div_ps(tval4, wval4);
						_mm512_stream_si512((__m512i*)(dptr + j), _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(tval1), _mm512_cvtps_epi32(tval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(tval3), _mm512_cvtps_epi32(tval4))));
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[(uchar)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)(sum / wsum);
				}
			}
		}
		else
		{
			uchar CV_DECL_ALIGNED(64) rbuf[64];
			uchar CV_DECL_ALIGNED(64) gbuf[64];
			uchar CV_DECL_ALIGNED(64) bbuf[64];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 64 * (radiusH / 64 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 64 * (radiusH / 64 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 64 * (radiusH / 64 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m512i bval0_64elem = _mm512_load_si512((__m512i*)(sptrbj));
						const __m512i gval0_64elem = _mm512_load_si512((__m512i*)(sptrgj));
						const __m512i rval0_64elem = _mm512_load_si512((__m512i*)(sptrrj));

						__m512 wval1 = _mm512_setzero_ps();
						__m512 bval1 = _mm512_setzero_ps();
						__m512 gval1 = _mm512_setzero_ps();
						__m512 rval1 = _mm512_setzero_ps();

						__m512 wval2 = _mm512_setzero_ps();
						__m512 bval2 = _mm512_setzero_ps();
						__m512 gval2 = _mm512_setzero_ps();
						__m512 rval2 = _mm512_setzero_ps();

						__m512 wval3 = _mm512_setzero_ps();
						__m512 bval3 = _mm512_setzero_ps();
						__m512 gval3 = _mm512_setzero_ps();
						__m512 rval3 = _mm512_setzero_ps();

						__m512 wval4 = _mm512_setzero_ps();
						__m512 bval4 = _mm512_setzero_ps();
						__m512 gval4 = _mm512_setzero_ps();
						__m512 rval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512i bref_64elem = _mm512_loadu_si512((__m512i*)(sptrbj + *ofs));
							__m512i gref_64elem = _mm512_loadu_si512((__m512i*)(sptrgj + *ofs));
							__m512i rref_64elem = _mm512_loadu_si512((__m512i*)(sptrrj + *ofs));
							_mm512_store_si512((__m512i*)bbuf, _mm512_add_epi8(_mm512_subs_epu8(bval0_64elem, bref_64elem), _mm512_subs_epu8(bref_64elem, bval0_64elem)));
							_mm512_store_si512((__m512i*)gbuf, _mm512_add_epi8(_mm512_subs_epu8(gval0_64elem, gref_64elem), _mm512_subs_epu8(gref_64elem, gval0_64elem)));
							_mm512_store_si512((__m512i*)rbuf, _mm512_add_epi8(_mm512_subs_epu8(rval0_64elem, rref_64elem), _mm512_subs_epu8(rref_64elem, rval0_64elem)));

							// lower bit
							__m512i bref_32elem = _mm512_unpacklo_epi8(bref_64elem, zero);
							__m512i gref_32elem = _mm512_unpacklo_epi8(gref_64elem, zero);
							__m512i rref_32elem = _mm512_unpacklo_epi8(rref_64elem, zero);

							__m512 _sw = _mm512_set1_ps(*spw);

							__m512 _w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[bbuf[51]], range_weight[bbuf[50]], range_weight[bbuf[49]], range_weight[bbuf[48]],
									range_weight[bbuf[35]], range_weight[bbuf[34]], range_weight[bbuf[33]], range_weight[bbuf[32]],
									range_weight[bbuf[19]], range_weight[bbuf[18]], range_weight[bbuf[17]], range_weight[bbuf[16]],
									range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[gbuf[51]], range_weight[gbuf[50]], range_weight[gbuf[49]], range_weight[gbuf[48]],
									range_weight[gbuf[35]], range_weight[gbuf[34]], range_weight[gbuf[33]], range_weight[gbuf[32]],
									range_weight[gbuf[19]], range_weight[gbuf[18]], range_weight[gbuf[17]], range_weight[gbuf[16]],
									range_weight[gbuf[3]], range_weight[gbuf[2]], range_weight[gbuf[1]], range_weight[gbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[rbuf[51]], range_weight[rbuf[50]], range_weight[rbuf[49]], range_weight[rbuf[48]],
									range_weight[rbuf[35]], range_weight[rbuf[34]], range_weight[rbuf[33]], range_weight[rbuf[32]],
									range_weight[rbuf[19]], range_weight[rbuf[18]], range_weight[rbuf[17]], range_weight[rbuf[16]],
									range_weight[rbuf[3]], range_weight[rbuf[2]], range_weight[rbuf[1]], range_weight[rbuf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							__m512 bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							__m512 gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							__m512 rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm512_fmadd_ps(bref, _w, bval1);
							gval1 = _mm512_fmadd_ps(gref, _w, gval1);
							rval1 = _mm512_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval1 = _mm512_add_ps(bval1, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval1 = _mm512_add_ps(gval1, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval1 = _mm512_add_ps(rval1, rref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[bbuf[55]], range_weight[bbuf[54]], range_weight[bbuf[53]], range_weight[bbuf[52]],
									range_weight[bbuf[39]], range_weight[bbuf[38]], range_weight[bbuf[37]], range_weight[bbuf[36]],
									range_weight[bbuf[23]], range_weight[bbuf[22]], range_weight[bbuf[21]], range_weight[bbuf[20]],
									range_weight[bbuf[7]], range_weight[bbuf[6]], range_weight[bbuf[5]], range_weight[bbuf[4]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[gbuf[55]], range_weight[gbuf[54]], range_weight[gbuf[53]], range_weight[gbuf[52]],
									range_weight[gbuf[39]], range_weight[gbuf[38]], range_weight[gbuf[37]], range_weight[gbuf[36]],
									range_weight[gbuf[23]], range_weight[gbuf[22]], range_weight[gbuf[21]], range_weight[gbuf[20]],
									range_weight[gbuf[7]], range_weight[gbuf[6]], range_weight[gbuf[5]], range_weight[gbuf[4]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[rbuf[55]], range_weight[rbuf[54]], range_weight[rbuf[53]], range_weight[rbuf[52]],
									range_weight[rbuf[39]], range_weight[rbuf[38]], range_weight[rbuf[37]], range_weight[rbuf[36]],
									range_weight[rbuf[23]], range_weight[rbuf[22]], range_weight[rbuf[21]], range_weight[rbuf[20]],
									range_weight[rbuf[7]], range_weight[rbuf[6]], range_weight[rbuf[5]], range_weight[rbuf[4]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm512_fmadd_ps(bref, _w, bval2);
							gval2 = _mm512_fmadd_ps(gref, _w, gval2);
							rval2 = _mm512_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval2 = _mm512_add_ps(bval2, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval2 = _mm512_add_ps(gval2, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval2 = _mm512_add_ps(rval2, rref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// higher bit
							bref_32elem = _mm512_unpackhi_epi8(bref_64elem, zero);
							gref_32elem = _mm512_unpackhi_epi8(gref_64elem, zero);
							rref_32elem = _mm512_unpackhi_epi8(rref_64elem, zero);

							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[bbuf[59]], range_weight[bbuf[58]], range_weight[bbuf[57]], range_weight[bbuf[56]],
									range_weight[bbuf[43]], range_weight[bbuf[42]], range_weight[bbuf[41]], range_weight[bbuf[40]],
									range_weight[bbuf[27]], range_weight[bbuf[26]], range_weight[bbuf[25]], range_weight[bbuf[24]],
									range_weight[bbuf[11]], range_weight[bbuf[10]], range_weight[bbuf[9]], range_weight[bbuf[8]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[gbuf[59]], range_weight[gbuf[58]], range_weight[gbuf[57]], range_weight[gbuf[56]],
									range_weight[gbuf[43]], range_weight[gbuf[42]], range_weight[gbuf[41]], range_weight[gbuf[40]],
									range_weight[gbuf[27]], range_weight[gbuf[26]], range_weight[gbuf[25]], range_weight[gbuf[24]],
									range_weight[gbuf[11]], range_weight[gbuf[10]], range_weight[gbuf[9]], range_weight[gbuf[8]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[rbuf[59]], range_weight[rbuf[58]], range_weight[rbuf[57]], range_weight[rbuf[56]],
									range_weight[rbuf[43]], range_weight[rbuf[42]], range_weight[rbuf[41]], range_weight[rbuf[40]],
									range_weight[rbuf[27]], range_weight[rbuf[26]], range_weight[rbuf[25]], range_weight[rbuf[24]],
									range_weight[rbuf[11]], range_weight[rbuf[10]], range_weight[rbuf[9]], range_weight[rbuf[8]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm512_fmadd_ps(bref, _w, bval3);
							gval3 = _mm512_fmadd_ps(gref, _w, gval3);
							rval3 = _mm512_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval3 = _mm512_add_ps(bval3, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval3 = _mm512_add_ps(gval3, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval3 = _mm512_add_ps(rval3, rref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[bbuf[63]], range_weight[bbuf[62]], range_weight[bbuf[61]], range_weight[bbuf[60]],
									range_weight[bbuf[47]], range_weight[bbuf[46]], range_weight[bbuf[45]], range_weight[bbuf[44]],
									range_weight[bbuf[31]], range_weight[bbuf[30]], range_weight[bbuf[29]], range_weight[bbuf[28]],
									range_weight[bbuf[15]], range_weight[bbuf[14]], range_weight[bbuf[13]], range_weight[bbuf[12]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[gbuf[63]], range_weight[gbuf[62]], range_weight[gbuf[61]], range_weight[gbuf[60]],
									range_weight[gbuf[47]], range_weight[gbuf[46]], range_weight[gbuf[45]], range_weight[gbuf[44]],
									range_weight[gbuf[31]], range_weight[gbuf[30]], range_weight[gbuf[29]], range_weight[gbuf[28]],
									range_weight[gbuf[15]], range_weight[gbuf[14]], range_weight[gbuf[13]], range_weight[gbuf[12]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(
								_w,
								_mm512_set_ps(
									range_weight[rbuf[63]], range_weight[rbuf[62]], range_weight[rbuf[61]], range_weight[rbuf[60]],
									range_weight[rbuf[47]], range_weight[rbuf[46]], range_weight[rbuf[45]], range_weight[rbuf[44]],
									range_weight[rbuf[31]], range_weight[rbuf[30]], range_weight[rbuf[29]], range_weight[rbuf[28]],
									range_weight[rbuf[15]], range_weight[rbuf[14]], range_weight[rbuf[13]], range_weight[rbuf[12]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm512_fmadd_ps(bref, _w, bval4);
							gval4 = _mm512_fmadd_ps(gref, _w, gval4);
							rval4 = _mm512_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval4 = _mm512_add_ps(bval4, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval4 = _mm512_add_ps(gval4, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval4 = _mm512_add_ps(rval4, rref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						bval1 = _mm512_div_ps(bval1, wval1);
						bval2 = _mm512_div_ps(bval2, wval2);
						bval3 = _mm512_div_ps(bval3, wval3);
						bval4 = _mm512_div_ps(bval4, wval4);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(bval1), _mm512_cvtps_epi32(bval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(bval3), _mm512_cvtps_epi32(bval4)));
						gval1 = _mm512_div_ps(gval1, wval1);
						gval2 = _mm512_div_ps(gval2, wval2);
						gval3 = _mm512_div_ps(gval3, wval3);
						gval4 = _mm512_div_ps(gval4, wval4);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(gval1), _mm512_cvtps_epi32(gval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(gval3), _mm512_cvtps_epi32(gval4)));
						rval1 = _mm512_div_ps(rval1, wval1);
						rval2 = _mm512_div_ps(rval2, wval2);
						rval3 = _mm512_div_ps(rval3, wval3);
						rval4 = _mm512_div_ps(rval4, wval4);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(rval1), _mm512_cvtps_epi32(rval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(rval3), _mm512_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm512_stream_epi8_color(dptrc, a, b, c);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr + j;
					const uchar* sptrgj = sptrg + j;
					const uchar* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)abs(b - b0)] * range_weight[(int)abs(g - g0)] * range_weight[(int)abs(r - r0)];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
};


/*************************************************
	using LUT with "gather instruction" x 3
**************************************************/
class BilateralFilter_LUT_Gatherx3_64f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_LUT_Gatherx3_64f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double *_space_weight, const double* _range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		static const long long CV_DECL_ALIGNED(64) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff,
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff
		};
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			const double* sptr = temp->ptr<double>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrj = sptr + j;
						const __m512d sval0 = _mm512_load_pd(sptrj);

						__m512d tval = _mm512_setzero_pd();
						__m512d wval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d sref = _mm512_loadu_pd((sptrj + *ofs));
							const __m512i diff = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(sval0, sref), *(const __m512d*)v64f_absmask));
							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d _cw = _mm512_i64gather_pd(diff, range_weight, 8);
							__m512d _w = _mm512_mul_pd(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_pd(sref, _w, tval);
#else
							sref = _mm512_mul_pd(sref, _w);
							tval = _mm512_add_pd(tval, sref);
#endif

							wval = _mm512_add_pd(wval, _w);
						}
						tval = _mm512_div_pd(tval, wval);
						_mm512_stream_pd((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double val0 = sptr[j];
					double sum = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];
						const double w = space_weight[k] * range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			double* sptrb = (double*)temp->ptr(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			double* sptrg = (double*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			double* sptrr = (double*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m512d bval0 = _mm512_load_pd(sptrbj);
						const __m512d gval0 = _mm512_load_pd(sptrgj);
						const __m512d rval0 = _mm512_load_pd(sptrrj);

						__m512d wval = _mm512_setzero_pd();
						__m512d bval = _mm512_setzero_pd();
						__m512d gval = _mm512_setzero_pd();
						__m512d rval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d bref = _mm512_loadu_pd(sptrbj + *ofs);
							__m512d gref = _mm512_loadu_pd(sptrgj + *ofs);
							__m512d rref = _mm512_loadu_pd(sptrrj + *ofs);
							__m512i bdiff = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(bval0, bref), *(const __m512d*)v64f_absmask));
							__m512i gdiff = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(gval0, gref), *(const __m512d*)v64f_absmask));
							__m512i rdiff = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(rval0, rref), *(const __m512d*)v64f_absmask));

							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d _w = _mm512_mul_pd(_sw, _mm512_i64gather_pd(bdiff, range_weight, 8));
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
							_w = _mm512_mul_pd(_w, _mm512_i64gather_pd(gdiff, range_weight, 8));
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
							_w = _mm512_mul_pd(_w, _mm512_i64gather_pd(rdiff, range_weight, 8));
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_pd(bref, _w, bval);
							gval = _mm512_fmadd_pd(gref, _w, gval);
							rval = _mm512_fmadd_pd(rref, _w, rval);
#else
							bref = _mm512_mul_pd(bref, _w);
							bval = _mm512_add_pd(bval, bref);
							gref = _mm512_mul_pd(gref, _w);
							gval = _mm512_add_pd(gval, gref);
							rref = _mm512_mul_pd(rref, _w);
							rval = _mm512_add_pd(rval, rref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						bval = _mm512_div_pd(bval, wval);
						gval = _mm512_div_pd(gval, wval);
						rval = _mm512_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm512_stream_pd_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double* sptrrj = sptrr + j;
					const double* sptrgj = sptrg + j;
					const double* sptrbj = sptrb + j;

					const double r0 = sptrrj[0];
					const double g0 = sptrgj[0];
					const double b0 = sptrbj[0];

					double sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const double w = space_weight[k] * range_weight[(int)abs(b - b0)] * range_weight[(int)abs(g - g0)] * range_weight[(int)abs(r - r0)];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}

private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *space_weight, *range_weight;
};

class BilateralFilter_LUT_Gatherx3_32f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_LUT_Gatherx3_32f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrj = sptr + j;
						const __m512 sval0 = _mm512_load_ps(sptrj);

						__m512 tval = _mm512_setzero_ps();
						__m512 wval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 sref = _mm512_loadu_ps((sptrj + *ofs));
							const __m512i diff = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(sval0, sref), *(const __m512*)v32f_absmask));
							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _cw = _mm512_i32gather_ps(diff, range_weight, 4);
							__m512 _w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_ps(sref, _w, tval);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval = _mm512_add_ps(tval, sref);
#endif

							wval = _mm512_add_ps(wval, _w);
						}
						tval = _mm512_div_ps(tval, wval);
						_mm512_stream_ps((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m512 bval0 = _mm512_load_ps(sptrbj);
						const __m512 gval0 = _mm512_load_ps(sptrgj);
						const __m512 rval0 = _mm512_load_ps(sptrrj);

						__m512 wval = _mm512_setzero_ps();
						__m512 bval = _mm512_setzero_ps();
						__m512 gval = _mm512_setzero_ps();
						__m512 rval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 bref = _mm512_load_ps(sptrbj + *ofs);
							__m512 gref = _mm512_load_ps(sptrgj + *ofs);
							__m512 rref = _mm512_load_ps(sptrrj + *ofs);
							__m512i bdiff = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(bval0, bref), *(const __m512*)v32f_absmask));
							__m512i gdiff = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(gval0, gref), *(const __m512*)v32f_absmask));
							__m512i rdiff = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(rval0, rref), *(const __m512*)v32f_absmask));

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(bdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(gdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(rdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_ps(bref, _w, bval);
							gval = _mm512_fmadd_ps(gref, _w, gval);
							rval = _mm512_fmadd_ps(rref, _w, rval);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval = _mm512_add_ps(bval, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval = _mm512_add_ps(gval, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval = _mm512_add_ps(rval, rref);
#endif

							wval = _mm512_add_ps(wval, _w);
						}
						bval = _mm512_div_ps(bval, wval);
						gval = _mm512_div_ps(gval, wval);
						rval = _mm512_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm512_stream_ps_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float* sptrrj = sptrr + j;
					const float* sptrgj = sptrg + j;
					const float* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)abs(b - b0)] * range_weight[(int)abs(g - g0)] * range_weight[(int)abs(r - r0)];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}

private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
};

class BilateralFilter_LUT_Gatherx3_8u_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_LUT_Gatherx3_8u_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F) && checkHardwareSupport(CV_CPU_AVX_512BW);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 64 * (radiusH / 64 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrj = sptr + j;
						const __m512i sval0_64elem = _mm512_load_si512((__m512i*)sptrj);

						__m512 tval1 = _mm512_setzero_ps();
						__m512 wval1 = _mm512_setzero_ps();
						__m512 tval2 = _mm512_setzero_ps();
						__m512 wval2 = _mm512_setzero_ps();
						__m512 tval3 = _mm512_setzero_ps();
						__m512 wval3 = _mm512_setzero_ps();
						__m512 tval4 = _mm512_setzero_ps();
						__m512 wval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i sref_64elem = _mm512_loadu_si512((__m512i*)(sptrj + *ofs));
							const __m512i diff_64elem = _mm512_add_epi8(_mm512_subs_epu8(sval0_64elem, sref_64elem), _mm512_subs_epu8(sref_64elem, sval0_64elem));

							// lower bit
							__m512i sref_32elem = _mm512_unpacklo_epi8(sref_64elem, zero);
							__m512i diff_32elem = _mm512_unpacklo_epi8(diff_64elem, zero);

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512i diff = _mm512_unpacklo_epi16(diff_32elem, zero);
							__m512 _rw = _mm512_i32gather_ps(diff, range_weight, 4);
							__m512 _w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							__m512 sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm512_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval1 = _mm512_add_ps(tval1, sref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							diff = _mm512_unpackhi_epi16(diff_32elem, zero);
							_rw = _mm512_i32gather_ps(diff, range_weight, 4);
							_w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm512_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval2 = _mm512_add_ps(tval2, sref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// higher bit
							sref_32elem = _mm512_unpackhi_epi8(sref_64elem, zero);
							diff_32elem = _mm512_unpackhi_epi8(diff_64elem, zero);

							diff = _mm512_unpacklo_epi16(diff_32elem, zero);
							_rw = _mm512_i32gather_ps(diff, range_weight, 4);
							_w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm512_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval3 = _mm512_add_ps(tval3, sref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							diff = _mm512_unpackhi_epi16(diff_32elem, zero);
							_rw = _mm512_i32gather_ps(diff, range_weight, 4);
							_w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm512_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval4 = _mm512_add_ps(tval4, sref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						tval1 = _mm512_div_ps(tval1, wval1);
						tval2 = _mm512_div_ps(tval2, wval2);
						tval3 = _mm512_div_ps(tval3, wval3);
						tval4 = _mm512_div_ps(tval4, wval4);
						_mm512_stream_si512((__m512i*)(dptr + j), _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(tval1), _mm512_cvtps_epi32(tval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(tval3), _mm512_cvtps_epi32(tval4))));
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)(sum / wsum);
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 64 * (radiusH / 64 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 64 * (radiusH / 64 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 64 * (radiusH / 64 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m512i bval0_64elem = _mm512_load_si512((__m512i*)(sptrbj));
						const __m512i gval0_64elem = _mm512_load_si512((__m512i*)(sptrgj));
						const __m512i rval0_64elem = _mm512_load_si512((__m512i*)(sptrrj));

						__m512 wval1 = _mm512_setzero_ps();
						__m512 bval1 = _mm512_setzero_ps();
						__m512 gval1 = _mm512_setzero_ps();
						__m512 rval1 = _mm512_setzero_ps();

						__m512 wval2 = _mm512_setzero_ps();
						__m512 bval2 = _mm512_setzero_ps();
						__m512 gval2 = _mm512_setzero_ps();
						__m512 rval2 = _mm512_setzero_ps();

						__m512 wval3 = _mm512_setzero_ps();
						__m512 bval3 = _mm512_setzero_ps();
						__m512 gval3 = _mm512_setzero_ps();
						__m512 rval3 = _mm512_setzero_ps();

						__m512 wval4 = _mm512_setzero_ps();
						__m512 bval4 = _mm512_setzero_ps();
						__m512 gval4 = _mm512_setzero_ps();
						__m512 rval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i bref_64elem = _mm512_loadu_si512((__m512i*)(sptrbj + *ofs));
							const __m512i gref_64elem = _mm512_loadu_si512((__m512i*)(sptrgj + *ofs));
							const __m512i rref_64elem = _mm512_loadu_si512((__m512i*)(sptrrj + *ofs));

							const __m512i bdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(bval0_64elem, bref_64elem), _mm512_subs_epu8(bref_64elem, bval0_64elem));
							const __m512i gdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(gval0_64elem, gref_64elem), _mm512_subs_epu8(gref_64elem, gval0_64elem));
							const __m512i rdiff_64elem = _mm512_add_epi8(_mm512_subs_epu8(rval0_64elem, rref_64elem), _mm512_subs_epu8(rref_64elem, rval0_64elem));

							// lower bit
							__m512i bref_32elem = _mm512_unpacklo_epi8(bref_64elem, zero);
							__m512i gref_32elem = _mm512_unpacklo_epi8(gref_64elem, zero);
							__m512i rref_32elem = _mm512_unpacklo_epi8(rref_64elem, zero);
							__m512i bdiff_32elem = _mm512_unpacklo_epi8(bdiff_64elem, zero);
							__m512i gdiff_32elem = _mm512_unpacklo_epi8(gdiff_64elem, zero);
							__m512i rdiff_32elem = _mm512_unpacklo_epi8(rdiff_64elem, zero);

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512i bdiff = _mm512_unpacklo_epi16(bdiff_32elem, zero);
							__m512i gdiff = _mm512_unpacklo_epi16(gdiff_32elem, zero);
							__m512i rdiff = _mm512_unpacklo_epi16(rdiff_32elem, zero);
							__m512 _w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(bdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(gdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(rdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							__m512 bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							__m512 gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							__m512 rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm512_fmadd_ps(bref, _w, bval1);
							gval1 = _mm512_fmadd_ps(gref, _w, gval1);
							rval1 = _mm512_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval1 = _mm512_add_ps(bval1, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval1 = _mm512_add_ps(gval1, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval1 = _mm512_add_ps(rval1, rref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							bdiff = _mm512_unpackhi_epi16(bdiff_32elem, zero);
							gdiff = _mm512_unpackhi_epi16(gdiff_32elem, zero);
							rdiff = _mm512_unpackhi_epi16(rdiff_32elem, zero);

							_w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(bdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(gdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(rdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm512_fmadd_ps(bref, _w, bval2);
							gval2 = _mm512_fmadd_ps(gref, _w, gval2);
							rval2 = _mm512_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval2 = _mm512_add_ps(bval2, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval2 = _mm512_add_ps(gval2, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval2 = _mm512_add_ps(rval2, rref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// higher bit
							bref_32elem = _mm512_unpackhi_epi8(bref_64elem, zero);
							gref_32elem = _mm512_unpackhi_epi8(gref_64elem, zero);
							rref_32elem = _mm512_unpackhi_epi8(rref_64elem, zero);
							bdiff_32elem = _mm512_unpackhi_epi8(bdiff_64elem, zero);
							gdiff_32elem = _mm512_unpackhi_epi8(gdiff_64elem, zero);
							rdiff_32elem = _mm512_unpackhi_epi8(rdiff_64elem, zero);

							bdiff = _mm512_unpacklo_epi16(bdiff_32elem, zero);
							gdiff = _mm512_unpacklo_epi16(gdiff_32elem, zero);
							rdiff = _mm512_unpacklo_epi16(rdiff_32elem, zero);
							_w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(bdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(gdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(rdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm512_fmadd_ps(bref, _w, bval3);
							gval3 = _mm512_fmadd_ps(gref, _w, gval3);
							rval3 = _mm512_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval3 = _mm512_add_ps(bval3, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval3 = _mm512_add_ps(gval3, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval3 = _mm512_add_ps(rval3, rref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							bdiff = _mm512_unpackhi_epi16(bdiff_32elem, zero);
							gdiff = _mm512_unpackhi_epi16(gdiff_32elem, zero);
							rdiff = _mm512_unpackhi_epi16(rdiff_32elem, zero);
							_w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(bdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(gdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							_w = _mm512_mul_ps(_w, _mm512_i32gather_ps(rdiff, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm512_fmadd_ps(bref, _w, bval4);
							gval4 = _mm512_fmadd_ps(gref, _w, gval4);
							rval4 = _mm512_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval4 = _mm512_add_ps(bval4, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval4 = _mm512_add_ps(gval4, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval4 = _mm512_add_ps(rval4, rref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						bval1 = _mm512_div_ps(bval1, wval1);
						bval2 = _mm512_div_ps(bval2, wval2);
						bval3 = _mm512_div_ps(bval3, wval3);
						bval4 = _mm512_div_ps(bval4, wval4);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(bval1), _mm512_cvtps_epi32(bval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(bval3), _mm512_cvtps_epi32(bval4)));
						gval1 = _mm512_div_ps(gval1, wval1);
						gval2 = _mm512_div_ps(gval2, wval2);
						gval3 = _mm512_div_ps(gval3, wval3);
						gval4 = _mm512_div_ps(gval4, wval4);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(gval1), _mm512_cvtps_epi32(gval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(gval3), _mm512_cvtps_epi32(gval4)));
						rval1 = _mm512_div_ps(rval1, wval1);
						rval2 = _mm512_div_ps(rval2, wval2);
						rval3 = _mm512_div_ps(rval3, wval3);
						rval4 = _mm512_div_ps(rval4, wval4);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(rval1), _mm512_cvtps_epi32(rval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(rval3), _mm512_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm512_stream_epi8_color(dptrc, a, b, c);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr + j;
					const uchar* sptrgj = sptrg + j;
					const uchar* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)abs(b - b0)] * range_weight[(int)abs(g - g0)] + range_weight[(int)abs(r - r0)];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
};


/*************************************************
	using quantized range LUT with "set instruction" x 1
**************************************************/
class BilateralFilter_QuantizationRangeLUT_Setx1_64f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationRangeLUT_Setx1_64f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double* _space_weight, const double* _range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		static const long long CV_DECL_ALIGNED(64) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff,
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff
		};
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			long long CV_DECL_ALIGNED(64) buf[8];

			const double* sptr = temp->ptr<double>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrj = sptr + j;
						const __m512d sval0 = _mm512_load_pd(sptrj);

						__m512d tval = _mm512_setzero_pd();
						__m512d wval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d sref = _mm512_loadu_pd((sptrj + *ofs));
							_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(sval0, sref), *(const __m512d*)v64f_absmask)));

							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d _w = _mm512_mul_pd(
								_sw,
								_mm512_set_pd(
									range_weight[buf[7]],
									range_weight[buf[6]],
									range_weight[buf[5]],
									range_weight[buf[4]],
									range_weight[buf[3]],
									range_weight[buf[2]],
									range_weight[buf[1]],
									range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_pd(sref, _w, tval);
#else
							sref = _mm512_mul_pd(sref, _w);
							tval = _mm512_add_pd(tval, sref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						tval = _mm512_div_pd(tval, wval);
						_mm512_stream_pd((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double val0 = sptr[j];
					double sum = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];
						const double w = space_weight[k] * range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)sum / wsum;
				}
			}
		}
		else
		{
			long long CV_DECL_ALIGNED(64) buf[8];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m512d bval0 = _mm512_load_pd(sptrbj);
						const __m512d gval0 = _mm512_load_pd(sptrgj);
						const __m512d rval0 = _mm512_load_pd(sptrrj);

						__m512d wval = _mm512_setzero_pd();
						__m512d bval = _mm512_setzero_pd();
						__m512d gval = _mm512_setzero_pd();
						__m512d rval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d bref = _mm512_loadu_pd(sptrbj + *ofs);
							__m512d gref = _mm512_loadu_pd(sptrgj + *ofs);
							__m512d rref = _mm512_loadu_pd(sptrrj + *ofs);

							__m512d bdiff = _mm512_sub_pd(bval0, bref);
							__m512d gdiff = _mm512_sub_pd(gval0, gref);
							__m512d rdiff = _mm512_sub_pd(rval0, rref);
							__m512d difft = _mm512_mul_pd(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm512_fmadd_pd(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_pd(gdiff, gdiff);
							difft = _mm512_add_pd(difft, gdiff);
							rdiff = _mm512_mul_pd(rdiff, rdiff);
							difft = _mm512_add_pd(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(difft));
							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d _w = _mm512_mul_pd(
								_sw,
								_mm512_set_pd(
									range_weight[buf[7]],
									range_weight[buf[6]],
									range_weight[buf[5]],
									range_weight[buf[4]],
									range_weight[buf[3]],
									range_weight[buf[2]],
									range_weight[buf[1]],
									range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_pd(bref, _w, bval);
							gval = _mm512_fmadd_pd(gref, _w, gval);
							rval = _mm512_fmadd_pd(rref, _w, rval);
#else
							bref = _mm512_mul_pd(bref, _w);
							bval = _mm512_add_pd(bval, bref);
							gref = _mm512_mul_pd(gref, _w);
							gval = _mm512_add_pd(gval, gref);
							rref = _mm512_mul_pd(rref, _w);
							rval = _mm512_add_pd(rval, rref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						bval = _mm512_div_pd(bval, wval);
						gval = _mm512_div_pd(gval, wval);
						rval = _mm512_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm512_stream_pd_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double* sptrrj = sptrr + j;
					const double* sptrgj = sptrg + j;
					const double* sptrbj = sptrb + j;

					const double r0 = sptrrj[0];
					const double g0 = sptrgj[0];
					const double b0 = sptrbj[0];

					double sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const double w = space_weight[k] * range_weight[(int)sqrt((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *space_weight, *range_weight;
};

class BilateralFilter_QuantizationRangeLUT_Setx1_32f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationRangeLUT_Setx1_32f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(64) buf[16];

			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrj = sptr + j;
						const __m512 sval0 = _mm512_load_ps(sptrj);

						__m512 tval = _mm512_setzero_ps();
						__m512 wval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 sref = _mm512_loadu_ps((sptrj + *ofs));
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(sval0, sref), *(const __m512*)v32f_absmask)));

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[15]],
									range_weight[buf[14]],
									range_weight[buf[13]],
									range_weight[buf[12]],
									range_weight[buf[11]],
									range_weight[buf[10]],
									range_weight[buf[9]],
									range_weight[buf[8]],
									range_weight[buf[7]],
									range_weight[buf[6]],
									range_weight[buf[5]],
									range_weight[buf[4]],
									range_weight[buf[3]],
									range_weight[buf[2]],
									range_weight[buf[1]],
									range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_ps(sref, _w, tval);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval = _mm512_add_ps(tval, sref);
#endif

							wval = _mm512_add_ps(wval, _w);
						}
						tval = _mm512_div_ps(tval, wval);
						_mm512_stream_ps((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(64) buf[16];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m512 bval0 = _mm512_load_ps(sptrbj);
						const __m512 gval0 = _mm512_load_ps(sptrgj);
						const __m512 rval0 = _mm512_load_ps(sptrrj);

						__m512 wval = _mm512_setzero_ps();
						__m512 bval = _mm512_setzero_ps();
						__m512 gval = _mm512_setzero_ps();
						__m512 rval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 bref = _mm512_load_ps(sptrbj + *ofs);
							__m512 gref = _mm512_load_ps(sptrgj + *ofs);
							__m512 rref = _mm512_load_ps(sptrrj + *ofs);

							__m512 bdiff = _mm512_sub_ps(bval0, bref);
							__m512 gdiff = _mm512_sub_ps(gval0, gref);
							__m512 rdiff = _mm512_sub_ps(rval0, rref);
							__m512 difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[15]],
									range_weight[buf[14]],
									range_weight[buf[13]],
									range_weight[buf[12]],
									range_weight[buf[11]],
									range_weight[buf[10]],
									range_weight[buf[9]],
									range_weight[buf[8]],
									range_weight[buf[7]],
									range_weight[buf[6]],
									range_weight[buf[5]],
									range_weight[buf[4]],
									range_weight[buf[3]],
									range_weight[buf[2]],
									range_weight[buf[1]],
									range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_ps(bref, _w, bval);
							gval = _mm512_fmadd_ps(gref, _w, gval);
							rval = _mm512_fmadd_ps(rref, _w, rval);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval = _mm512_add_ps(bval, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval = _mm512_add_ps(gval, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval = _mm512_add_ps(rval, rref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						bval = _mm512_div_ps(bval, wval);
						gval = _mm512_div_ps(gval, wval);
						rval = _mm512_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm512_stream_ps_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float* sptrrj = sptrr + j;
					const float* sptrgj = sptrg + j;
					const float* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)sqrt((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
};

class BilateralFilter_QuantizationRangeLUT_Setx1_8u_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationRangeLUT_Setx1_8u_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const  float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F) && checkHardwareSupport(CV_CPU_AVX_512BW);
#endif
		if (cn == 1)
		{
			uchar CV_DECL_ALIGNED(64) buf[64];

			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 64 * (radiusH / 64 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrj = sptr + j;
						const __m512i sval0_64elem = _mm512_load_si512((__m512i*)sptrj);

						__m512 tval1 = _mm512_setzero_ps();
						__m512 wval1 = _mm512_setzero_ps();
						__m512 tval2 = _mm512_setzero_ps();
						__m512 wval2 = _mm512_setzero_ps();
						__m512 tval3 = _mm512_setzero_ps();
						__m512 wval3 = _mm512_setzero_ps();
						__m512 tval4 = _mm512_setzero_ps();
						__m512 wval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i sref_64elem = _mm512_loadu_si512((__m512i*)(sptrj + *ofs));
							_mm512_store_si512((__m512i*)buf, _mm512_add_epi8(_mm512_subs_epu8(sval0_64elem, sref_64elem), _mm512_subs_epu8(sref_64elem, sval0_64elem)));

							// lower bit
							__m512i sref_32elem = _mm512_unpacklo_epi8(sref_64elem, zero);
							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[51]], range_weight[buf[50]], range_weight[buf[49]], range_weight[buf[48]],
									range_weight[buf[35]], range_weight[buf[34]], range_weight[buf[33]], range_weight[buf[32]],
									range_weight[buf[19]], range_weight[buf[18]], range_weight[buf[17]], range_weight[buf[16]],
									range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							__m512 sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm512_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval1 = _mm512_add_ps(tval1, sref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[55]], range_weight[buf[54]], range_weight[buf[53]], range_weight[buf[52]],
									range_weight[buf[39]], range_weight[buf[38]], range_weight[buf[37]], range_weight[buf[36]],
									range_weight[buf[23]], range_weight[buf[22]], range_weight[buf[21]], range_weight[buf[20]],
									range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm512_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval2 = _mm512_add_ps(tval2, sref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// higher bit
							sref_32elem = _mm512_unpackhi_epi8(sref_64elem, zero);
							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[59]], range_weight[buf[58]], range_weight[buf[57]], range_weight[buf[56]],
									range_weight[buf[43]], range_weight[buf[42]], range_weight[buf[41]], range_weight[buf[40]],
									range_weight[buf[27]], range_weight[buf[26]], range_weight[buf[25]], range_weight[buf[24]],
									range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm512_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval3 = _mm512_add_ps(tval3, sref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[63]], range_weight[buf[62]], range_weight[buf[61]], range_weight[buf[60]],
									range_weight[buf[47]], range_weight[buf[46]], range_weight[buf[45]], range_weight[buf[44]],
									range_weight[buf[31]], range_weight[buf[30]], range_weight[buf[29]], range_weight[buf[28]],
									range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm512_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval4 = _mm512_add_ps(tval4, sref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						tval1 = _mm512_div_ps(tval1, wval1);
						tval2 = _mm512_div_ps(tval2, wval2);
						tval3 = _mm512_div_ps(tval3, wval3);
						tval4 = _mm512_div_ps(tval4, wval4);
						_mm512_stream_si512((__m512i*)(dptr + j), _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(tval1), _mm512_cvtps_epi32(tval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(tval3), _mm512_cvtps_epi32(tval4))));
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[int(abs(val - val0))];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)(sum / wsum);
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(64) buf[16];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 64 * (radiusH / 64 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 64 * (radiusH / 64 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 64 * (radiusH / 64 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m512i bval0_32elem = _mm512_load_si512((__m512i*)(sptrbj));
						const __m512i gval0_32elem = _mm512_load_si512((__m512i*)(sptrgj));
						const __m512i rval0_32elem = _mm512_load_si512((__m512i*)(sptrrj));

						__m512i temp = _mm512_unpacklo_epi8(bval0_32elem, zero);
						const __m512 bval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 bval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(bval0_32elem, zero);
						const __m512 bval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 bval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpacklo_epi8(gval0_32elem, zero);
						const __m512 gval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 gval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(gval0_32elem, zero);
						const __m512 gval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 gval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpacklo_epi8(rval0_32elem, zero);
						const __m512 rval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 rval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(rval0_32elem, zero);
						const __m512 rval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 rval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));

						__m512 wval1 = _mm512_setzero_ps();
						__m512 bval1 = _mm512_setzero_ps();
						__m512 gval1 = _mm512_setzero_ps();
						__m512 rval1 = _mm512_setzero_ps();

						__m512 wval2 = _mm512_setzero_ps();
						__m512 bval2 = _mm512_setzero_ps();
						__m512 gval2 = _mm512_setzero_ps();
						__m512 rval2 = _mm512_setzero_ps();

						__m512 wval3 = _mm512_setzero_ps();
						__m512 bval3 = _mm512_setzero_ps();
						__m512 gval3 = _mm512_setzero_ps();
						__m512 rval3 = _mm512_setzero_ps();

						__m512 wval4 = _mm512_setzero_ps();
						__m512 bval4 = _mm512_setzero_ps();
						__m512 gval4 = _mm512_setzero_ps();
						__m512 rval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i bref_64elem = _mm512_loadu_si512((__m512i*)(sptrbj + *ofs));
							const __m512i gref_64elem = _mm512_loadu_si512((__m512i*)(sptrgj + *ofs));
							const __m512i rref_64elem = _mm512_loadu_si512((__m512i*)(sptrrj + *ofs));

							// low
							__m512i bref_32elem = _mm512_unpacklo_epi8(bref_64elem, zero);
							__m512i gref_32elem = _mm512_unpacklo_epi8(gref_64elem, zero);
							__m512i rref_32elem = _mm512_unpacklo_epi8(rref_64elem, zero);

							__m512 bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							__m512 gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							__m512 rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));

							__m512 bdiff = _mm512_sub_ps(bval0_1, bref);
							__m512 gdiff = _mm512_sub_ps(gval0_1, gref);
							__m512 rdiff = _mm512_sub_ps(rval0_1, rref);
							__m512 difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							const __m512 _sw = _mm512_set1_ps(*spw);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							__m512 _w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
									range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval1 = _mm512_fmadd_ps(bref, _w, bval1);
							gval1 = _mm512_fmadd_ps(gref, _w, gval1);
							rval1 = _mm512_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval1 = _mm512_add_ps(bval1, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval1 = _mm512_add_ps(gval1, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval1 = _mm512_add_ps(rval1, rref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_2, bref);
							gdiff = _mm512_sub_ps(gval0_2, gref);
							rdiff = _mm512_sub_ps(rval0_2, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
									range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm512_fmadd_ps(bref, _w, bval2);
							gval2 = _mm512_fmadd_ps(gref, _w, gval2);
							rval2 = _mm512_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval2 = _mm512_add_ps(bval2, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval2 = _mm512_add_ps(gval2, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval2 = _mm512_add_ps(rval2, rref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// high
							bref_32elem = _mm512_unpackhi_epi8(bref_64elem, zero);
							gref_32elem = _mm512_unpackhi_epi8(gref_64elem, zero);
							rref_32elem = _mm512_unpackhi_epi8(rref_64elem, zero);

							bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_3, bref);
							gdiff = _mm512_sub_ps(gval0_3, gref);
							rdiff = _mm512_sub_ps(rval0_3, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif

							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
									range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm512_fmadd_ps(bref, _w, bval3);
							gval3 = _mm512_fmadd_ps(gref, _w, gval3);
							rval3 = _mm512_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval3 = _mm512_add_ps(bval3, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval3 = _mm512_add_ps(gval3, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval3 = _mm512_add_ps(rval3, rref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_4, bref);
							gdiff = _mm512_sub_ps(gval0_4, gref);
							rdiff = _mm512_sub_ps(rval0_4, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							_w = _mm512_mul_ps(
								_sw,
								_mm512_set_ps(
									range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
									range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
								)
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm512_fmadd_ps(bref, _w, bval4);
							gval4 = _mm512_fmadd_ps(gref, _w, gval4);
							rval4 = _mm512_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval4 = _mm512_add_ps(bval4, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval4 = _mm512_add_ps(gval4, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval4 = _mm512_add_ps(rval4, rref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						bval1 = _mm512_div_ps(bval1, wval1);
						bval2 = _mm512_div_ps(bval2, wval2);
						bval3 = _mm512_div_ps(bval3, wval3);
						bval4 = _mm512_div_ps(bval4, wval4);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(bval1), _mm512_cvtps_epi32(bval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(bval3), _mm512_cvtps_epi32(bval4)));
						gval1 = _mm512_div_ps(gval1, wval1);
						gval2 = _mm512_div_ps(gval2, wval2);
						gval3 = _mm512_div_ps(gval3, wval3);
						gval4 = _mm512_div_ps(gval4, wval4);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(gval1), _mm512_cvtps_epi32(gval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(gval3), _mm512_cvtps_epi32(gval4)));
						rval1 = _mm512_div_ps(rval1, wval1);
						rval2 = _mm512_div_ps(rval2, wval2);
						rval3 = _mm512_div_ps(rval3, wval3);
						rval4 = _mm512_div_ps(rval4, wval4);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(rval1), _mm512_cvtps_epi32(rval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(rval3), _mm512_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm512_stream_epi8_color(dptrc, a, b, c);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr + j;
					const uchar* sptrgj = sptrg + j;
					const uchar* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)sqrt((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
};


/*************************************************
	using quantized range LUT with "gather instruction" x 1
*************************************************/
class BilateralFilter_QuantizationRangeLUT_Gatherx1_64f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationRangeLUT_Gatherx1_64f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double *_space_weight, const double *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		static const long long CV_DECL_ALIGNED(64) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff,
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff
		};
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			const double* sptr = temp->ptr<double>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrj = sptr + j;
						const __m512d sval0 = _mm512_load_pd(sptrj);

						__m512d tval = _mm512_setzero_pd();
						__m512d wval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d sref = _mm512_loadu_pd((sptrj + *ofs));
							const __m512i diff = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(sval0, sref), *(const __m512d*)v64f_absmask));

							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d _rw = _mm512_i64gather_pd(diff, range_weight, 8);

							__m512d _w = _mm512_mul_pd(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_pd(sref, _w, tval);
#else
							sref = _mm512_mul_pd(sref, _w);
							tval = _mm512_add_pd(tval, sref);
#endif

							wval = _mm512_add_pd(wval, _w);
						}
						tval = _mm512_div_pd(tval, wval);
						_mm512_stream_pd((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double val0 = sptr[j];
					double sum = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];
						const double w = space_weight[k] * range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double *sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m512d bval0 = _mm512_load_pd(sptrbj);
						const __m512d gval0 = _mm512_load_pd(sptrgj);
						const __m512d rval0 = _mm512_load_pd(sptrrj);

						__m512d wval = _mm512_setzero_pd();
						__m512d bval = _mm512_setzero_pd();
						__m512d gval = _mm512_setzero_pd();
						__m512d rval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d bref = _mm512_loadu_pd(sptrbj + *ofs);
							__m512d gref = _mm512_loadu_pd(sptrgj + *ofs);
							__m512d rref = _mm512_loadu_pd(sptrrj + *ofs);

							__m512d bdiff = _mm512_sub_pd(bval0, bref);
							__m512d gdiff = _mm512_sub_pd(gval0, gref);
							__m512d rdiff = _mm512_sub_pd(rval0, rref);
							__m512d difft = _mm512_mul_pd(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm512_fmadd_pd(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_pd(gdiff, gdiff);
							difft = _mm512_add_pd(difft, gdiff);
							rdiff = _mm512_mul_pd(rdiff, rdiff);
							difft = _mm512_add_pd(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, difft);
							__m512i mmidx = _mm512_cvtpd_epi64(difft);
							const __m512d _sw = _mm512_set1_pd(*spw);
							__m512d _w = _mm512_mul_pd(_sw, _mm512_i64gather_pd(mmidx, range_weight, 8));
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_pd(bref, _w, bval);
							gval = _mm512_fmadd_pd(gref, _w, gval);
							rval = _mm512_fmadd_pd(rref, _w, rval);
#else
							bref = _mm512_mul_pd(bref, _w);
							bval = _mm512_add_pd(bval, bref);
							gref = _mm512_mul_pd(gref, _w);
							gval = _mm512_add_pd(gval, gref);
							rref = _mm512_mul_pd(rref, _w);
							rval = _mm512_add_pd(rval, rref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						bval = _mm512_div_pd(bval, wval);
						gval = _mm512_div_pd(gval, wval);
						rval = _mm512_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm512_stream_pd_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double* sptrrj = sptrr + j;
					const double* sptrgj = sptrg + j;
					const double* sptrbj = sptrb + j;

					const double r0 = sptrrj[0];
					const double g0 = sptrgj[0];
					const double b0 = sptrbj[0];

					double sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const double w = space_weight[k] * range_weight[(int)sqrt((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}

private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *space_weight, *range_weight;
};

class BilateralFilter_QuantizationRangeLUT_Gatherx1_32f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationRangeLUT_Gatherx1_32f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrj = sptr + j;
						const __m512 sval0 = _mm512_load_ps(sptrj);

						__m512 tval = _mm512_setzero_ps();
						__m512 wval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 sref = _mm512_loadu_ps((sptrj + *ofs));
							const __m512i diff = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(sval0, sref), *(const __m512*)v32f_absmask));

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _rw = _mm512_i32gather_ps(diff, range_weight, 4);

							__m512 _w = _mm512_mul_ps(_sw, _rw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_ps(sref, _w, tval);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval = _mm512_add_ps(tval, sref);
#endif

							wval = _mm512_add_ps(wval, _w);
						}
						tval = _mm512_div_ps(tval, wval);
						_mm512_stream_ps((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if QUANTIZATION_SIZE_MODE == 1
					const __m512 mQpara = _mm512_set1_ps(QUANTIZATION_SIZE_PRAMETER);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m512 bval0 = _mm512_load_ps(sptrbj);
						const __m512 gval0 = _mm512_load_ps(sptrgj);
						const __m512 rval0 = _mm512_load_ps(sptrrj);

						__m512 wval = _mm512_setzero_ps();
						__m512 bval = _mm512_setzero_ps();
						__m512 gval = _mm512_setzero_ps();
						__m512 rval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 bref = _mm512_load_ps(sptrbj + *ofs);
							__m512 gref = _mm512_load_ps(sptrgj + *ofs);
							__m512 rref = _mm512_load_ps(sptrrj + *ofs);

							__m512 bdiff = _mm512_sub_ps(bval0, bref);
							__m512 gdiff = _mm512_sub_ps(gval0, gref);
							__m512 rdiff = _mm512_sub_ps(rval0, rref);
							__m512 difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif

#if QUANTIZATION_MODE == 0
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
#endif
#if QUANTIZATION_SIZE_MODE == 1
							difft = _mm512_mul_ps(difft, mQpara);
#endif

							__m512i mmidx = _mm512_cvtps_epi32(difft);
							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512 _w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(mmidx, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_ps(bref, _w, bval);
							gval = _mm512_fmadd_ps(gref, _w, gval);
							rval = _mm512_fmadd_ps(rref, _w, rval);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval = _mm512_add_ps(bval, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval = _mm512_add_ps(gval, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval = _mm512_add_ps(rval, rref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						bval = _mm512_div_ps(bval, wval);
						gval = _mm512_div_ps(gval, wval);
						rval = _mm512_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm512_stream_ps_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float* sptrrj = sptrr + j;
					const float* sptrgj = sptrg + j;
					const float* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)sqrt((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}

private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
};

class BilateralFilter_QuantizationRangeLUT_Gatherx1_8u_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationRangeLUT_Gatherx1_8u_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F) && checkHardwareSupport(CV_CPU_AVX_512BW);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 64 * (radiusH / 64 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrj = sptr + j;
						const __m512i sval0_64elem = _mm512_load_si512((__m512i*)sptrj);

						__m512 tval1 = _mm512_setzero_ps();
						__m512 wval1 = _mm512_setzero_ps();
						__m512 tval2 = _mm512_setzero_ps();
						__m512 wval2 = _mm512_setzero_ps();
						__m512 tval3 = _mm512_setzero_ps();
						__m512 wval3 = _mm512_setzero_ps();
						__m512 tval4 = _mm512_setzero_ps();
						__m512 wval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i sref_64elem = _mm512_loadu_si512((__m512i*)(sptrj + *ofs));
							const __m512i diff_64elem = _mm512_add_epi8(_mm512_subs_epu8(sval0_64elem, sref_64elem), _mm512_subs_epu8(sref_64elem, sval0_64elem));

							// lower bit
							__m512i sref_16elem = _mm512_unpacklo_epi8(sref_64elem, zero);
							__m512i diff_16elem = _mm512_unpacklo_epi8(diff_64elem, zero);

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512i diff = _mm512_unpacklo_epi16(diff_16elem, zero);
							__m512 _cw = _mm512_i32gather_ps(diff, range_weight, 4);
							__m512 _w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							__m512 sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm512_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval1 = _mm512_add_ps(tval1, sref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							diff = _mm512_unpackhi_epi16(diff_16elem, zero);
							_cw = _mm512_i32gather_ps(diff, range_weight, 4);
							_w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm512_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval2 = _mm512_add_ps(tval2, sref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);

							// higher bit
							sref_16elem = _mm512_unpackhi_epi8(sref_64elem, zero);
							diff_16elem = _mm512_unpackhi_epi8(diff_64elem, zero);

							diff = _mm512_unpacklo_epi16(diff_16elem, zero);
							_cw = _mm512_i32gather_ps(diff, range_weight, 4);
							_w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm512_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval3 = _mm512_add_ps(tval3, sref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							diff = _mm512_unpackhi_epi16(diff_16elem, zero);
							_cw = _mm512_i32gather_ps(diff, range_weight, 4);
							_w = _mm512_mul_ps(_sw, _cw);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm512_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval4 = _mm512_add_ps(tval4, sref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						tval1 = _mm512_div_ps(tval1, wval1);
						tval2 = _mm512_div_ps(tval2, wval2);
						tval3 = _mm512_div_ps(tval3, wval3);
						tval4 = _mm512_div_ps(tval4, wval4);
						_mm512_stream_si512((__m512i*)(dptr + j), _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(tval1), _mm512_cvtps_epi32(tval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(tval3), _mm512_cvtps_epi32(tval4))));
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[int(abs(val - val0))];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)(sum / wsum);
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 64 * (radiusH / 64 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 64 * (radiusH / 64 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 64 * (radiusH / 64 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m512i bval0_64elem = _mm512_load_si512((__m512i*)(sptrbj));
						const __m512i gval0_64elem = _mm512_load_si512((__m512i*)(sptrgj));
						const __m512i rval0_64elem = _mm512_load_si512((__m512i*)(sptrrj));

						__m512i temp = _mm512_unpacklo_epi8(bval0_64elem, zero);
						const __m512 bval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 bval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(bval0_64elem, zero);
						const __m512 bval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 bval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpacklo_epi8(gval0_64elem, zero);
						const __m512 gval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 gval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(gval0_64elem, zero);
						const __m512 gval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 gval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpacklo_epi8(rval0_64elem, zero);
						const __m512 rval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 rval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(rval0_64elem, zero);
						const __m512 rval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 rval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));

						__m512 wval1 = _mm512_setzero_ps();
						__m512 bval1 = _mm512_setzero_ps();
						__m512 gval1 = _mm512_setzero_ps();
						__m512 rval1 = _mm512_setzero_ps();

						__m512 wval2 = _mm512_setzero_ps();
						__m512 bval2 = _mm512_setzero_ps();
						__m512 gval2 = _mm512_setzero_ps();
						__m512 rval2 = _mm512_setzero_ps();

						__m512 wval3 = _mm512_setzero_ps();
						__m512 bval3 = _mm512_setzero_ps();
						__m512 gval3 = _mm512_setzero_ps();
						__m512 rval3 = _mm512_setzero_ps();

						__m512 wval4 = _mm512_setzero_ps();
						__m512 bval4 = _mm512_setzero_ps();
						__m512 gval4 = _mm512_setzero_ps();
						__m512 rval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i bref_64elem = _mm512_loadu_si512((__m512i*)(sptrbj + *ofs));
							const __m512i gref_64elem = _mm512_loadu_si512((__m512i*)(sptrgj + *ofs));
							const __m512i rref_64elem = _mm512_loadu_si512((__m512i*)(sptrrj + *ofs));

							// low
							__m512i bref_32elem = _mm512_unpacklo_epi8(bref_64elem, zero);
							__m512i gref_32elem = _mm512_unpacklo_epi8(gref_64elem, zero);
							__m512i rref_32elem = _mm512_unpacklo_epi8(rref_64elem, zero);

							__m512 bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							__m512 gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							__m512 rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));

							__m512 bdiff = _mm512_sub_ps(bval0_1, bref);
							__m512 gdiff = _mm512_sub_ps(gval0_1, gref);
							__m512 rdiff = _mm512_sub_ps(rval0_1, rref);
							__m512 difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);

							const __m512 _sw = _mm512_set1_ps(*spw);
							__m512i mmidx = _mm512_cvtps_epi32(difft);
							__m512 _w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(mmidx, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm512_fmadd_ps(bref, _w, bval1);
							gval1 = _mm512_fmadd_ps(gref, _w, gval1);
							rval1 = _mm512_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval1 = _mm512_add_ps(bval1, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval1 = _mm512_add_ps(gval1, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval1 = _mm512_add_ps(rval1, rref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_2, bref);
							gdiff = _mm512_sub_ps(gval0_2, gref);
							rdiff = _mm512_sub_ps(rval0_2, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							mmidx = _mm512_cvtps_epi32(difft);
							_w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(mmidx, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm512_fmadd_ps(bref, _w, bval2);
							gval2 = _mm512_fmadd_ps(gref, _w, gval2);
							rval2 = _mm512_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval2 = _mm512_add_ps(bval2, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval2 = _mm512_add_ps(gval2, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval2 = _mm512_add_ps(rval2, rref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// high
							bref_32elem = _mm512_unpackhi_epi8(bref_64elem, zero);
							gref_32elem = _mm512_unpackhi_epi8(gref_64elem, zero);
							rref_32elem = _mm512_unpackhi_epi8(rref_64elem, zero);

							bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_3, bref);
							gdiff = _mm512_sub_ps(gval0_3, gref);
							rdiff = _mm512_sub_ps(rval0_3, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);

							mmidx = _mm512_cvtps_epi32(difft);
							_w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(mmidx, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm512_fmadd_ps(bref, _w, bval3);
							gval3 = _mm512_fmadd_ps(gref, _w, gval3);
							rval3 = _mm512_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval3 = _mm512_add_ps(bval3, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval3 = _mm512_add_ps(gval3, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval3 = _mm512_add_ps(rval3, rref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_4, bref);
							gdiff = _mm512_sub_ps(gval0_4, gref);
							rdiff = _mm512_sub_ps(rval0_4, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							mmidx = _mm512_cvtps_epi32(difft);
							_w = _mm512_mul_ps(_sw, _mm512_i32gather_ps(mmidx, range_weight, 4));
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm512_fmadd_ps(bref, _w, bval4);
							gval4 = _mm512_fmadd_ps(gref, _w, gval4);
							rval4 = _mm512_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval4 = _mm512_add_ps(bval4, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval4 = _mm512_add_ps(gval4, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval4 = _mm512_add_ps(rval4, rref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						bval1 = _mm512_div_ps(bval1, wval1);
						bval2 = _mm512_div_ps(bval2, wval2);
						bval3 = _mm512_div_ps(bval3, wval3);
						bval4 = _mm512_div_ps(bval4, wval4);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(bval1), _mm512_cvtps_epi32(bval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(bval3), _mm512_cvtps_epi32(bval4)));
						gval1 = _mm512_div_ps(gval1, wval1);
						gval2 = _mm512_div_ps(gval2, wval2);
						gval3 = _mm512_div_ps(gval3, wval3);
						gval4 = _mm512_div_ps(gval4, wval4);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(gval1), _mm512_cvtps_epi32(gval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(gval3), _mm512_cvtps_epi32(gval4)));
						rval1 = _mm512_div_ps(rval1, wval1);
						rval2 = _mm512_div_ps(rval2, wval2);
						rval3 = _mm512_div_ps(rval3, wval3);
						rval4 = _mm512_div_ps(rval4, wval4);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(rval1), _mm512_cvtps_epi32(rval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(rval3), _mm512_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm512_stream_epi8_color(dptrc, a, b, c);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr + j;
					const uchar* sptrgj = sptrg + j;
					const uchar* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = space_weight[k] * range_weight[(int)sqrt((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
};


/*************************************************
	using quantized LUT with "set instruction" x 1
**************************************************/
class BilateralFilter_QuantizationLUT_Setx1_64f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationLUT_Setx1_64f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double _sigma_range, const double _sigma_space, const double *_space_weight, const double *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const double coeff = (sigma_range * sigma_range) / (sigma_space * sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			long long CV_DECL_ALIGNED(64) buf[8];

			const double* sptr = temp->ptr<double>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d mcoeff = _mm512_set1_pd(coeff);
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrj = sptr + j;
						const __m512d sval0 = _mm512_load_pd(sptrj);

						__m512d tval = _mm512_setzero_pd();
						__m512d wval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512d _sw = _mm512_mul_pd(_mm512_set1_pd(*spw), mcoeff);
							__m512d sref = _mm512_loadu_pd((sptrj + *ofs));
							const __m512d diff = _mm512_sub_pd(sval0, sref);
							const __m512d difft =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_pd(diff, diff, _sw);
#else
								_mm512_add_pd(_mm512_mul_pd(diff, diff), _sw);
#endif
							_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, difft))));
							__m512d _w = _mm512_set_pd(
								range_weight[buf[7]],
								range_weight[buf[6]],
								range_weight[buf[5]],
								range_weight[buf[4]],
								range_weight[buf[3]],
								range_weight[buf[2]],
								range_weight[buf[1]],
								range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_pd(sref, _w, tval);
#else
							sref = _mm512_mul_pd(sref, _w);
							tval = _mm512_add_pd(tval, sref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						tval = _mm512_div_pd(tval, wval);
						_mm512_stream_pd((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double val0 = sptr[j];
					double sum = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];
						const double w = range_weight[(int)sqrt(coeff * space_weight[k] + (val - val0) * (val - val0))];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)sum / wsum;
				}
			}
		}
		else
		{
			long long CV_DECL_ALIGNED(64) buf[8];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d mcoeff = _mm512_set1_pd(coeff);
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m512d bval0 = _mm512_load_pd(sptrbj);
						const __m512d gval0 = _mm512_load_pd(sptrgj);
						const __m512d rval0 = _mm512_load_pd(sptrrj);

						__m512d wval = _mm512_setzero_pd();
						__m512d bval = _mm512_setzero_pd();
						__m512d gval = _mm512_setzero_pd();
						__m512d rval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d bref = _mm512_load_pd(sptrbj + *ofs);
							__m512d gref = _mm512_load_pd(sptrgj + *ofs);
							__m512d rref = _mm512_load_pd(sptrrj + *ofs);

							__m512d bdiff = _mm512_sub_pd(bval0, bref);
							__m512d gdiff = _mm512_sub_pd(gval0, gref);
							__m512d rdiff = _mm512_sub_pd(rval0, rref);

							const __m512d _sw = _mm512_mul_pd(_mm512_set1_pd(*spw), mcoeff);
#if __USE_FMA_INSTRUCTION__
							__m512d difft = _mm512_fmadd_pd(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm512_fmadd_pd(rdiff, rdiff, difft);
#else
							__m512d difft = _mm512_add_pd(_mm512_mul_pd(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_pd(gdiff, gdiff);
							difft = _mm512_add_pd(difft, gdiff);
							rdiff = _mm512_mul_pd(rdiff, rdiff);
							difft = _mm512_add_pd(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(difft));
							__m512d _w = _mm512_set_pd(
								range_weight[buf[7]],
								range_weight[buf[6]],
								range_weight[buf[5]],
								range_weight[buf[4]],
								range_weight[buf[3]],
								range_weight[buf[2]],
								range_weight[buf[1]],
								range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_pd(bref, _w, bval);
							gval = _mm512_fmadd_pd(gref, _w, gval);
							rval = _mm512_fmadd_pd(rref, _w, rval);
#else
							bref = _mm512_mul_pd(bref, _w);
							bval = _mm512_add_pd(bval, bref);
							gref = _mm512_mul_pd(gref, _w);
							gval = _mm512_add_pd(gval, gref);
							rref = _mm512_mul_pd(rref, _w);
							rval = _mm512_add_pd(rval, rref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						bval = _mm512_div_pd(bval, wval);
						gval = _mm512_div_pd(gval, wval);
						rval = _mm512_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm512_stream_pd_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double* sptrrj = sptrr + j;
					const double* sptrgj = sptrg + j;
					const double* sptrbj = sptrb + j;

					const double r0 = sptrrj[0];
					const double g0 = sptrgj[0];
					const double b0 = sptrbj[0];

					double sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const double w = range_weight[(int)sqrt(coeff * space_weight[k] + (b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *space_weight, *range_weight;
	double sigma_range, sigma_space;
};

class BilateralFilter_QuantizationLUT_Setx1_32f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationLUT_Setx1_32f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float _sigma_range, const float _sigma_space, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float coeff = (sigma_range * sigma_range) / (sigma_space * sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(64) buf[16];

			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcoeff = _mm512_set1_ps(coeff);
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrj = sptr + j;
						const __m512 sval0 = _mm512_load_ps(sptrj);

						__m512 tval = _mm512_setzero_ps();
						__m512 wval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), mcoeff);
							__m512 sref = _mm512_loadu_ps((sptrj + *ofs));
							const __m512 diff = _mm512_sub_ps(sval0, sref);
							const __m512 difft =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(diff, diff, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
#endif
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft))));
							__m512 _w = _mm512_set_ps(
								range_weight[buf[15]],
								range_weight[buf[14]],
								range_weight[buf[13]],
								range_weight[buf[12]],
								range_weight[buf[11]],
								range_weight[buf[10]],
								range_weight[buf[9]],
								range_weight[buf[8]],
								range_weight[buf[7]],
								range_weight[buf[6]],
								range_weight[buf[5]],
								range_weight[buf[4]],
								range_weight[buf[3]],
								range_weight[buf[2]],
								range_weight[buf[1]],
								range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_ps(sref, _w, tval);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval = _mm512_add_ps(tval, sref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						tval = _mm512_div_ps(tval, wval);
						_mm512_stream_ps((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = range_weight[(int)sqrt(coeff * space_weight[k] + (val - val0) * (val - val0))];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(64) buf[16];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcoeff = _mm512_set1_ps(coeff);
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m512 bval0 = _mm512_load_ps(sptrbj);
						const __m512 gval0 = _mm512_load_ps(sptrgj);
						const __m512 rval0 = _mm512_load_ps(sptrrj);

						__m512 wval = _mm512_setzero_ps();
						__m512 bval = _mm512_setzero_ps();
						__m512 gval = _mm512_setzero_ps();
						__m512 rval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 bref = _mm512_load_ps(sptrbj + *ofs);
							__m512 gref = _mm512_load_ps(sptrgj + *ofs);
							__m512 rref = _mm512_load_ps(sptrrj + *ofs);

							__m512 bdiff = _mm512_sub_ps(bval0, bref);
							__m512 gdiff = _mm512_sub_ps(gval0, gref);
							__m512 rdiff = _mm512_sub_ps(rval0, rref);

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), mcoeff);
#if __USE_FMA_INSTRUCTION__
							__m512 difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							__m512 difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							__m512 _w = _mm512_set_ps(
								range_weight[buf[15]],
								range_weight[buf[14]],
								range_weight[buf[13]],
								range_weight[buf[12]],
								range_weight[buf[11]],
								range_weight[buf[10]],
								range_weight[buf[9]],
								range_weight[buf[8]],
								range_weight[buf[7]],
								range_weight[buf[6]],
								range_weight[buf[5]],
								range_weight[buf[4]],
								range_weight[buf[3]],
								range_weight[buf[2]],
								range_weight[buf[1]],
								range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_ps(bref, _w, bval);
							gval = _mm512_fmadd_ps(gref, _w, gval);
							rval = _mm512_fmadd_ps(rref, _w, rval);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval = _mm512_add_ps(bval, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval = _mm512_add_ps(gval, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval = _mm512_add_ps(rval, rref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						bval = _mm512_div_ps(bval, wval);
						gval = _mm512_div_ps(gval, wval);
						rval = _mm512_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm512_stream_ps_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float* sptrrj = sptrr + j;
					const float* sptrgj = sptrg + j;
					const float* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = range_weight[(int)sqrt(coeff * space_weight[k] + (b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
	float sigma_range, sigma_space;
};

class BilateralFilter_QuantizationLUT_Setx1_8u_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationLUT_Setx1_8u_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float _sigma_range, const float _sigma_space, const float *_space_weight, const  float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float coeff = (sigma_range * sigma_range) / (sigma_space * sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F) && checkHardwareSupport(CV_CPU_AVX_512BW);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(64) buf[16];

			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 64 * (radiusH / 64 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcoeff = _mm512_set1_ps(coeff);
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrj = sptr + j;
						const __m512i sval0_64elem = _mm512_load_si512((__m512i*)sptrj);

						__m512 tval1 = _mm512_setzero_ps();
						__m512 wval1 = _mm512_setzero_ps();
						__m512 tval2 = _mm512_setzero_ps();
						__m512 wval2 = _mm512_setzero_ps();
						__m512 tval3 = _mm512_setzero_ps();
						__m512 wval3 = _mm512_setzero_ps();
						__m512 tval4 = _mm512_setzero_ps();
						__m512 wval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i sref_64elem = _mm512_loadu_si512((__m512i*)(sptrj + *ofs));
							const __m512i diff_64elem = _mm512_add_epi8(_mm512_subs_epu8(sval0_64elem, sref_64elem), _mm512_subs_epu8(sref_64elem, sval0_64elem));
							__m512i diff_32elem = _mm512_unpacklo_epi8(diff_64elem, zero);
							__m512 diff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(diff_32elem, zero));

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), mcoeff);
							__m512 difft =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(diff, diff, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
#endif

							// lower bit
							__m512i sref_32elem = _mm512_unpacklo_epi8(sref_64elem, zero);

							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft))));
							__m512 _w = _mm512_set_ps(
								range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]],
								range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
								range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]],
								range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

							__m512 sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm512_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval1 = _mm512_add_ps(tval1, sref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							diff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(diff_32elem, zero));
							difft =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(diff, diff, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
#endif
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft))));
							_w = _mm512_set_ps(
								range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]],
								range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
								range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]],
								range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm512_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval2 = _mm512_add_ps(tval2, sref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// higher bit
							sref_32elem = _mm512_unpackhi_epi8(sref_64elem, zero);
							diff_32elem = _mm512_unpackhi_epi8(diff_64elem, zero);
							diff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(diff_32elem, zero));
							difft =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(diff, diff, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
#endif
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft))));
							_w = _mm512_set_ps(
								range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]],
								range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
								range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]],
								range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm512_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval3 = _mm512_add_ps(tval3, sref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							diff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(diff_32elem, zero));
							difft =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(diff, diff, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
#endif
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft))));
							_w = _mm512_set_ps(
								range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]],
								range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
								range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]],
								range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_32elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm512_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval4 = _mm512_add_ps(tval4, sref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						tval1 = _mm512_div_ps(tval1, wval1);
						tval2 = _mm512_div_ps(tval2, wval2);
						tval3 = _mm512_div_ps(tval3, wval3);
						tval4 = _mm512_div_ps(tval4, wval4);
						_mm512_stream_si512((__m512i*)(dptr + j), _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(tval1), _mm512_cvtps_epi32(tval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(tval3), _mm512_cvtps_epi32(tval4))));
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = range_weight[(int)sqrt(coeff * space_weight[k] + (val - val0)*(val - val0))];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)(sum / wsum);
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(64) buf[16];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 64 * (radiusH / 64 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 64 * (radiusH / 64 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 64 * (radiusH / 64 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
					const __m512 mcoeff = _mm512_set1_ps(coeff);
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m512i bval0_32elem = _mm512_load_si512((__m512i*)(sptrbj));
						const __m512i gval0_32elem = _mm512_load_si512((__m512i*)(sptrgj));
						const __m512i rval0_32elem = _mm512_load_si512((__m512i*)(sptrrj));

						__m512i temp = _mm512_unpacklo_epi8(bval0_32elem, zero);
						const __m512 bval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 bval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(bval0_32elem, zero);
						const __m512 bval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 bval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpacklo_epi8(gval0_32elem, zero);
						const __m512 gval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 gval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(gval0_32elem, zero);
						const __m512 gval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 gval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpacklo_epi8(rval0_32elem, zero);
						const __m512 rval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 rval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(rval0_32elem, zero);
						const __m512 rval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 rval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));

						__m512 wval1 = _mm512_setzero_ps();
						__m512 bval1 = _mm512_setzero_ps();
						__m512 gval1 = _mm512_setzero_ps();
						__m512 rval1 = _mm512_setzero_ps();

						__m512 wval2 = _mm512_setzero_ps();
						__m512 bval2 = _mm512_setzero_ps();
						__m512 gval2 = _mm512_setzero_ps();
						__m512 rval2 = _mm512_setzero_ps();

						__m512 wval3 = _mm512_setzero_ps();
						__m512 bval3 = _mm512_setzero_ps();
						__m512 gval3 = _mm512_setzero_ps();
						__m512 rval3 = _mm512_setzero_ps();

						__m512 wval4 = _mm512_setzero_ps();
						__m512 bval4 = _mm512_setzero_ps();
						__m512 gval4 = _mm512_setzero_ps();
						__m512 rval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i bref_64elem = _mm512_loadu_si512((__m512i*)(sptrbj + *ofs));
							const __m512i gref_64elem = _mm512_loadu_si512((__m512i*)(sptrgj + *ofs));
							const __m512i rref_64elem = _mm512_loadu_si512((__m512i*)(sptrrj + *ofs));

							// low
							__m512i bref_32elem = _mm512_unpacklo_epi8(bref_64elem, zero);
							__m512i gref_32elem = _mm512_unpacklo_epi8(gref_64elem, zero);
							__m512i rref_32elem = _mm512_unpacklo_epi8(rref_64elem, zero);

							__m512 bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							__m512 gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							__m512 rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));

							__m512 bdiff = _mm512_sub_ps(bval0_1, bref);
							__m512 gdiff = _mm512_sub_ps(gval0_1, gref);
							__m512 rdiff = _mm512_sub_ps(rval0_1, rref);
							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), mcoeff);
#if __USE_FMA_INSTRUCTION__
							__m512 difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							__m512 difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							__m512 _w = _mm512_set_ps(
								range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
								range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval1 = _mm512_fmadd_ps(bref, _w, bval1);
							gval1 = _mm512_fmadd_ps(gref, _w, gval1);
							rval1 = _mm512_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval1 = _mm512_add_ps(bval1, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval1 = _mm512_add_ps(gval1, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval1 = _mm512_add_ps(rval1, rref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_2, bref);
							gdiff = _mm512_sub_ps(gval0_2, gref);
							rdiff = _mm512_sub_ps(rval0_2, rref);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							_w = _mm512_set_ps(
								range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
								range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm512_fmadd_ps(bref, _w, bval2);
							gval2 = _mm512_fmadd_ps(gref, _w, gval2);
							rval2 = _mm512_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval2 = _mm512_add_ps(bval2, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval2 = _mm512_add_ps(gval2, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval2 = _mm512_add_ps(rval2, rref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// high
							bref_32elem = _mm512_unpackhi_epi8(bref_64elem, zero);
							gref_32elem = _mm512_unpackhi_epi8(gref_64elem, zero);
							rref_32elem = _mm512_unpackhi_epi8(rref_64elem, zero);

							bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_3, bref);
							gdiff = _mm512_sub_ps(gval0_3, gref);
							rdiff = _mm512_sub_ps(rval0_3, rref);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							_w = _mm512_set_ps(
								range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
								range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm512_fmadd_ps(bref, _w, bval3);
							gval3 = _mm512_fmadd_ps(gref, _w, gval3);
							rval3 = _mm512_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval3 = _mm512_add_ps(bval3, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval3 = _mm512_add_ps(gval3, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval3 = _mm512_add_ps(rval3, rref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_4, bref);
							gdiff = _mm512_sub_ps(gval0_4, gref);
							rdiff = _mm512_sub_ps(rval0_4, rref);
#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(difft));
							_w = _mm512_set_ps(
								range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]],
								range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]
							);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm512_fmadd_ps(bref, _w, bval4);
							gval4 = _mm512_fmadd_ps(gref, _w, gval4);
							rval4 = _mm512_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval4 = _mm512_add_ps(bval4, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval4 = _mm512_add_ps(gval4, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval4 = _mm512_add_ps(rval4, rref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						bval1 = _mm512_div_ps(bval1, wval1);
						bval2 = _mm512_div_ps(bval2, wval2);
						bval3 = _mm512_div_ps(bval3, wval3);
						bval4 = _mm512_div_ps(bval4, wval4);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(bval1), _mm512_cvtps_epi32(bval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(bval3), _mm512_cvtps_epi32(bval4)));
						gval1 = _mm512_div_ps(gval1, wval1);
						gval2 = _mm512_div_ps(gval2, wval2);
						gval3 = _mm512_div_ps(gval3, wval3);
						gval4 = _mm512_div_ps(gval4, wval4);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(gval1), _mm512_cvtps_epi32(gval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(gval3), _mm512_cvtps_epi32(gval4)));
						rval1 = _mm512_div_ps(rval1, wval1);
						rval2 = _mm512_div_ps(rval2, wval2);
						rval3 = _mm512_div_ps(rval3, wval3);
						rval4 = _mm512_div_ps(rval4, wval4);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(rval1), _mm512_cvtps_epi32(rval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(rval3), _mm512_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm512_stream_epi8_color(dptrc, a, b, c);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr + j;
					const uchar* sptrgj = sptrg + j;
					const uchar* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = range_weight[(int)sqrt(coeff * space_weight[k] + (b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
	float sigma_range, sigma_space;
};


/*************************************************
	using quantized LUT with "gather instruction" x 1
*************************************************/
class BilateralFilter_QuantizationLUT_Gatherx1_64f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationLUT_Gatherx1_64f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double _sigma_range, const double _sigma_space, const double *_space_weight, const double *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const double coeff = (sigma_range * sigma_range) / (sigma_space * sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			const double* sptr = temp->ptr<double>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d mcoeff = _mm512_set1_pd(coeff);
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrj = sptr + j;
						const __m512d sval0 = _mm512_load_pd(sptrj);

						__m512d tval = _mm512_setzero_pd();
						__m512d wval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d sref = _mm512_loadu_pd((sptrj + *ofs));
							const __m512d diff = _mm512_sub_pd(sval0, sref);

							const __m512d _sw = _mm512_mul_pd(_mm512_set1_pd(*spw), mcoeff);
							const __m512d difft =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_pd(diff, diff, _sw);
#else
								_mm512_add_pd(_mm512_mul_pd(diff, diff), _sw);
#endif
							const __m512i midx = _mm512_cvtpd_epi64(_mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, difft)));
							__m512d _w = _mm512_i64gather_pd(midx, range_weight, 8);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_pd(sref, _w, tval);
#else
							sref = _mm512_mul_pd(sref, _w);
							tval = _mm512_add_pd(tval, sref);
#endif

							wval = _mm512_add_pd(wval, _w);
						}
						tval = _mm512_div_pd(tval, wval);
						_mm512_stream_pd((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double val0 = sptr[j];
					double sum = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double val = sptr[j + space_ofs[k]];
						const double w = range_weight[(int)sqrt(coeff * space_weight[k] + (val - val0) * (val - val0))];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d mcoeff = _mm512_set1_pd(coeff);
#if __BF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const double* spw = space_weight;

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m512d bval0 = _mm512_load_pd(sptrbj);
						const __m512d gval0 = _mm512_load_pd(sptrgj);
						const __m512d rval0 = _mm512_load_pd(sptrrj);

						__m512d wval = _mm512_setzero_pd();
						__m512d bval = _mm512_setzero_pd();
						__m512d gval = _mm512_setzero_pd();
						__m512d rval = _mm512_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512d bref = _mm512_load_pd(sptrbj + *ofs);
							__m512d gref = _mm512_load_pd(sptrgj + *ofs);
							__m512d rref = _mm512_load_pd(sptrrj + *ofs);

							__m512d bdiff = _mm512_sub_pd(bval0, bref);
							__m512d gdiff = _mm512_sub_pd(gval0, gref);
							__m512d rdiff = _mm512_sub_pd(rval0, rref);

							const __m512d _sw = _mm512_mul_pd(_mm512_set1_pd(*spw), mcoeff);
#if __USE_FMA_INSTRUCTION__
							__m512d difft = _mm512_fmadd_pd(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm512_fmadd_pd(rdiff, rdiff, difft);
#else
							__m512d difft = _mm512_add_pd(_mm512_mul_pd(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_pd(gdiff, gdiff);
							difft = _mm512_add_pd(difft, gdiff);
							rdiff = _mm512_mul_pd(rdiff, rdiff);
							difft = _mm512_add_pd(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, difft);
							__m512i mmidx = _mm512_cvtpd_epi64(difft);
							__m512d _w = _mm512_i64gather_pd(mmidx, range_weight, 8);
#if __BF_POSTVENTION__
							_w = _mm512_max_pd(_w, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_pd(bref, _w, bval);
							gval = _mm512_fmadd_pd(gref, _w, gval);
							rval = _mm512_fmadd_pd(rref, _w, rval);
#else
							bref = _mm512_mul_pd(bref, _w);
							bval = _mm512_add_pd(bval, bref);
							gref = _mm512_mul_pd(gref, _w);
							gval = _mm512_add_pd(gval, gref);
							rref = _mm512_mul_pd(rref, _w);
							rval = _mm512_add_pd(rval, rref);
#endif
							wval = _mm512_add_pd(wval, _w);
						}
						bval = _mm512_div_pd(bval, wval);
						gval = _mm512_div_pd(gval, wval);
						rval = _mm512_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm512_stream_pd_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const double* sptrrj = sptrr + j;
					const double* sptrgj = sptrg + j;
					const double* sptrbj = sptrb + j;

					const double r0 = sptrrj[0];
					const double g0 = sptrgj[0];
					const double b0 = sptrbj[0];

					double sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					double wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const double r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const double w = range_weight[(int)sqrt(coeff * space_weight[k] + (b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}

private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *space_weight, *range_weight;
	double sigma_range, sigma_space;
};

class BilateralFilter_QuantizationLUT_Gatherx1_32f_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationLUT_Gatherx1_32f_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float _sigma_range, const float _sigma_space, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float coeff = (sigma_range * sigma_range) / (sigma_space * sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
#endif
		if (cn == 1)
		{
			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcoeff = _mm512_set1_ps(coeff);
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrj = sptr + j;
						const __m512 sval0 = _mm512_load_ps(sptrj);

						__m512 tval = _mm512_setzero_ps();
						__m512 wval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 sref = _mm512_loadu_ps((sptrj + *ofs));
							const __m512 diff = _mm512_sub_ps(sval0, sref);

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), mcoeff);
							const __m512 difft =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(diff, diff, _sw);
#else
								_mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
#endif
							const __m512i midx = _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft)));
							__m512 _w = _mm512_i32gather_ps(midx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							tval = _mm512_fmadd_ps(sref, _w, tval);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval = _mm512_add_ps(tval, sref);
#endif

							wval = _mm512_add_ps(wval, _w);
						}
						tval = _mm512_div_ps(tval, wval);
						_mm512_stream_ps((dptr + j), tval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = range_weight[(int)sqrt(coeff * space_weight[k] + (val - val0) * (val - val0))];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcoeff = _mm512_set1_ps(coeff);
#if QUANTIZATION_SIZE_MODE == 1
					const __m512 mQpara = _mm512_set1_ps(QUANTIZATION_SIZE_PRAMETER);
#endif
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m512 bval0 = _mm512_load_ps(sptrbj);
						const __m512 gval0 = _mm512_load_ps(sptrgj);
						const __m512 rval0 = _mm512_load_ps(sptrrj);

						__m512 wval = _mm512_setzero_ps();
						__m512 bval = _mm512_setzero_ps();
						__m512 gval = _mm512_setzero_ps();
						__m512 rval = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							__m512 bref = _mm512_load_ps(sptrbj + *ofs);
							__m512 gref = _mm512_load_ps(sptrgj + *ofs);
							__m512 rref = _mm512_load_ps(sptrrj + *ofs);

							__m512 bdiff = _mm512_sub_ps(bval0, bref);
							__m512 gdiff = _mm512_sub_ps(gval0, gref);
							__m512 rdiff = _mm512_sub_ps(rval0, rref);

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), mcoeff);
#if __USE_FMA_INSTRUCTION__
							__m512 difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							__m512 difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
#if QUANTIZATION_MODE == 0
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
#endif
#if QUANTIZATION_SIZE_MODE == 1
							difft = _mm512_mul_ps(difft, mQpara);
#endif
							__m512i mmidx = _mm512_cvtps_epi32(difft);
							__m512 _w = _mm512_i32gather_ps(mmidx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval = _mm512_fmadd_ps(bref, _w, bval);
							gval = _mm512_fmadd_ps(gref, _w, gval);
							rval = _mm512_fmadd_ps(rref, _w, rval);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval = _mm512_add_ps(bval, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval = _mm512_add_ps(gval, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval = _mm512_add_ps(rval, rref);
#endif
							wval = _mm512_add_ps(wval, _w);
						}
						bval = _mm512_div_ps(bval, wval);
						gval = _mm512_div_ps(gval, wval);
						rval = _mm512_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm512_stream_ps_color(dptrc, bval, gval, rval);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float* sptrrj = sptrr + j;
					const float* sptrgj = sptrg + j;
					const float* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = range_weight[(int)sqrt(coeff * space_weight[k] + (b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}

private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
	float sigma_range, sigma_space;
};

class BilateralFilter_QuantizationLUT_Gatherx1_8u_InvokerAVX512 : public cv::ParallelLoopBody
{
public:
	BilateralFilter_QuantizationLUT_Gatherx1_8u_InvokerAVX512(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float _sigma_range, const float _sigma_space, const float *_space_weight, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), space_weight(_space_weight), range_weight(_range_weight), sigma_range(_sigma_range), sigma_space(_sigma_space)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float coeff = (sigma_range * sigma_range) / (sigma_space * sigma_space);

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F) && checkHardwareSupport(CV_CPU_AVX_512BW);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 64 * (radiusH / 64 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zero = _mm512_setzero_si512();
					const __m512 mcoeff = _mm512_set1_ps(coeff);
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrj = sptr + j;
						const __m512i sval0_64elem = _mm512_load_si512((__m512i*)sptrj);

						__m512 tval1 = _mm512_setzero_ps();
						__m512 wval1 = _mm512_setzero_ps();
						__m512 tval2 = _mm512_setzero_ps();
						__m512 wval2 = _mm512_setzero_ps();
						__m512 tval3 = _mm512_setzero_ps();
						__m512 wval3 = _mm512_setzero_ps();
						__m512 tval4 = _mm512_setzero_ps();
						__m512 wval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i sref_64elem = _mm512_loadu_si512((__m512i*)(sptrj + *ofs));
							const __m512i diff_64elem = _mm512_add_epi8(_mm512_subs_epu8(sval0_64elem, sref_64elem), _mm512_subs_epu8(sref_64elem, sval0_64elem));

							// lower bit
							__m512i sref_16elem = _mm512_unpacklo_epi8(sref_64elem, zero);
							__m512i diff_16elem = _mm512_unpacklo_epi8(diff_64elem, zero);

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), mcoeff);
							__m512 diff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(diff_16elem, zero));
							__m512 difft = _mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
							__m512i midx = _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft)));
							__m512 _w = _mm512_i32gather_ps(midx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							__m512 sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm512_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval1 = _mm512_add_ps(tval1, sref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							diff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(diff_16elem, zero));
							difft = _mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
							midx = _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft)));
							_w = _mm512_i32gather_ps(midx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm512_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval2 = _mm512_add_ps(tval2, sref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);

							// higher bit
							sref_16elem = _mm512_unpackhi_epi8(sref_64elem, zero);
							diff_16elem = _mm512_unpackhi_epi8(diff_64elem, zero);

							diff = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(diff_16elem, zero));
							difft = _mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
							midx = _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft)));
							_w = _mm512_i32gather_ps(midx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm512_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval3 = _mm512_add_ps(tval3, sref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							diff = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(diff_16elem, zero));
							difft = _mm512_add_ps(_mm512_mul_ps(diff, diff), _sw);
							midx = _mm512_cvtps_epi32(_mm512_maskz_rcp14_ps(_MM_K0_REG, _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft)));
							_w = _mm512_i32gather_ps(midx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
							sref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm512_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm512_mul_ps(sref, _w);
							tval4 = _mm512_add_ps(tval4, sref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						tval1 = _mm512_div_ps(tval1, wval1);
						tval2 = _mm512_div_ps(tval2, wval2);
						tval3 = _mm512_div_ps(tval3, wval3);
						tval4 = _mm512_div_ps(tval4, wval4);
						_mm512_stream_si512((__m512i*)(dptr + j), _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(tval1), _mm512_cvtps_epi32(tval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(tval3), _mm512_cvtps_epi32(tval4))));
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const float val0 = sptr[j];
					float sum = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float val = sptr[j + space_ofs[k]];
						const float w = space_weight[k] * range_weight[int(abs(val - val0))];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (int)(sum / wsum);
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 64 * (radiusH / 64 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 64 * (radiusH / 64 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 64 * (radiusH / 64 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 mcoeff = _mm512_set1_ps(coeff);
					static const __m512i zero = _mm512_setzero_si512();
#if __BF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 64)//64 pixel unit
					{
						const int* ofs = &space_ofs[0];
						const float* spw = space_weight;

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m512i bval0_64elem = _mm512_load_si512((__m512i*)(sptrbj));
						const __m512i gval0_64elem = _mm512_load_si512((__m512i*)(sptrgj));
						const __m512i rval0_64elem = _mm512_load_si512((__m512i*)(sptrrj));

						__m512i temp = _mm512_unpacklo_epi8(bval0_64elem, zero);
						const __m512 bval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 bval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(bval0_64elem, zero);
						const __m512 bval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 bval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpacklo_epi8(gval0_64elem, zero);
						const __m512 gval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 gval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(gval0_64elem, zero);
						const __m512 gval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 gval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpacklo_epi8(rval0_64elem, zero);
						const __m512 rval0_1 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 rval0_2 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));
						temp = _mm512_unpackhi_epi8(rval0_64elem, zero);
						const __m512 rval0_3 = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp, zero));
						const __m512 rval0_4 = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp, zero));

						__m512 wval1 = _mm512_setzero_ps();
						__m512 bval1 = _mm512_setzero_ps();
						__m512 gval1 = _mm512_setzero_ps();
						__m512 rval1 = _mm512_setzero_ps();

						__m512 wval2 = _mm512_setzero_ps();
						__m512 bval2 = _mm512_setzero_ps();
						__m512 gval2 = _mm512_setzero_ps();
						__m512 rval2 = _mm512_setzero_ps();

						__m512 wval3 = _mm512_setzero_ps();
						__m512 bval3 = _mm512_setzero_ps();
						__m512 gval3 = _mm512_setzero_ps();
						__m512 rval3 = _mm512_setzero_ps();

						__m512 wval4 = _mm512_setzero_ps();
						__m512 bval4 = _mm512_setzero_ps();
						__m512 gval4 = _mm512_setzero_ps();
						__m512 rval4 = _mm512_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++, spw++)
						{
							const __m512i bref_64elem = _mm512_loadu_si512((__m512i*)(sptrbj + *ofs));
							const __m512i gref_64elem = _mm512_loadu_si512((__m512i*)(sptrgj + *ofs));
							const __m512i rref_64elem = _mm512_loadu_si512((__m512i*)(sptrrj + *ofs));

							// low
							__m512i bref_32elem = _mm512_unpacklo_epi8(bref_64elem, zero);
							__m512i gref_32elem = _mm512_unpacklo_epi8(gref_64elem, zero);
							__m512i rref_32elem = _mm512_unpacklo_epi8(rref_64elem, zero);

							__m512 bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							__m512 gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							__m512 rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));

							__m512 bdiff = _mm512_sub_ps(bval0_1, bref);
							__m512 gdiff = _mm512_sub_ps(gval0_1, gref);
							__m512 rdiff = _mm512_sub_ps(rval0_1, rref);

							const __m512 _sw = _mm512_mul_ps(_mm512_set1_ps(*spw), mcoeff);
#if __USE_FMA_INSTRUCTION__
							__m512 difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							__m512 difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);

							__m512i mmidx = _mm512_cvtps_epi32(difft);
							__m512 _w = _mm512_i32gather_ps(mmidx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm512_fmadd_ps(bref, _w, bval1);
							gval1 = _mm512_fmadd_ps(gref, _w, gval1);
							rval1 = _mm512_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval1 = _mm512_add_ps(bval1, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval1 = _mm512_add_ps(gval1, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval1 = _mm512_add_ps(rval1, rref);
#endif
							wval1 = _mm512_add_ps(wval1, _w);


							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_2, bref);
							gdiff = _mm512_sub_ps(gval0_2, gref);
							rdiff = _mm512_sub_ps(rval0_2, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							mmidx = _mm512_cvtps_epi32(difft);
							_w = _mm512_i32gather_ps(mmidx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm512_fmadd_ps(bref, _w, bval2);
							gval2 = _mm512_fmadd_ps(gref, _w, gval2);
							rval2 = _mm512_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval2 = _mm512_add_ps(bval2, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval2 = _mm512_add_ps(gval2, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval2 = _mm512_add_ps(rval2, rref);
#endif
							wval2 = _mm512_add_ps(wval2, _w);


							// high
							bref_32elem = _mm512_unpackhi_epi8(bref_64elem, zero);
							gref_32elem = _mm512_unpackhi_epi8(gref_64elem, zero);
							rref_32elem = _mm512_unpackhi_epi8(rref_64elem, zero);

							bref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_3, bref);
							gdiff = _mm512_sub_ps(gval0_3, gref);
							rdiff = _mm512_sub_ps(rval0_3, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);

							mmidx = _mm512_cvtps_epi32(difft);
							_w = _mm512_i32gather_ps(mmidx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm512_fmadd_ps(bref, _w, bval3);
							gval3 = _mm512_fmadd_ps(gref, _w, gval3);
							rval3 = _mm512_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval3 = _mm512_add_ps(bval3, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval3 = _mm512_add_ps(gval3, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval3 = _mm512_add_ps(rval3, rref);
#endif
							wval3 = _mm512_add_ps(wval3, _w);


							bref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(bref_32elem, zero));
							gref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(gref_32elem, zero));
							rref = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(rref_32elem, zero));

							bdiff = _mm512_sub_ps(bval0_4, bref);
							gdiff = _mm512_sub_ps(gval0_4, gref);
							rdiff = _mm512_sub_ps(rval0_4, rref);
							difft = _mm512_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm512_fmadd_ps(bdiff, bdiff, _sw);
							difft = _mm512_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm512_fmadd_ps(rdiff, rdiff, difft);
#else
							difft = _mm512_add_ps(_mm512_mul_ps(bdiff, bdiff), _sw);
							gdiff = _mm512_mul_ps(gdiff, gdiff);
							difft = _mm512_add_ps(difft, gdiff);
							rdiff = _mm512_mul_ps(rdiff, rdiff);
							difft = _mm512_add_ps(difft, rdiff);
#endif
							difft = _mm512_maskz_rsqrt14_ps(_MM_K0_REG, difft);
							difft = _mm512_maskz_rcp14_ps(_MM_K0_REG, difft);
							mmidx = _mm512_cvtps_epi32(difft);
							_w = _mm512_i32gather_ps(mmidx, range_weight, 4);
#if __BF_POSTVENTION__
							_w = _mm512_max_ps(_w, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm512_fmadd_ps(bref, _w, bval4);
							gval4 = _mm512_fmadd_ps(gref, _w, gval4);
							rval4 = _mm512_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm512_mul_ps(bref, _w);
							bval4 = _mm512_add_ps(bval4, bref);
							gref = _mm512_mul_ps(gref, _w);
							gval4 = _mm512_add_ps(gval4, gref);
							rref = _mm512_mul_ps(rref, _w);
							rval4 = _mm512_add_ps(rval4, rref);
#endif
							wval4 = _mm512_add_ps(wval4, _w);
						}
						bval1 = _mm512_div_ps(bval1, wval1);
						bval2 = _mm512_div_ps(bval2, wval2);
						bval3 = _mm512_div_ps(bval3, wval3);
						bval4 = _mm512_div_ps(bval4, wval4);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(bval1), _mm512_cvtps_epi32(bval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(bval3), _mm512_cvtps_epi32(bval4)));
						gval1 = _mm512_div_ps(gval1, wval1);
						gval2 = _mm512_div_ps(gval2, wval2);
						gval3 = _mm512_div_ps(gval3, wval3);
						gval4 = _mm512_div_ps(gval4, wval4);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(gval1), _mm512_cvtps_epi32(gval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(gval3), _mm512_cvtps_epi32(gval4)));
						rval1 = _mm512_div_ps(rval1, wval1);
						rval2 = _mm512_div_ps(rval2, wval2);
						rval3 = _mm512_div_ps(rval3, wval3);
						rval4 = _mm512_div_ps(rval4, wval4);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(rval1), _mm512_cvtps_epi32(rval2)), _mm512_packs_epi32(_mm512_cvtps_epi32(rval3), _mm512_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm512_stream_epi8_color(dptrc, a, b, c);
					}
				}
#endif
				for (; j < size.width; j++)
				{
					const uchar* sptrrj = sptrr + j;
					const uchar* sptrgj = sptrg + j;
					const uchar* sptrbj = sptrb + j;

					const float r0 = sptrrj[0];
					const float g0 = sptrgj[0];
					const float b0 = sptrbj[0];

					float sum_r = 0.0f, sum_b = 0.0f, sum_g = 0.0f;
					float wsum = 0.0f;
					for (k = 0; k < maxk; k++)
					{
						const float r = sptrrj[space_ofs[k]], g = sptrgj[space_ofs[k]], b = sptrbj[space_ofs[k]];
						const float w = range_weight[(int)sqrt(coeff * space_weight[k] + (b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}
private:
	const Mat *temp;

	Mat *dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *space_weight, *range_weight;
	float sigma_range, sigma_space;
};


namespace bf
{

	void bilateralFilter_AVX512_64f(const Mat& src, Mat& dst, const Size kernelSize, double sigma_range, double sigma_space, const int borderType, const bool isRectangle, const WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0) { src.copyTo(dst); return; }
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_64FC1 || src.type() == CV_64FC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
			sigma_range = 1;
		if (sigma_space <= 0)
			sigma_space = 1;

		const double gauss_range_coeff = -0.5 / (sigma_range*sigma_range);
		const double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		const int radiusH = kernelSize.width >> 1;
		const int radiusV = kernelSize.height >> 1;

		Mat temp;

		const int dpad = (8 - src.cols % 8) % 8;
		int spad = dpad + (8 - (2 * radiusH) % 8) % 8;
		if (spad < 8) spad += 8;
		const int lpad = 8 * (radiusH / 8 + 1) - radiusH;
		const int rpad = spad - lpad;

		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleaveAVX(temp2, temp);
		}

		vector<double> _space_weight(kernelSize.area());
		vector<int> _space_ofs(kernelSize.area());
		double* space_weight = &_space_weight[0];
		int* space_ofs = &_space_ofs[0];

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);

			const BilateralFilter_EXP_64f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, sigma_range, sigma_space);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_VECTOR_EXP_WITH_SPACE_LUT:
		{
#if __BF_PREVENTION__
			const double max_digits = floor(log2(DBL_MAX / (255.*kernelSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - 2.) / 2.);
			const double bias = pow(2, log2(DBL_MAX / (255.*kernelSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#else
			const double exp_clip_val = -200;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			const BilateralFilter_EXP_With_SpaceLUT_64f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, sigma_range, exp_clip_val);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
#if __BF_PREVENTION__
			const double max_digits = floor(log2(DBL_MAX / (255.*kernelSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - (cn + 1)) / (cn + 1));
			const double bias = pow(2, log2(DBL_MAX / (255.*kernelSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 256;
			vector<double> _range_weight(value_range);
			double* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], DBL_MIN);
#endif
			}

			const BilateralFilter_LUT_Setx3_64f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
		{
#if __BF_PREVENTION__
			const double max_digits = floor(log2(DBL_MAX / (255.*kernelSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - (cn + 1)) / (cn + 1));
			const double bias = pow(2, log2(DBL_MAX / (255.*kernelSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 256;
			vector<double> _range_weight(value_range);
			double* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], DBL_MIN);
#endif
			}

			const BilateralFilter_LUT_Gatherx3_64f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
#if __BF_PREVENTION__
			const double max_digits = floor(log2(DBL_MAX / (255.*kernelSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - 2.) / 2.);
			const double bias = pow(2, log2(DBL_MAX / (255.*kernelSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 442;
			vector<double> _range_weight(value_range);
			double* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], DBL_MIN);
#endif
			}

			const BilateralFilter_QuantizationRangeLUT_Setx1_64f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHER:
		{
#if __BF_PREVENTION__
			const double max_digits = floor(log2(DBL_MAX / (255.*kernelSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - 2.) / 2.);
			const double bias = pow(2, log2(DBL_MAX / (255.*kernelSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 442;
			vector<double> _range_weight(value_range);
			double* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], DBL_MIN);
#endif
			}

			const BilateralFilter_QuantizationRangeLUT_Gatherx1_64f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_MERGED_QUANTIZATION_LUT_SET:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_MERGED_QUANTIZATION_LUT_SET);

			const double coef = (sigma_range * sigma_range) / (sigma_space * sigma_space);
			const double max_s = radiusH * radiusH + radiusV * radiusV;
			const double max_r = 255 * 255 * cn;

			const int value_range = ceil(sqrt(coef * max_s + max_r));
			vector<double> _weight(value_range);
			double* range_weight = &_weight[0];

			// initialize color-related bilateral filter coefficients
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], DBL_MIN);
#endif
			}

			const BilateralFilter_QuantizationLUT_Setx1_64f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range, sigma_space, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_MERGER_QUANTIZATION_LUT_GATHER:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_MERGER_QUANTIZATION_LUT_GATHER);

			const double coef = (sigma_range * sigma_range) / (sigma_space * sigma_space);
			const double max_s = radiusH * radiusH + radiusV * radiusV;
			const double max_r = 255 * 255 * cn;

			const int value_range = ceil(sqrt(coef * max_s + max_r));
			vector<double> _weight(value_range);
			double* range_weight = &_weight[0];

			// initialize color-related bilateral filter coefficients
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], DBL_MIN);
#endif
			}

			const BilateralFilter_QuantizationLUT_Gatherx1_64f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range, sigma_space, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		default:
			break;
		}
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void bilateralFilter_AVX512_32f(const Mat& src, Mat& dst, const Size kernelSize, float sigma_range, float sigma_space, const int borderType, const bool isRectangle, const WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0) { src.copyTo(dst); return; }
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
			sigma_range = 1;
		if (sigma_space <= 0)
			sigma_space = 1;

		const float gauss_range_coeff = -0.5 / (sigma_range*sigma_range);
		const float gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

		const int radiusH = kernelSize.width >> 1;
		const int radiusV = kernelSize.height >> 1;

		Mat temp;

		const int dpad = (16 - src.cols % 16) % 16;
		int spad = dpad + (16 - (2 * radiusH) % 16) % 16;
		if (spad < 16) spad += 16;
		const int lpad = 16 * (radiusH / 16 + 1) - radiusH;
		const int rpad = spad - lpad;

		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleaveAVX(temp2, temp);
		}

		vector<float> _space_weight(kernelSize.area());
		vector<int> _space_ofs(kernelSize.area());
		float* space_weight = &_space_weight[0];
		int* space_ofs = &_space_ofs[0];

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);

			const BilateralFilter_EXP_32f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, sigma_range, sigma_space);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_VECTOR_EXP_WITH_SPACE_LUT:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - 2.f) / 2.f);
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float exp_clip_val = -200;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			const BilateralFilter_EXP_With_SpaceLUT_32f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, sigma_range, exp_clip_val);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - (cn + 1)) / (cn + 1));
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 256;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_LUT_Setx3_32f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - (cn + 1)) / (cn + 1));
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 256;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_LUT_Gatherx3_32f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - 2.f) / 2.f);
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 442;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_QuantizationRangeLUT_Setx1_32f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHER:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - 2.f) / 2.f);
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
#if QUANTIZATION_MODE == 1
#if QUANTIZATION_SIZE_MODE == 0
			const int value_range = 255 * 255 * cn + 1;
#else
			const int value_range = (255 * 255 * cn + 1)*QUANTIZATION_SIZE_PRAMETER;
#endif
#else
#if QUANTIZATION_SIZE_MODE == 0
			const int value_range = 442;
#else
			const int value_range = 442 * QUANTIZATION_SIZE_PRAMETER;
#endif
#endif
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
#if QUANTIZATION_MODE == 1
#if QUANTIZATION_SIZE_MODE == 0
				float aw = i * gauss_range_coeff;
#else
				float aw = (i / QUANTIZATION_SIZE_PRAMETER) * gauss_range_coeff;
#endif
#else
#if QUANTIZATION_SIZE_MODE == 0
				float aw = i * i * gauss_range_coeff;
#else
				float aw = (i / QUANTIZATION_SIZE_PRAMETER) * (i / QUANTIZATION_SIZE_PRAMETER) * gauss_range_coeff;
#endif
#endif
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_QuantizationRangeLUT_Gatherx1_32f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_MERGED_QUANTIZATION_LUT_SET:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_MERGED_QUANTIZATION_LUT_SET);

			const float coef = (sigma_range * sigma_range) / (sigma_space * sigma_space);
			const float max_s = radiusH * radiusH + radiusV * radiusV;
			const float max_r = 255 * 255 * cn;

			const int value_range = ceil(sqrt(coef * max_s + max_r));
			vector<float> _weight(value_range);
			float* range_weight = &_weight[0];

			// initialize color-related bilateral filter coefficients
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_QuantizationLUT_Setx1_32f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range, sigma_space, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_MERGER_QUANTIZATION_LUT_GATHER:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_MERGER_QUANTIZATION_LUT_GATHER);

			const float coef = (sigma_range * sigma_range) / (sigma_space * sigma_space);
			const float max_s = radiusH * radiusH + radiusV * radiusV;
			const float max_r = 255 * 255 * cn;

#if QUANTIZATION_MODE == 1
#if QUANTIZATION_SIZE_MODE == 0
			const int value_range = ceil(coef * max_s + max_r);
#else
			const int value_range = ceil(coef * max_s + max_r*QUANTIZATION_SIZE_PRAMETER);
#endif
#else
#if QUANTIZATION_SIZE_MODE == 0
			const int value_range = ceil(sqrt(coef * max_s + max_r));
#else
			const int value_range = ceil(sqrt(coef * max_s + max_r))*QUANTIZATION_SIZE_PRAMETER;
#endif
#endif
			vector<float> _weight(value_range);
			float* range_weight = &_weight[0];

			// initialize color-related bilateral filter coefficients
			for (int i = 0; i < value_range; i++)
			{
#if QUANTIZATION_MODE == 1
#if QUANTIZATION_SIZE_MODE == 0
				float aw = i * gauss_range_coeff;
#else
				float aw = (i / QUANTIZATION_SIZE_PRAMETER) * gauss_range_coeff;
#endif
#else
#if QUANTIZATION_SIZE_MODE == 0
				float aw = i * i * gauss_range_coeff;
#else
				float aw = (i / QUANTIZATION_SIZE_PRAMETER) * (i / QUANTIZATION_SIZE_PRAMETER) * gauss_range_coeff;
#endif
#endif
#if __BF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_QuantizationLUT_Gatherx1_32f_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range, sigma_space, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		default:
			break;
		}
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void bilateralFilter_AVX512_8u(const Mat& src, Mat& dst, const Size kernelSize, float sigma_range, float sigma_space, const int borderType, const bool isRectangle, const WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0) { src.copyTo(dst); return; }
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
			sigma_range = 1;
		if (sigma_space <= 0)
			sigma_space = 1;

		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);
		const float gauss_space_coeff = -0.5f / (sigma_space*sigma_space);

		const int radiusH = kernelSize.width >> 1;
		const int radiusV = kernelSize.height >> 1;

		Mat temp;

		const int dpad = (64 - src.cols % 64) % 64;
		int spad = dpad + (64 - (2 * radiusH) % 64) % 64;
		if (spad < 64) spad += 64;
		const int lpad = 64 * (radiusH / 64 + 1) - radiusH;
		const int rpad = spad - lpad;
		if (cn == 1)
		{
			Mat temp2, temp3;
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleaveAVX(temp2, temp);
		}

		vector<float> _space_weight(kernelSize.area());
		vector<int> _space_ofs(kernelSize.area());
		float* space_weight = &_space_weight[0];
		int* space_ofs = &_space_ofs[0];

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), src.type());
		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);

			const BilateralFilter_EXP_8u_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, sigma_range, sigma_space);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_VECTOR_EXP_WITH_SPACE_LUT:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - 2.f) / 2.f);
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float exp_clip_val = -200;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			const BilateralFilter_EXP_With_SpaceLUT_8u_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, sigma_range, exp_clip_val);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - (cn + 1)) / (cn + 1));
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 256;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_LUT_Setx3_8u_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - (cn + 1)) / (cn + 1));
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 256;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_LUT_Gatherx3_8u_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - 2.f) / 2.f);
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 442;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_QuantizationRangeLUT_Setx1_8u_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHER:
		{
#if __BF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - 2.f) / 2.f);
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#endif
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
#if __BF_PREVENTION__
			setSpaceKernel_expArgClip(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, exp_clip_val, bias);
#elif __BF_POSTVENTION__
			setSpaceKernel_denormalSuppression(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle);
#else
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_VECTOR_EXP);
#endif

			// initialize color-related bilateral filter coefficients
			const int value_range = 442;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_QuantizationRangeLUT_Gatherx1_8u_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_MERGED_QUANTIZATION_LUT_SET:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_MERGED_QUANTIZATION_LUT_SET);

			const float coef = (sigma_range * sigma_range) / (sigma_space * sigma_space);
			const float max_s = radiusH * radiusH + radiusV * radiusV;
			const float max_r = 255 * 255 * cn;

			const int value_range = ceil(sqrt(coef * max_s + max_r));
			vector<float> _weight(value_range);
			float* range_weight = &_weight[0];

			// initialize color-related bilateral filter coefficients
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_QuantizationLUT_Setx1_8u_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range, sigma_space, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_MERGER_QUANTIZATION_LUT_GATHER:
		{
			// initialize space-related bilateral filter coefficients
			int maxk = 0;
			setSpaceKernel(space_weight, space_ofs, maxk, radiusH, radiusV, gauss_space_coeff, temp.cols * cn, isRectangle, weightingMethod != WEIGHT_MERGER_QUANTIZATION_LUT_GATHER);

			const float coef = (sigma_range * sigma_range) / (sigma_space * sigma_space);
			const float max_s = radiusH * radiusH + radiusV * radiusV;
			const float max_r = 255 * 255 * cn;

			const int value_range = ceil(sqrt(coef * max_s + max_r));
			vector<float> _weight(value_range);
			float* range_weight = &_weight[0];

			// initialize color-related bilateral filter coefficients
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __BF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				range_weight[i] = exp(aw);
#if __BF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const BilateralFilter_QuantizationLUT_Gatherx1_8u_InvokerAVX512 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range, sigma_space, space_weight, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		default:
			break;
		}
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	/**
	 * \brief
	 * \param src_
	 * \param dest
	 * \param kernelSize
	 * \param sigma_range
	 * \param sigma_space
	 * \param kernel_type
	 * \param borderType
	 * \param weightingMethod
	 */
	void bilateralFilter_AVX512(cv::InputArray src_, cv::OutputArray dest, const cv::Size kernelSize, const double sigma_range, const double sigma_space, const int kernel_type, const int borderType, const WEIGHT_MODE weightingMethod)
	{
		if (dest.empty() || dest.type() != src_.type() || dest.size() != src_.size()) dest.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dst = dest.getMat();

		switch (kernel_type)
		{
		case FILTER_RECTANGLE:
		{
			switch (src.depth())
			{
			case CV_8U:
			{
				bilateralFilter_AVX512_8u(src, dst, kernelSize, sigma_range, sigma_space, borderType, true, weightingMethod);
				break;
			}
			case CV_32F:
			{
				bilateralFilter_AVX512_32f(src, dst, kernelSize, sigma_range, sigma_space, borderType, true, weightingMethod);
				break;
			}
			case CV_64F:
			{
				bilateralFilter_AVX512_64f(src, dst, kernelSize, sigma_range, sigma_space, borderType, true, weightingMethod);
				break;
			}
			default:
				break;
			}
			break;
		}
		case FILTER_CIRCLE:
		case FILTER_SEPARABLE:
		default:
			break;
		}

	}
}
