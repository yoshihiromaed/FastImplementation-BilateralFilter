#include "color.h"
#include "filter.h"
#include "fmath.hpp"
#include "simd_util.h"

using namespace std;
using namespace cv;


/*************************************************
	using exp function
*************************************************/
class GaussianRangeFilter_EXP_64f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_EXP_64f_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double _sigma_range) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), sigma_range(_sigma_range)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const double gauss_range_coeff = -0.5 / (sigma_range*sigma_range);

#if CV_SSE4_1
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			const double* sptr = temp->ptr<double>(range.start + radiusV) + 2 * (radiusH / 2 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					const __m128d mcw = _mm_set1_pd(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m128d exp_arg_min = _mm_set1_pd(EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
#if __GRF_POSTVENTION__
					static const __m128d double_min = _mm_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 2)//2 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrj = sptr + j;
						const __m128d sval0 = _mm_load_pd(sptrj);

						__m128d tval = _mm_set1_pd(0.f);
						__m128d wval = _mm_set1_pd(0.f);

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128d sref = _mm_loadu_pd((sptrj + *ofs));

							const __m128d  mvdiff = _mm_sub_pd(sval0, sref);

							__m128d aw = _mm_mul_pd(_mm_mul_pd(mvdiff, mvdiff), mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_pd(aw, exp_arg_min);
#endif
							__m128d _w =
#if __USE_INTEL_EXP__
								_mm_exp_pd(aw);
#else
								//fmath::exp_pd(aw); //TODO: exp_pd
								_mm_set1_pd(1);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_pd(_w, double_min);
#endif
							sref = _mm_mul_pd(_w, sref);
							tval = _mm_add_pd(tval, sref);
							wval = _mm_add_pd(wval, _w);
						}
						tval = _mm_div_pd(tval, wval);
						_mm_stream_pd((dptr + j), tval);
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
						const double w = exp(gauss_range_coeff * ((val - val0)*(val - val0)));
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
			const int dstep = dest->cols * 3;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start + 0) + 2 * (radiusH / 2 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 2 * (radiusH / 2 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 2 * (radiusH / 2 + 1);

			double* dptr = dest->ptr<double>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					const __m128d mcw = _mm_set1_pd(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m128d exp_arg_min = _mm_set1_pd(EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
#if __GRF_POSTVENTION__
					static const __m128d double_min = _mm_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 2)//2 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrrj = sptrr + j;
						const double* sptrgj = sptrg + j;
						const double* sptrbj = sptrb + j;

						const __m128d bval0 = _mm_load_pd((sptrbj));
						const __m128d gval0 = _mm_load_pd((sptrgj));
						const __m128d rval0 = _mm_load_pd((sptrrj));

						__m128d wval = _mm_set1_pd(0.0f);
						__m128d bval = _mm_set1_pd(0.0f);
						__m128d gval = _mm_set1_pd(0.0f);
						__m128d rval = _mm_set1_pd(0.0f);

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128d bref = _mm_loadu_pd((sptrbj + *ofs));
							__m128d gref = _mm_loadu_pd((sptrgj + *ofs));
							__m128d rref = _mm_loadu_pd((sptrrj + *ofs));

							__m128d  mvdiff = _mm_sub_pd(bval0, bref);
							__m128d  mvdiffa = _mm_mul_pd(mvdiff, mvdiff);

							mvdiff = _mm_sub_pd(gval0, gref);
							mvdiffa = _mm_add_pd(mvdiffa, _mm_mul_pd(mvdiff, mvdiff));

							mvdiff = _mm_sub_pd(rval0, rref);
							mvdiffa = _mm_add_pd(mvdiffa, _mm_mul_pd(mvdiff, mvdiff));

							__m128d aw = _mm_mul_pd(mvdiffa, mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_pd(aw, exp_arg_min);
#endif
							__m128d _w =
#if __USE_INTEL_EXP__
								_mm_exp_pd(aw);
#else
								//fmath::exp_pd(aw); //TODO: exp_pd
								_mm_set1_pd(1);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_pd(_w, double_min);
#endif
							bref = _mm_mul_pd(_w, bref);
							gref = _mm_mul_pd(_w, gref);
							rref = _mm_mul_pd(_w, rref);

							bval = _mm_add_pd(bval, bref);
							gval = _mm_add_pd(gval, gref);
							rval = _mm_add_pd(rval, rref);
							wval = _mm_add_pd(wval, _w);
						}
						rval = _mm_div_pd(rval, wval);
						gval = _mm_div_pd(gval, wval);
						bval = _mm_div_pd(bval, wval);

						double* dptrc = dptr + 3 * j;
						_mm_stream_pd_color(dptrc, bval, gval, rval);
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
						const double w = exp(gauss_range_coeff*((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
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
	const int *space_ofs;
	double sigma_range;
};

class GaussianRangeFilter_EXP_32f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_EXP_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float _sigma_range) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), sigma_range(_sigma_range)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);

#if CV_SSE4_1
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					const __m128 mcw = _mm_set1_ps(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m128 exp_arg_min = _mm_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __GRF_POSTVENTION__
					static const __m128 float_min = _mm_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrj = sptr + j;
						const __m128 sval0 = _mm_load_ps(sptrj);

						__m128 tval = _mm_setzero_ps();
						__m128 wval = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128 sref = _mm_loadu_ps((sptrj + *ofs));

							const __m128  mvdiff = _mm_sub_ps(sval0, sref);
							__m128 aw = _mm_mul_ps(_mm_mul_ps(mvdiff, mvdiff), mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							__m128 _w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif

							sref = _mm_mul_ps(_w, sref);
							tval = _mm_add_ps(tval, sref);
							wval = _mm_add_ps(wval, _w);
						}
						tval = _mm_div_ps(tval, wval);
						_mm_stream_ps((dptr + j), tval);
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

						const float w = exp(gauss_range_coeff * ((val - val0)*(val - val0)));
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
			const int dstep = dest->cols * 3;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

			float* dptr = dest->ptr<float>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					const __m128 mcw = _mm_set1_ps(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m128 exp_arg_min = _mm_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __GRF_POSTVENTION__
					static const __m128 float_min = _mm_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;

						const __m128 bval0 = _mm_load_ps((sptrbj));
						const __m128 gval0 = _mm_load_ps((sptrgj));
						const __m128 rval0 = _mm_load_ps((sptrrj));

						__m128 wval = _mm_setzero_ps();
						__m128 bval = _mm_setzero_ps();
						__m128 gval = _mm_setzero_ps();
						__m128 rval = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128 bref = _mm_loadu_ps((sptrbj + *ofs));
							__m128 gref = _mm_loadu_ps((sptrgj + *ofs));
							__m128 rref = _mm_loadu_ps((sptrrj + *ofs));

							__m128  mvdiff = mvdiff = _mm_sub_ps(bval0, bref);
							__m128  mvdiffa = _mm_mul_ps(mvdiff, mvdiff);

							mvdiff = _mm_sub_ps(gval0, gref);
							mvdiffa = _mm_add_ps(mvdiffa, _mm_mul_ps(mvdiff, mvdiff));

							mvdiff = _mm_sub_ps(rval0, rref);
							mvdiffa = _mm_add_ps(mvdiffa, _mm_mul_ps(mvdiff, mvdiff));
							__m128 aw = _mm_mul_ps(mvdiffa, mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							__m128 _w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif

							bref = _mm_mul_ps(_w, bref);
							gref = _mm_mul_ps(_w, gref);
							rref = _mm_mul_ps(_w, rref);

							bval = _mm_add_ps(bval, bref);
							gval = _mm_add_ps(gval, gref);
							rval = _mm_add_ps(rval, rref);
							wval = _mm_add_ps(wval, _w);
						}
						rval = _mm_div_ps(rval, wval);
						gval = _mm_div_ps(gval, wval);
						bval = _mm_div_ps(bval, wval);

						float* dptrc = dptr + 3 * j;
						_mm_stream_ps_color(dptrc, bval, gval, rval);
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
						const float w = exp(gauss_range_coeff*((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
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
	const int *space_ofs;
	float sigma_range;
};

class GaussianRangeFilter_EXP_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_EXP_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float _sigma_range) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), sigma_range(_sigma_range)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);

#if CV_SSE4_1
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					const __m128 mcw = _mm_set1_ps(gauss_range_coeff);
					static const __m128i zero = _mm_setzero_si128();
#if __GRF_PREVENTION__
					static const __m128 exp_arg_min = _mm_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __GRF_POSTVENTION__
					static const __m128 float_min = _mm_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrj = sptr + j;
						const __m128i sval0 = _mm_load_si128((__m128i*)sptrj);

						__m128 wval1 = _mm_setzero_ps();
						__m128 tval1 = _mm_setzero_ps();
						__m128 wval2 = _mm_setzero_ps();
						__m128 tval2 = _mm_setzero_ps();
						__m128 wval3 = _mm_setzero_ps();
						__m128 tval3 = _mm_setzero_ps();
						__m128 wval4 = _mm_setzero_ps();
						__m128 tval4 = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
							const __m128i mvdiff = _mm_add_epi8(_mm_subs_epu8(sval0, sref), _mm_subs_epu8(sref, sval0));

							// low
							__m128i m = _mm_unpacklo_epi8(sref, zero);
							__m128i d = _mm_unpacklo_epi8(mvdiff, zero);
							__m128 _diffF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(d, zero));

							__m128 aw = _mm_mul_ps(_mm_mul_ps(_diffF, _diffF), mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							__m128 _w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_mul_ps(_w, float_min);
#endif
							__m128 _valF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(m, zero));
							_valF = _mm_mul_ps(_w, _valF);
							tval1 = _mm_add_ps(tval1, _valF);
							wval1 = _mm_add_ps(wval1, _w);


							_diffF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(d, zero));
							aw = _mm_mul_ps(_mm_mul_ps(_diffF, _diffF), mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_valF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(m, zero));
							_valF = _mm_mul_ps(_w, _valF);
							tval2 = _mm_add_ps(tval2, _valF);
							wval2 = _mm_add_ps(wval2, _w);


							// high
							m = _mm_unpackhi_epi8(sref, zero);
							d = _mm_unpackhi_epi8(mvdiff, zero);

							_diffF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(d, zero));
							aw = _mm_mul_ps(_mm_mul_ps(_diffF, _diffF), mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_valF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(m, zero));
							_valF = _mm_mul_ps(_w, _valF);
							wval3 = _mm_add_ps(wval3, _w);
							tval3 = _mm_add_ps(tval3, _valF);


							_diffF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(d, zero));
							aw = _mm_mul_ps(_mm_mul_ps(_diffF, _diffF), mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_valF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(m, zero));
							_valF = _mm_mul_ps(_w, _valF);
							wval4 = _mm_add_ps(wval4, _w);
							tval4 = _mm_add_ps(tval4, _valF);
						}
						tval1 = _mm_div_ps(tval1, wval1);
						tval2 = _mm_div_ps(tval2, wval2);
						tval3 = _mm_div_ps(tval3, wval3);
						tval4 = _mm_div_ps(tval4, wval4);
						_mm_stream_si128((__m128i*)(dptr + j), _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)), _mm_packs_epi32(_mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
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
						const float w = exp(gauss_range_coeff * ((val - val0)*(val - val0)));
						sum += val * w;
						wsum += w;
					}
					dptr[j] = cvRound(sum / wsum);
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = dest->cols * 3;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					const __m128 mcw = _mm_set1_ps(gauss_range_coeff);
					static const __m128i zero = _mm_setzero_si128();
#if __GRF_PREVENTION__
					static const __m128 exp_arg_min = _mm_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __GRF_POSTVENTION__
					static const __m128 float_min = _mm_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m128i bval0 = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval0 = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval0 = _mm_load_si128((__m128i*)(sptrrj));

						__m128 wval1 = _mm_setzero_ps();
						__m128 bval1 = _mm_setzero_ps();
						__m128 gval1 = _mm_setzero_ps();
						__m128 rval1 = _mm_setzero_ps();

						__m128 wval2 = _mm_setzero_ps();
						__m128 bval2 = _mm_setzero_ps();
						__m128 gval2 = _mm_setzero_ps();
						__m128 rval2 = _mm_setzero_ps();

						__m128 wval3 = _mm_setzero_ps();
						__m128 bval3 = _mm_setzero_ps();
						__m128 gval3 = _mm_setzero_ps();
						__m128 rval3 = _mm_setzero_ps();

						__m128 wval4 = _mm_setzero_ps();
						__m128 bval4 = _mm_setzero_ps();
						__m128 gval4 = _mm_setzero_ps();
						__m128 rval4 = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
							const __m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
							const __m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));
							const __m128i bdiff = _mm_add_epi8(_mm_subs_epu8(bval0, bref), _mm_subs_epu8(bref, bval0));
							const __m128i gdiff = _mm_add_epi8(_mm_subs_epu8(gval0, gref), _mm_subs_epu8(gref, gval0));
							const __m128i rdiff = _mm_add_epi8(_mm_subs_epu8(rval0, rref), _mm_subs_epu8(rref, rval0));

							// low
							__m128i br = _mm_unpacklo_epi8(bref, zero);
							__m128i bd = _mm_unpacklo_epi8(bdiff, zero);
							__m128i gr = _mm_unpacklo_epi8(gref, zero);
							__m128i gd = _mm_unpacklo_epi8(gdiff, zero);
							__m128i rr = _mm_unpacklo_epi8(rref, zero);
							__m128i rd = _mm_unpacklo_epi8(rdiff, zero);

							__m128 _bdiffF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(bd, zero));
							__m128 _gdiffF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(gd, zero));
							__m128 _rdiffF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(rd, zero));
							__m128 _diffFa = _mm_add_ps(_mm_mul_ps(_bdiffF, _bdiffF), _mm_add_ps(_mm_mul_ps(_rdiffF, _rdiffF), _mm_mul_ps(_gdiffF, _gdiffF)));

							__m128 aw = _mm_mul_ps(_diffFa, mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							__m128 _w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							__m128 _bvalF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(br, zero));
							__m128 _gvalF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(gr, zero));
							__m128 _rvalF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(rr, zero));
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval1 = _mm_add_ps(bval1, _bvalF);
							gval1 = _mm_add_ps(gval1, _gvalF);
							rval1 = _mm_add_ps(rval1, _rvalF);
							wval1 = _mm_add_ps(wval1, _w);

							_bdiffF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(bd, zero));
							_gdiffF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(gd, zero));
							_rdiffF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(rd, zero));
							_diffFa = _mm_add_ps(_mm_mul_ps(_bdiffF, _bdiffF), _mm_add_ps(_mm_mul_ps(_rdiffF, _rdiffF), _mm_mul_ps(_gdiffF, _gdiffF)));

							aw = _mm_mul_ps(_diffFa, mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_bvalF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(br, zero));
							_gvalF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(gr, zero));
							_rvalF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(rr, zero));
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval2 = _mm_add_ps(bval2, _bvalF);
							gval2 = _mm_add_ps(gval2, _gvalF);
							rval2 = _mm_add_ps(rval2, _rvalF);
							wval2 = _mm_add_ps(wval2, _w);


							// high
							br = _mm_unpackhi_epi8(bref, zero);
							bd = _mm_unpackhi_epi8(bdiff, zero);
							gr = _mm_unpackhi_epi8(gref, zero);
							gd = _mm_unpackhi_epi8(gdiff, zero);
							rr = _mm_unpackhi_epi8(rref, zero);
							rd = _mm_unpackhi_epi8(rdiff, zero);

							_bdiffF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(bd, zero));
							_gdiffF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(gd, zero));
							_rdiffF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(rd, zero));
							_diffFa = _mm_add_ps(_mm_mul_ps(_bdiffF, _bdiffF), _mm_add_ps(_mm_mul_ps(_rdiffF, _rdiffF), _mm_mul_ps(_gdiffF, _gdiffF)));

							aw = _mm_mul_ps(_diffFa, mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif

							_bvalF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(br, zero));
							_gvalF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(gr, zero));
							_rvalF = _mm_cvtepi32_ps(_mm_unpacklo_epi16(rr, zero));
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval3 = _mm_add_ps(bval3, _bvalF);
							gval3 = _mm_add_ps(gval3, _gvalF);
							rval3 = _mm_add_ps(rval3, _rvalF);
							wval3 = _mm_add_ps(wval3, _w);


							_bdiffF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(bd, zero));
							_gdiffF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(gd, zero));
							_rdiffF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(rd, zero));
							_diffFa = _mm_add_ps(_mm_mul_ps(_bdiffF, _bdiffF), _mm_add_ps(_mm_mul_ps(_rdiffF, _rdiffF), _mm_mul_ps(_gdiffF, _gdiffF)));

							aw = _mm_mul_ps(_diffFa, mcw);
#if __GRF_PREVENTION__
							aw = _mm_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm_exp_ps(aw);
#else
								fmath::exp_ps(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_bvalF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(br, zero));
							_gvalF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(gr, zero));
							_rvalF = _mm_cvtepi32_ps(_mm_unpackhi_epi16(rr, zero));
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval4 = _mm_add_ps(bval4, _bvalF);
							gval4 = _mm_add_ps(gval4, _gvalF);
							rval4 = _mm_add_ps(rval4, _rvalF);
							wval4 = _mm_add_ps(wval4, _w);
						}
						bval1 = _mm_div_ps(bval1, wval1);
						bval2 = _mm_div_ps(bval2, wval2);
						bval3 = _mm_div_ps(bval3, wval3);
						bval4 = _mm_div_ps(bval4, wval4);
						const __m128i a = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)), _mm_packs_epi32(_mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));
						gval1 = _mm_div_ps(gval1, wval1);
						gval2 = _mm_div_ps(gval2, wval2);
						gval3 = _mm_div_ps(gval3, wval3);
						gval4 = _mm_div_ps(gval4, wval4);
						const __m128i b = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)), _mm_packs_epi32(_mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						rval1 = _mm_div_ps(rval1, wval1);
						rval2 = _mm_div_ps(rval2, wval2);
						rval3 = _mm_div_ps(rval3, wval3);
						rval4 = _mm_div_ps(rval4, wval4);
						const __m128i c = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)), _mm_packs_epi32(_mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm_stream_epi8_color(dptrc, a, b, c);
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
						const float w = exp(gauss_range_coeff * ((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0)));
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
	const int *space_ofs;
	float sigma_range;
};


/*************************************************
	using LUT with "set instruction" x 3
*************************************************/
class GaussianRangeFilter_LUT_Setx3_64f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Setx3_64f_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double *_range_weight, const double offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_SSE4_1
		static const long long CV_DECL_ALIGNED(16) v64f_absmask[] = { 0x7fffffffffffffff, 0x7fffffffffffffff };
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(16) buf[4];

			const double* sptr = temp->ptr<double>(range.start + radiusV) + 2 * (radiusH / 2 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					for (; j < size.width; j += 2)//2 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrj = sptr + j;
						const __m128d sval0 = _mm_load_pd(sptrj);

						__m128d tval = _mm_set1_pd(0.f);
						__m128d wval = _mm_set1_pd(0.f);

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128d sref = _mm_loadu_pd((sptrj + *ofs));
							_mm_store_si128((__m128i*)buf, _mm_cvtpd_epi32(_mm_and_pd(_mm_sub_pd(sval0, sref), *(const __m128d*)v64f_absmask)));

							__m128d _w = _mm_set_pd(range_weight[buf[1]], range_weight[buf[0]]);
							sref = _mm_mul_pd(_w, sref);
							tval = _mm_add_pd(tval, sref);
							wval = _mm_add_pd(wval, _w);
						}
						tval = _mm_div_pd(tval, wval);
						_mm_stream_pd((dptr + j), tval);
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
						const double w = range_weight[(uchar)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(16) rbuf[4];
			int CV_DECL_ALIGNED(16) gbuf[4];
			int CV_DECL_ALIGNED(16) bbuf[4];

			const int sstep = 3 * temp->cols;
			const int dstep = dest->cols * 3;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start + 0) + 2 * (radiusH / 2 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 2 * (radiusH / 2 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 2 * (radiusH / 2 + 1);

			double* dptr = dest->ptr<double>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
#if  __GRF_PREVENTION__	
					const __m128d moffset = _mm_set1_pd(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m128d double_min = _mm_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 2)//2 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m128d bval0 = _mm_load_pd(sptrbj);
						const __m128d gval0 = _mm_load_pd(sptrgj);
						const __m128d rval0 = _mm_load_pd(sptrrj);

						__m128d wval = _mm_set1_pd(0.0f);
						__m128d bval = _mm_set1_pd(0.0f);
						__m128d gval = _mm_set1_pd(0.0f);
						__m128d rval = _mm_set1_pd(0.0f);

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128d bref = _mm_load_pd(sptrbj + *ofs);
							__m128d gref = _mm_load_pd(sptrgj + *ofs);
							__m128d rref = _mm_load_pd(sptrrj + *ofs);
							_mm_store_si128((__m128i*)bbuf, _mm_cvtpd_epi32(_mm_and_pd(_mm_sub_pd(bval0, bref), *(const __m128d*)v64f_absmask)));
							_mm_store_si128((__m128i*)gbuf, _mm_cvtpd_epi32(_mm_and_pd(_mm_sub_pd(gval0, gref), *(const __m128d*)v64f_absmask)));
							_mm_store_si128((__m128i*)rbuf, _mm_cvtpd_epi32(_mm_and_pd(_mm_sub_pd(rval0, rref), *(const __m128d*)v64f_absmask)));

#if __GRF_PREVENTION__
							__m128d _w = _mm_mul_pd(
								moffset,
								_mm_set_pd(range_weight[bbuf[1]], range_weight[bbuf[0]])
							);
#else
							__m128d _w = _mm_set_pd(range_weight[bbuf[1]], range_weight[bbuf[0]]);
#endif
							_w = _mm_mul_pd(_w, _mm_set_pd(range_weight[gbuf[1]], range_weight[gbuf[0]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_pd(_w, double_min);
#endif
							_w = _mm_mul_pd(_w, _mm_set_pd(range_weight[rbuf[1]], range_weight[rbuf[0]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_pd(_w, double_min);
#endif
							bref = _mm_mul_pd(_w, bref);
							gref = _mm_mul_pd(_w, gref);
							rref = _mm_mul_pd(_w, rref);

							bval = _mm_add_pd(bval, bref);
							gval = _mm_add_pd(gval, gref);
							rval = _mm_add_pd(rval, rref);
							wval = _mm_add_pd(wval, _w);
						}
						rval = _mm_div_pd(rval, wval);
						gval = _mm_div_pd(gval, wval);
						bval = _mm_div_pd(bval, wval);

						double* dptrc = dptr + 3 * j;
						_mm_stream_pd_color(dptrc, bval, gval, rval);
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
						const double w = range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)(r - r0)];
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
	const int *space_ofs;
	const double *range_weight;
	const double offset;
};

class GaussianRangeFilter_LUT_Setx3_32f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Setx3_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_range_weight, const float offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_SSE4_1
		static const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(16) buf[4];

			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrj = sptr + j;
						const __m128 sval0 = _mm_load_ps(sptrj);

						__m128 tval = _mm_setzero_ps();
						__m128 wval = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128 sref = _mm_loadu_ps((sptrj + *ofs));
							_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(sval0, sref), *(const __m128*)v32f_absmask)));

							__m128 _w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							sref = _mm_mul_ps(_w, sref);
							tval = _mm_add_ps(tval, sref);
							wval = _mm_add_ps(wval, _w);
						}
						tval = _mm_div_ps(tval, wval);
						_mm_stream_ps((dptr + j), tval);
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

						const float w = range_weight[(uchar)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(16) rbuf[4];
			int CV_DECL_ALIGNED(16) gbuf[4];
			int CV_DECL_ALIGNED(16) bbuf[4];

			const int sstep = 3 * temp->cols;
			const int dstep = dest->cols * 3;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

			float* dptr = dest->ptr<float>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
#if  __GRF_PREVENTION__	
					const __m128 moffset = _mm_set1_ps(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m128 float_min = _mm_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m128 bval0 = _mm_load_ps(sptrbj);
						const __m128 gval0 = _mm_load_ps(sptrgj);
						const __m128 rval0 = _mm_load_ps(sptrrj);

						__m128 wval = _mm_setzero_ps();
						__m128 bval = _mm_setzero_ps();
						__m128 gval = _mm_setzero_ps();
						__m128 rval = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128 bref = _mm_load_ps(sptrbj + *ofs);
							__m128 gref = _mm_load_ps(sptrgj + *ofs);
							__m128 rref = _mm_load_ps(sptrrj + *ofs);
							_mm_store_si128((__m128i*)bbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(bval0, bref), *(const __m128*)v32f_absmask)));
							_mm_store_si128((__m128i*)gbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(gval0, gref), *(const __m128*)v32f_absmask)));
							_mm_store_si128((__m128i*)rbuf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(rval0, rref), *(const __m128*)v32f_absmask)));

#if __GRF_PREVENTION__
							__m128 _w = _mm_mul_ps(
								moffset,
								_mm_set_ps(range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]])
							);
#else
							__m128 _w = _mm_set_ps(range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]]);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[gbuf[3]], range_weight[gbuf[2]], range_weight[gbuf[1]], range_weight[gbuf[0]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[rbuf[3]], range_weight[rbuf[2]], range_weight[rbuf[1]], range_weight[rbuf[0]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							bref = _mm_mul_ps(_w, bref);
							gref = _mm_mul_ps(_w, gref);
							rref = _mm_mul_ps(_w, rref);

							bval = _mm_add_ps(bval, bref);
							gval = _mm_add_ps(gval, gref);
							rval = _mm_add_ps(rval, rref);
							wval = _mm_add_ps(wval, _w);
						}
						rval = _mm_div_ps(rval, wval);
						gval = _mm_div_ps(gval, wval);
						bval = _mm_div_ps(bval, wval);

						float* dptrc = dptr + 3 * j;
						_mm_stream_ps_color(dptrc, bval, gval, rval);
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
						const float w = range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)(r - r0)];
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
	const int *space_ofs;
	const float *range_weight;
	const float offset;
};

class GaussianRangeFilter_LUT_Setx3_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Setx3_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_range_weight, const float offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_SSE4_1
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			uchar CV_DECL_ALIGNED(16) buf[16];

			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					static const __m128i zero = _mm_setzero_si128();
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrj = sptr + j;
						const __m128i sval0 = _mm_load_si128((__m128i*)sptrj);

						__m128 tval1 = _mm_setzero_ps();
						__m128 wval1 = _mm_setzero_ps();
						__m128 tval2 = _mm_setzero_ps();
						__m128 wval2 = _mm_setzero_ps();
						__m128 tval3 = _mm_setzero_ps();
						__m128 wval3 = _mm_setzero_ps();
						__m128 tval4 = _mm_setzero_ps();
						__m128 wval4 = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
							_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval0, sref), _mm_subs_epu8(sref, sval0)));

							// lower bit
							__m128i m1 = _mm_unpacklo_epi8(sref, zero);
							__m128i m2 = _mm_unpackhi_epi16(m1, zero);
							m1 = _mm_unpacklo_epi16(m1, zero);

							__m128 _w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							__m128 _valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							tval1 = _mm_add_ps(tval1, _valF);
							wval1 = _mm_add_ps(wval1, _w);


							_w = _mm_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]]);
							_valF = _mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							tval2 = _mm_add_ps(tval2, _valF);
							wval2 = _mm_add_ps(wval2, _w);


							// higher bit
							m1 = _mm_unpackhi_epi8(sref, zero);
							m2 = _mm_unpackhi_epi16(m1, zero);
							m1 = _mm_unpacklo_epi16(m1, zero);

							_w = _mm_set_ps(range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]]);
							_valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							wval3 = _mm_add_ps(wval3, _w);
							tval3 = _mm_add_ps(tval3, _valF);


							_w = _mm_set_ps(range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]]);
							_valF = _mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							wval4 = _mm_add_ps(wval4, _w);
							tval4 = _mm_add_ps(tval4, _valF);
						}
						tval1 = _mm_div_ps(tval1, wval1);
						tval2 = _mm_div_ps(tval2, wval2);
						tval3 = _mm_div_ps(tval3, wval3);
						tval4 = _mm_div_ps(tval4, wval4);
						_mm_stream_si128((__m128i*)(dptr + j), _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)), _mm_packs_epi32(_mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
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
						const float w = range_weight[(uchar)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (uchar)(sum / wsum);
				}
			}
		}
		else
		{
			uchar CV_DECL_ALIGNED(16) rbuf[16];
			uchar CV_DECL_ALIGNED(16) gbuf[16];
			uchar CV_DECL_ALIGNED(16) bbuf[16];

			const int sstep = 3 * temp->cols;
			const int dstep = dest->cols * 3;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					static const __m128i zero = _mm_setzero_si128();
#if  __GRF_PREVENTION__	
					const __m128 moffset = _mm_set1_ps(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m128 float_min = _mm_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m128i bval0 = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval0 = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval0 = _mm_load_si128((__m128i*)(sptrrj));

						__m128 wval1 = _mm_setzero_ps();
						__m128 bval1 = _mm_setzero_ps();
						__m128 gval1 = _mm_setzero_ps();
						__m128 rval1 = _mm_setzero_ps();

						__m128 wval2 = _mm_setzero_ps();
						__m128 bval2 = _mm_setzero_ps();
						__m128 gval2 = _mm_setzero_ps();
						__m128 rval2 = _mm_setzero_ps();

						__m128 wval3 = _mm_setzero_ps();
						__m128 bval3 = _mm_setzero_ps();
						__m128 gval3 = _mm_setzero_ps();
						__m128 rval3 = _mm_setzero_ps();

						__m128 wval4 = _mm_setzero_ps();
						__m128 bval4 = _mm_setzero_ps();
						__m128 gval4 = _mm_setzero_ps();
						__m128 rval4 = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
							const __m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
							const __m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));
							_mm_store_si128((__m128i*)bbuf, _mm_add_epi8(_mm_subs_epu8(bval0, bref), _mm_subs_epu8(bref, bval0)));
							_mm_store_si128((__m128i*)gbuf, _mm_add_epi8(_mm_subs_epu8(gval0, gref), _mm_subs_epu8(gref, gval0)));
							_mm_store_si128((__m128i*)rbuf, _mm_add_epi8(_mm_subs_epu8(rval0, rref), _mm_subs_epu8(rref, rval0)));

							// lower bit
							__m128i b1 = _mm_unpacklo_epi8(bref, zero);
							__m128i b2 = _mm_unpackhi_epi16(b1, zero);
							b1 = _mm_unpacklo_epi16(b1, zero);
							__m128i g1 = _mm_unpacklo_epi8(gref, zero);
							__m128i g2 = _mm_unpackhi_epi16(g1, zero);
							g1 = _mm_unpacklo_epi16(g1, zero);
							__m128i r1 = _mm_unpacklo_epi8(rref, zero);
							__m128i r2 = _mm_unpackhi_epi16(r1, zero);
							r1 = _mm_unpacklo_epi16(r1, zero);

#if __GRF_PREVENTION__
							__m128 _w = _mm_mul_ps(
								moffset,
								_mm_set_ps(range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]])
							);
#else
							__m128 _w = _mm_set_ps(range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]]);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[gbuf[3]], range_weight[gbuf[2]], range_weight[gbuf[1]], range_weight[gbuf[0]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[rbuf[3]], range_weight[rbuf[2]], range_weight[rbuf[1]], range_weight[rbuf[0]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							__m128 _bvalF = _mm_cvtepi32_ps(b1);
							__m128 _gvalF = _mm_cvtepi32_ps(g1);
							__m128 _rvalF = _mm_cvtepi32_ps(r1);
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval1 = _mm_add_ps(bval1, _bvalF);
							gval1 = _mm_add_ps(gval1, _gvalF);
							rval1 = _mm_add_ps(rval1, _rvalF);
							wval1 = _mm_add_ps(wval1, _w);


#if __GRF_PREVENTION__
							_w = _mm_mul_ps(
								moffset,
								_mm_set_ps(range_weight[bbuf[7]], range_weight[bbuf[6]], range_weight[bbuf[5]], range_weight[bbuf[4]])
							);
#else
							_w = _mm_set_ps(range_weight[bbuf[7]], range_weight[bbuf[6]], range_weight[bbuf[5]], range_weight[bbuf[4]]);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[gbuf[7]], range_weight[gbuf[6]], range_weight[gbuf[5]], range_weight[gbuf[4]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[rbuf[7]], range_weight[rbuf[6]], range_weight[rbuf[5]], range_weight[rbuf[4]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_bvalF = _mm_cvtepi32_ps(b2);
							_gvalF = _mm_cvtepi32_ps(g2);
							_rvalF = _mm_cvtepi32_ps(r2);
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval2 = _mm_add_ps(bval2, _bvalF);
							gval2 = _mm_add_ps(gval2, _gvalF);
							rval2 = _mm_add_ps(rval2, _rvalF);
							wval2 = _mm_add_ps(wval2, _w);


							// higher bit
							b1 = _mm_unpackhi_epi8(bref, zero);
							b2 = _mm_unpackhi_epi16(b1, zero);
							b1 = _mm_unpacklo_epi16(b1, zero);
							g1 = _mm_unpackhi_epi8(gref, zero);
							g2 = _mm_unpackhi_epi16(g1, zero);
							g1 = _mm_unpacklo_epi16(g1, zero);
							r1 = _mm_unpackhi_epi8(rref, zero);
							r2 = _mm_unpackhi_epi16(r1, zero);
							r1 = _mm_unpacklo_epi16(r1, zero);

#if __GRF_PREVENTION__
							_w = _mm_mul_ps(
								moffset,
								_mm_set_ps(range_weight[bbuf[11]], range_weight[bbuf[10]], range_weight[bbuf[9]], range_weight[bbuf[8]])
							);
#else
							_w = _mm_set_ps(range_weight[bbuf[11]], range_weight[bbuf[10]], range_weight[bbuf[9]], range_weight[bbuf[8]]);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[gbuf[11]], range_weight[gbuf[10]], range_weight[gbuf[9]], range_weight[gbuf[8]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[rbuf[11]], range_weight[rbuf[10]], range_weight[rbuf[9]], range_weight[rbuf[8]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_bvalF = _mm_cvtepi32_ps(b1);
							_gvalF = _mm_cvtepi32_ps(g1);
							_rvalF = _mm_cvtepi32_ps(r1);
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval3 = _mm_add_ps(bval3, _bvalF);
							gval3 = _mm_add_ps(gval3, _gvalF);
							rval3 = _mm_add_ps(rval3, _rvalF);
							wval3 = _mm_add_ps(wval3, _w);

#if __GRF_PREVENTION__
							_w = _mm_mul_ps(
								moffset,
								_mm_set_ps(range_weight[bbuf[15]], range_weight[bbuf[14]], range_weight[bbuf[13]], range_weight[bbuf[12]])
							);
#else
							_w = _mm_set_ps(range_weight[bbuf[15]], range_weight[bbuf[14]], range_weight[bbuf[13]], range_weight[bbuf[12]]);
#endif
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[gbuf[15]], range_weight[gbuf[14]], range_weight[gbuf[13]], range_weight[gbuf[12]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_w = _mm_mul_ps(_w, _mm_set_ps(range_weight[rbuf[15]], range_weight[rbuf[14]], range_weight[rbuf[13]], range_weight[rbuf[12]]));
#if __GRF_POSTVENTION__
							_w = _mm_max_ps(_w, float_min);
#endif
							_bvalF = _mm_cvtepi32_ps(b2);
							_gvalF = _mm_cvtepi32_ps(g2);
							_rvalF = _mm_cvtepi32_ps(r2);
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval4 = _mm_add_ps(bval4, _bvalF);
							gval4 = _mm_add_ps(gval4, _gvalF);
							rval4 = _mm_add_ps(rval4, _rvalF);
							wval4 = _mm_add_ps(wval4, _w);
						}
						bval1 = _mm_div_ps(bval1, wval1);
						bval2 = _mm_div_ps(bval2, wval2);
						bval3 = _mm_div_ps(bval3, wval3);
						bval4 = _mm_div_ps(bval4, wval4);
						const __m128i a = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)), _mm_packs_epi32(_mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));
						gval1 = _mm_div_ps(gval1, wval1);
						gval2 = _mm_div_ps(gval2, wval2);
						gval3 = _mm_div_ps(gval3, wval3);
						gval4 = _mm_div_ps(gval4, wval4);
						const __m128i b = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)), _mm_packs_epi32(_mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						rval1 = _mm_div_ps(rval1, wval1);
						rval2 = _mm_div_ps(rval2, wval2);
						rval3 = _mm_div_ps(rval3, wval3);
						rval4 = _mm_div_ps(rval4, wval4);
						const __m128i c = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)), _mm_packs_epi32(_mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm_stream_epi8_color(dptrc, a, b, c);
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
						const float w = range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)abs(r - r0)];
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
	const float  *range_weight;
	const float offset;
};

/*************************************************
	using quantized range LUT with "set instruction" x 1
*************************************************/
class GaussianRangeFilter_QuantizationRangeLUT_Setx1_64f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationRangeLUT_Setx1_64f_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_SSE4_1
		static const long long CV_DECL_ALIGNED(16) v64f_absmask[] = { 0x7fffffffffffffff, 0x7fffffffffffffff };
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(16) buf[4];

			const double* sptr = temp->ptr<double>(range.start + radiusV) + 2 * (radiusH / 2 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					for (; j < size.width; j += 2)//2 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrj = sptr + j;
						const __m128d sval0 = _mm_load_pd(sptrj);

						__m128d tval = _mm_set1_pd(0.f);
						__m128d wval = _mm_set1_pd(0.f);

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128d sref = _mm_loadu_pd((sptrj + *ofs));
							_mm_store_si128((__m128i*)buf, _mm_cvtpd_epi32(_mm_and_pd(_mm_sub_pd(sval0, sref), *(const __m128d*)v64f_absmask)));

							__m128d _w = _mm_set_pd(range_weight[buf[1]], range_weight[buf[0]]);
							sref = _mm_mul_pd(_w, sref);
							tval = _mm_add_pd(tval, sref);
							wval = _mm_add_pd(wval, _w);
						}
						tval = _mm_div_pd(tval, wval);
						_mm_stream_pd((dptr + j), tval);
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
						const double w = range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(16) buf[4];

			const int sstep = 3 * temp->cols;
			const int dstep = dest->cols * 3;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start + 0) + 2 * (radiusH / 2 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 2 * (radiusH / 2 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 2 * (radiusH / 2 + 1);

			double* dptr = dest->ptr<double>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					for (; j < size.width; j += 2)//2 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m128d bval0 = _mm_load_pd(sptrbj);
						const __m128d gval0 = _mm_load_pd(sptrgj);
						const __m128d rval0 = _mm_load_pd(sptrrj);

						__m128d wval = _mm_set1_pd(0.0f);
						__m128d bval = _mm_set1_pd(0.0f);
						__m128d gval = _mm_set1_pd(0.0f);
						__m128d rval = _mm_set1_pd(0.0f);

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128d bref = _mm_load_pd(sptrbj + *ofs);
							__m128d gref = _mm_load_pd(sptrgj + *ofs);
							__m128d rref = _mm_load_pd(sptrrj + *ofs);

							const __m128d bdiff = _mm_sub_pd(bval0, bref);
							__m128d gdiff = _mm_sub_pd(gval0, gref);
							__m128d rdiff = _mm_sub_pd(rval0, rref);
							__m128d difft = _mm_mul_pd(bdiff, bdiff);

							gdiff = _mm_mul_pd(gdiff, gdiff);
							difft = _mm_add_pd(difft, gdiff);
							rdiff = _mm_mul_pd(rdiff, rdiff);
							difft = _mm_add_pd(difft, rdiff);

							difft = _mm_sqrt_pd(difft);
							_mm_store_si128((__m128i*)buf, _mm_cvtpd_epi32(difft));

							__m128d _w = _mm_set_pd(range_weight[buf[1]], range_weight[buf[0]]);
							bref = _mm_mul_pd(_w, bref);
							gref = _mm_mul_pd(_w, gref);
							rref = _mm_mul_pd(_w, rref);

							bval = _mm_add_pd(bval, bref);
							gval = _mm_add_pd(gval, gref);
							rval = _mm_add_pd(rval, rref);
							wval = _mm_add_pd(wval, _w);
						}
						rval = _mm_div_pd(rval, wval);
						gval = _mm_div_pd(gval, wval);
						bval = _mm_div_pd(bval, wval);

						double* dptrc = dptr + 3 * j;
						_mm_stream_pd_color(dptrc, bval, gval, rval);
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
						const double w = range_weight[(int)sqrt((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
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
	const double  *range_weight;
};

class GaussianRangeFilter_QuantizationRangeLUT_Setx1_32f_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationRangeLUT_Setx1_32f_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_SSE4_1
		static const int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(16) buf[4];

			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrj = sptr + j;
						const __m128 sval0 = _mm_load_ps(sptrj);

						__m128 tval = _mm_setzero_ps();
						__m128 wval = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128 sref = _mm_loadu_ps((sptrj + *ofs));
							_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(_mm_and_ps(_mm_sub_ps(sval0, sref), *(const __m128*)v32f_absmask)));

							__m128 _w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							sref = _mm_mul_ps(_w, sref);
							tval = _mm_add_ps(tval, sref);
							wval = _mm_add_ps(wval, _w);
						}
						tval = _mm_div_ps(tval, wval);
						_mm_stream_ps((dptr + j), tval);
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
						const float w = range_weight[(int)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(16) buf[4];

			const int sstep = 3 * temp->cols;
			const int dstep = dest->cols * 3;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

			float* dptr = dest->ptr<float>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m128 bval0 = _mm_load_ps(sptrbj);
						const __m128 gval0 = _mm_load_ps(sptrgj);
						const __m128 rval0 = _mm_load_ps(sptrrj);

						__m128 wval = _mm_setzero_ps();
						__m128 bval = _mm_setzero_ps();
						__m128 gval = _mm_setzero_ps();
						__m128 rval = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							__m128 bref = _mm_load_ps(sptrbj + *ofs);
							__m128 gref = _mm_load_ps(sptrgj + *ofs);
							__m128 rref = _mm_load_ps(sptrrj + *ofs);

							const __m128 bdiff = _mm_sub_ps(bval0, bref);
							__m128 gdiff = _mm_sub_ps(gval0, gref);
							__m128 rdiff = _mm_sub_ps(rval0, rref);
							__m128 difft = _mm_mul_ps(bdiff, bdiff);

							gdiff = _mm_mul_ps(gdiff, gdiff);
							difft = _mm_add_ps(difft, gdiff);
							rdiff = _mm_mul_ps(rdiff, rdiff);
							difft = _mm_add_ps(difft, rdiff);

							difft = _mm_rcp_ps(_mm_rsqrt_ps(difft));
							_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(difft));

							__m128 _w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							bref = _mm_mul_ps(_w, bref);
							gref = _mm_mul_ps(_w, gref);
							rref = _mm_mul_ps(_w, rref);

							bval = _mm_add_ps(bval, bref);
							gval = _mm_add_ps(gval, gref);
							rval = _mm_add_ps(rval, rref);
							wval = _mm_add_ps(wval, _w);
						}
						rval = _mm_div_ps(rval, wval);
						gval = _mm_div_ps(gval, wval);
						bval = _mm_div_ps(bval, wval);

						float* dptrc = dptr + 3 * j;
						_mm_stream_ps_color(dptrc, bval, gval, rval);
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
						const float w = range_weight[(int)sqrt((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
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
	const float  *range_weight;
};

class GaussianRangeFilter_QuantizationRangeLUT_Setx1_8u_InvokerSSE4 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationRangeLUT_Setx1_8u_InvokerSSE4(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float *_range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator() (const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_SSE4_1
		const bool haveSSE4 = checkHardwareSupport(CV_CPU_SSE4_1);
#endif
		if (cn == 1)
		{
			uchar CV_DECL_ALIGNED(16) buf[16];

			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 16 * (radiusH / 16 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					static const __m128i zero = _mm_setzero_si128();
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrj = sptr + j;
						const __m128i sval0 = _mm_load_si128((__m128i*)sptrj);

						__m128 tval1 = _mm_setzero_ps();
						__m128 wval1 = _mm_setzero_ps();
						__m128 tval2 = _mm_setzero_ps();
						__m128 wval2 = _mm_setzero_ps();
						__m128 tval3 = _mm_setzero_ps();
						__m128 wval3 = _mm_setzero_ps();
						__m128 tval4 = _mm_setzero_ps();
						__m128 wval4 = _mm_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m128i sref = _mm_loadu_si128((__m128i*)(sptrj + *ofs));
							_mm_store_si128((__m128i*)buf, _mm_add_epi8(_mm_subs_epu8(sval0, sref), _mm_subs_epu8(sref, sval0)));

							// lower bit
							__m128i m1 = _mm_unpacklo_epi8(sref, zero);
							__m128i m2 = _mm_unpackhi_epi16(m1, zero);
							m1 = _mm_unpacklo_epi16(m1, zero);

							__m128 _w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							__m128 _valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							tval1 = _mm_add_ps(tval1, _valF);
							wval1 = _mm_add_ps(wval1, _w);


							_w = _mm_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]]);
							_valF = _mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							tval2 = _mm_add_ps(tval2, _valF);
							wval2 = _mm_add_ps(wval2, _w);


							// higher bit
							m1 = _mm_unpackhi_epi8(sref, zero);
							m2 = _mm_unpackhi_epi16(m1, zero);
							m1 = _mm_unpacklo_epi16(m1, zero);

							_w = _mm_set_ps(range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]]);
							_valF = _mm_cvtepi32_ps(m1);
							_valF = _mm_mul_ps(_w, _valF);
							wval3 = _mm_add_ps(wval3, _w);
							tval3 = _mm_add_ps(tval3, _valF);


							_w = _mm_set_ps(range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]]);
							_valF = _mm_cvtepi32_ps(m2);
							_valF = _mm_mul_ps(_w, _valF);
							wval4 = _mm_add_ps(wval4, _w);
							tval4 = _mm_add_ps(tval4, _valF);
						}
						tval1 = _mm_div_ps(tval1, wval1);
						tval2 = _mm_div_ps(tval2, wval2);
						tval3 = _mm_div_ps(tval3, wval3);
						tval4 = _mm_div_ps(tval4, wval4);
						_mm_stream_si128((__m128i*)(dptr + j), _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(tval1), _mm_cvtps_epi32(tval2)), _mm_packs_epi32(_mm_cvtps_epi32(tval3), _mm_cvtps_epi32(tval4))));
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
						const float w = range_weight[(uchar)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (uchar)(sum / wsum);
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(16) buf[4];

			const int sstep = 3 * temp->cols;
			const int dstep = dest->cols * 3;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 16 * (radiusH / 16 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 16 * (radiusH / 16 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 16 * (radiusH / 16 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_SSE4_1
				if (haveSSE4)
				{
					static const __m128i zero = _mm_setzero_si128();
					for (; j < size.width; j += 16)//16 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m128i bval0_16elem = _mm_load_si128((__m128i*)(sptrbj));
						const __m128i gval0_16elem = _mm_load_si128((__m128i*)(sptrgj));
						const __m128i rval0_16elem = _mm_load_si128((__m128i*)(sptrrj));

						__m128i temp = _mm_unpacklo_epi8(bval0_16elem, zero);
						const __m128 bval0_1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(temp, zero));
						const __m128 bval0_2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(temp, zero));
						temp = _mm_unpackhi_epi8(bval0_16elem, zero);
						const __m128 bval0_3 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(temp, zero));
						const __m128 bval0_4 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(temp, zero));
						temp = _mm_unpacklo_epi8(gval0_16elem, zero);
						const __m128 gval0_1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(temp, zero));
						const __m128 gval0_2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(temp, zero));
						temp = _mm_unpackhi_epi8(gval0_16elem, zero);
						const __m128 gval0_3 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(temp, zero));
						const __m128 gval0_4 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(temp, zero));
						temp = _mm_unpacklo_epi8(rval0_16elem, zero);
						const __m128 rval0_1 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(temp, zero));
						const __m128 rval0_2 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(temp, zero));
						temp = _mm_unpackhi_epi8(rval0_16elem, zero);
						const __m128 rval0_3 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(temp, zero));
						const __m128 rval0_4 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(temp, zero));

						__m128 wval1 = _mm_setzero_ps();
						__m128 bval1 = _mm_setzero_ps();
						__m128 gval1 = _mm_setzero_ps();
						__m128 rval1 = _mm_setzero_ps();

						__m128 wval2 = _mm_setzero_ps();
						__m128 bval2 = _mm_setzero_ps();
						__m128 gval2 = _mm_setzero_ps();
						__m128 rval2 = _mm_setzero_ps();

						__m128 wval3 = _mm_setzero_ps();
						__m128 bval3 = _mm_setzero_ps();
						__m128 gval3 = _mm_setzero_ps();
						__m128 rval3 = _mm_setzero_ps();

						__m128 wval4 = _mm_setzero_ps();
						__m128 bval4 = _mm_setzero_ps();
						__m128 gval4 = _mm_setzero_ps();
						__m128 rval4 = _mm_setzero_ps();


						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m128i bref = _mm_loadu_si128((__m128i*)(sptrbj + *ofs));
							const __m128i gref = _mm_loadu_si128((__m128i*)(sptrgj + *ofs));
							const __m128i rref = _mm_loadu_si128((__m128i*)(sptrrj + *ofs));

							// lower bit
							__m128i b1 = _mm_unpacklo_epi8(bref, zero);
							__m128i b2 = _mm_unpackhi_epi16(b1, zero);
							b1 = _mm_unpacklo_epi16(b1, zero);
							__m128i g1 = _mm_unpacklo_epi8(gref, zero);
							__m128i g2 = _mm_unpackhi_epi16(g1, zero);
							g1 = _mm_unpacklo_epi16(g1, zero);
							__m128i r1 = _mm_unpacklo_epi8(rref, zero);
							__m128i r2 = _mm_unpackhi_epi16(r1, zero);
							r1 = _mm_unpacklo_epi16(r1, zero);

							__m128 _bvalF = _mm_cvtepi32_ps(b1);
							__m128 _gvalF = _mm_cvtepi32_ps(g1);
							__m128 _rvalF = _mm_cvtepi32_ps(r1);

							__m128 bdiff = _mm_sub_ps(bval0_1, _bvalF);
							__m128 gdiff = _mm_sub_ps(gval0_1, _gvalF);
							__m128 rdiff = _mm_sub_ps(rval0_1, _rvalF);

							__m128 difft = _mm_mul_ps(bdiff, bdiff);
							gdiff = _mm_mul_ps(gdiff, gdiff);
							difft = _mm_add_ps(difft, gdiff);
							rdiff = _mm_mul_ps(rdiff, rdiff);
							difft = _mm_add_ps(difft, rdiff);

							difft = _mm_rcp_ps(_mm_rsqrt_ps(difft));
							_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(difft));
							__m128 _w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval1 = _mm_add_ps(bval1, _bvalF);
							gval1 = _mm_add_ps(gval1, _gvalF);
							rval1 = _mm_add_ps(rval1, _rvalF);
							wval1 = _mm_add_ps(wval1, _w);


							_bvalF = _mm_cvtepi32_ps(b2);
							_gvalF = _mm_cvtepi32_ps(g2);
							_rvalF = _mm_cvtepi32_ps(r2);

							bdiff = _mm_sub_ps(bval0_2, _bvalF);
							gdiff = _mm_sub_ps(gval0_2, _gvalF);
							rdiff = _mm_sub_ps(rval0_2, _rvalF);

							difft = _mm_mul_ps(bdiff, bdiff);
							gdiff = _mm_mul_ps(gdiff, gdiff);
							difft = _mm_add_ps(difft, gdiff);
							rdiff = _mm_mul_ps(rdiff, rdiff);
							difft = _mm_add_ps(difft, rdiff);

							difft = _mm_rcp_ps(_mm_rsqrt_ps(difft));
							_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(difft));
							_w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval2 = _mm_add_ps(bval2, _bvalF);
							gval2 = _mm_add_ps(gval2, _gvalF);
							rval2 = _mm_add_ps(rval2, _rvalF);
							wval2 = _mm_add_ps(wval2, _w);


							// higher bit
							b1 = _mm_unpackhi_epi8(bref, zero);
							b2 = _mm_unpackhi_epi16(b1, zero);
							b1 = _mm_unpacklo_epi16(b1, zero);
							g1 = _mm_unpackhi_epi8(gref, zero);
							g2 = _mm_unpackhi_epi16(g1, zero);
							g1 = _mm_unpacklo_epi16(g1, zero);
							r1 = _mm_unpackhi_epi8(rref, zero);
							r2 = _mm_unpackhi_epi16(r1, zero);
							r1 = _mm_unpacklo_epi16(r1, zero);

							_bvalF = _mm_cvtepi32_ps(b1);
							_gvalF = _mm_cvtepi32_ps(g1);
							_rvalF = _mm_cvtepi32_ps(r1);

							bdiff = _mm_sub_ps(bval0_3, _bvalF);
							gdiff = _mm_sub_ps(gval0_3, _gvalF);
							rdiff = _mm_sub_ps(rval0_3, _rvalF);

							difft = _mm_mul_ps(bdiff, bdiff);
							gdiff = _mm_mul_ps(gdiff, gdiff);
							difft = _mm_add_ps(difft, gdiff);
							rdiff = _mm_mul_ps(rdiff, rdiff);
							difft = _mm_add_ps(difft, rdiff);

							difft = _mm_rcp_ps(_mm_rsqrt_ps(difft));
							_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(difft));
							_w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval3 = _mm_add_ps(bval3, _bvalF);
							gval3 = _mm_add_ps(gval3, _gvalF);
							rval3 = _mm_add_ps(rval3, _rvalF);
							wval3 = _mm_add_ps(wval3, _w);


							_bvalF = _mm_cvtepi32_ps(b2);
							_gvalF = _mm_cvtepi32_ps(g2);
							_rvalF = _mm_cvtepi32_ps(r2);

							bdiff = _mm_sub_ps(bval0_4, _bvalF);
							gdiff = _mm_sub_ps(gval0_4, _gvalF);
							rdiff = _mm_sub_ps(rval0_4, _rvalF);

							difft = _mm_mul_ps(bdiff, bdiff);
							gdiff = _mm_mul_ps(gdiff, gdiff);
							difft = _mm_add_ps(difft, gdiff);
							rdiff = _mm_mul_ps(rdiff, rdiff);
							difft = _mm_add_ps(difft, rdiff);

							difft = _mm_rcp_ps(_mm_rsqrt_ps(difft));
							_mm_store_si128((__m128i*)buf, _mm_cvtps_epi32(difft));
							_w = _mm_set_ps(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							_bvalF = _mm_mul_ps(_w, _bvalF);
							_gvalF = _mm_mul_ps(_w, _gvalF);
							_rvalF = _mm_mul_ps(_w, _rvalF);
							bval4 = _mm_add_ps(bval4, _bvalF);
							gval4 = _mm_add_ps(gval4, _gvalF);
							rval4 = _mm_add_ps(rval4, _rvalF);
							wval4 = _mm_add_ps(wval4, _w);
						}
						bval1 = _mm_div_ps(bval1, wval1);
						bval2 = _mm_div_ps(bval2, wval2);
						bval3 = _mm_div_ps(bval3, wval3);
						bval4 = _mm_div_ps(bval4, wval4);
						const __m128i a = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(bval1), _mm_cvtps_epi32(bval2)), _mm_packs_epi32(_mm_cvtps_epi32(bval3), _mm_cvtps_epi32(bval4)));
						gval1 = _mm_div_ps(gval1, wval1);
						gval2 = _mm_div_ps(gval2, wval2);
						gval3 = _mm_div_ps(gval3, wval3);
						gval4 = _mm_div_ps(gval4, wval4);
						const __m128i b = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(gval1), _mm_cvtps_epi32(gval2)), _mm_packs_epi32(_mm_cvtps_epi32(gval3), _mm_cvtps_epi32(gval4)));
						rval1 = _mm_div_ps(rval1, wval1);
						rval2 = _mm_div_ps(rval2, wval2);
						rval3 = _mm_div_ps(rval3, wval3);
						rval4 = _mm_div_ps(rval4, wval4);
						const __m128i c = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(rval1), _mm_cvtps_epi32(rval2)), _mm_packs_epi32(_mm_cvtps_epi32(rval3), _mm_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm_stream_epi8_color(dptrc, a, b, c);
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
						const float w = range_weight[(int)sqrt((b - b0)*(b - b0) + (g - g0)*(g - g0) + (r - r0)*(r - r0))];
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
	const int *space_ofs;
	const float  *range_weight;
};


namespace grf
{
	void gaussianRangeFilter_SSE_64f(const Mat& src, Mat& dst, Size kernelSize, double sigma_range, const int borderType, const bool isRectangle, const WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0) { src.copyTo(dst); return; }
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_64FC1 || src.type() == CV_64FC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
			sigma_range = 1;

		const double gauss_range_coeff = -0.5 / (sigma_range*sigma_range);

		const int radiusH = kernelSize.width >> 1;
		const int radiusV = kernelSize.height >> 1;

		Mat temp;

		const int dpad = (2 - src.cols % 2) % 2;
		int spad = dpad + (2 - (2 * radiusH) % 2) % 2;
		if (spad < 2) spad += 2;
		const int lpad = 2 * (radiusH / 2 + 1) - radiusH;
		const int rpad = spad - lpad;
		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleaveSSE(temp2, temp);
		}

		vector<int> _space_ofs(kernelSize.area());
		int* space_ofs = &_space_ofs[0];
		int maxk = 0;
		setSpaceKernel(space_ofs, maxk, radiusH, radiusV, temp.cols * cn, isRectangle);

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			// initialize space-related gaussianRange filter coefficients
			const GaussianRangeFilter_EXP_64f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 256;
			vector<double> _range_weight(value_range);
			double* range_weight = &_range_weight[0];
#if __GRF_PREVENTION__
			const double max_digits = floor(log2(DBL_MAX / (255.0*kernelSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - cn) / cn);
			const double bias = pow(2, log2(DBL_MAX / (255.0*kernelSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#else
			const double bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_range_coeff;
#if __GRF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __GRF_POSTVENTION__
				range_weight[i] = max(range_weight[i], DBL_MIN);
#endif
			}

			const GaussianRangeFilter_LUT_Setx3_64f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 442;	//ceil(sqrt(255^2*3))
			vector<double> _range_weight(value_range);
			double* range_weight = &_range_weight[0];

			// initialize color-related gaussianRange filter coefficients
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_range_coeff;
#if __GRF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
				range_weight[i] = exp(aw);
#if __GRF_POSTVENTION__
				range_weight[i] = max(range_weight[i], DBL_MIN);
#endif
			}

			const GaussianRangeFilter_QuantizationRangeLUT_Setx1_64f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		default:
			break;
		}
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void gaussianRangeFilter_SSE_32f(const Mat& src, Mat& dst, Size kernelSize, float sigma_range, const int borderType, const bool isRectangle, const WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0) { src.copyTo(dst); return; }
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
			sigma_range = 1;

		const float gauss_range_coeff = -0.5f / (sigma_range*sigma_range);

		const int radiusH = kernelSize.width >> 1;
		const int radiusV = kernelSize.height >> 1;

		Mat temp;

		const int dpad = (4 - src.cols % 4) % 4;
		int spad = dpad + (4 - (2 * radiusH) % 4) % 4;
		if (spad < 4) spad += 4;
		const int lpad = 4 * (radiusH / 4 + 1) - radiusH;
		const int rpad = spad - lpad;
		if (cn == 1)
		{
			copyMakeBorder(src, temp, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp2;
			copyMakeBorder(src, temp2, radiusV, radiusV, radiusH + lpad, radiusH + rpad, borderType);
			splitBGRLineInterleaveSSE(temp2, temp);
		}


		vector<int> _space_ofs(kernelSize.area());
		int* space_ofs = &_space_ofs[0];
		int maxk = 0;
		setSpaceKernel(space_ofs, maxk, radiusH, radiusV, temp.cols * cn, isRectangle);

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			const GaussianRangeFilter_EXP_32f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 256;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];
#if __GRF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - cn) / cn);
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __GRF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __GRF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}
			const GaussianRangeFilter_LUT_Setx3_32f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 442;	//ceil(sqrt(255^2*3))
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];
			// initialize color-related gaussianRange filter coefficients
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __GRF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				range_weight[i] = exp(aw);
#if __GRF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const GaussianRangeFilter_QuantizationRangeLUT_Setx1_32f_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		default:
			break;
		}
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void gaussianRangeFilter_SSE_8u(const Mat& src, Mat& dst, Size kernelSize, float sigma_range, const int borderType, const bool isRectangle, const WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0) { src.copyTo(dst); return; }
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
			sigma_range = 1;

		const float gauss_range_coeff = -0.5 / (sigma_range*sigma_range);

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
			splitBGRLineInterleaveSSE(temp2, temp);
		}

		vector<int> _space_ofs(kernelSize.area());
		int* space_ofs = &_space_ofs[0];
		int maxk = 0;
		setSpaceKernel(space_ofs, maxk, radiusH, radiusV, temp.cols * cn, isRectangle);

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), src.type());
		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			// initialize space-related gaussianRange filter coefficients
			const GaussianRangeFilter_EXP_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients

			const int value_range = 256;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];
#if __GRF_PREVENTION__
			const float max_digits = floor(log2(FLT_MAX / (255.f*kernelSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - cn) / cn);
			const float bias = pow(2, log2(FLT_MAX / (255.f*kernelSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __GRF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				range_weight[i] = exp(aw);
#if __GRF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const GaussianRangeFilter_LUT_Setx3_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
			const int value_range = 442;	//ceil(sqrt(255^2*3))
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			// initialize color-related gaussianRange filter coefficients
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_range_coeff;
#if __GRF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				range_weight[i] = exp(aw);
#if __GRF_POSTVENTION__
				range_weight[i] = max(range_weight[i], FLT_MIN);
#endif
			}

			const GaussianRangeFilter_QuantizationRangeLUT_Setx1_8u_InvokerSSE4 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		default:
			break;
		}

		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void gaussianRangeFilter_SSE(cv::InputArray src_, cv::OutputArray dest, const cv::Size kernelSize, const double sigma_range, const int kernel_type, const int borderType, const WEIGHT_MODE weightingMethod)
	{
		if (dest.empty() || dest.type() != src_.type() || dest.size() != src_.size()) dest.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dst = dest.getMat();

		switch (kernel_type)
		{
		case FILTER_RECTANGLE:
		{
			if (src.depth() == CV_8U)
			{
				gaussianRangeFilter_SSE_8u(src, dst, kernelSize, sigma_range, borderType, true, weightingMethod);
			}
			else if (src.depth() == CV_32F)
			{
				gaussianRangeFilter_SSE_32f(src, dst, kernelSize, sigma_range, borderType, true, weightingMethod);
			}
			else if (src.depth() == CV_64F)
			{
				gaussianRangeFilter_SSE_64f(src, dst, kernelSize, sigma_range, borderType, true, weightingMethod);
			}
			break;
		}
		case FILTER_CIRCLE:
		{
			if (src.depth() == CV_8U)
			{
				gaussianRangeFilter_SSE_8u(src, dst, kernelSize, sigma_range, borderType, false, weightingMethod);
			}
			else if (src.depth() == CV_32F)
			{
				gaussianRangeFilter_SSE_32f(src, dst, kernelSize, sigma_range, borderType, false, weightingMethod);
			}
			else if (src.depth() == CV_64F)
			{
				gaussianRangeFilter_SSE_64f(src, dst, kernelSize, sigma_range, borderType, false, weightingMethod);
			}
			break;
		}
		case FILTER_SEPARABLE:
		default:
			break;
		}
	}
}
