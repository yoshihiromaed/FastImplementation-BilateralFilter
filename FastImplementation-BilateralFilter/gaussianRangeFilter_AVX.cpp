#include "color.h"
#include "filter.h"
#include "fmath.hpp"
#include "simd_util.h"

using namespace std;
using namespace cv;


/*************************************************
	using exp function
*************************************************/
class GaussianRangeFilter_EXP_64f_InvokerAVX : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_EXP_64f_InvokerAVX(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double _sigma_range) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), sigma_range(_sigma_range)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const double gauss_range_coeff = -0.5 / (sigma_range * sigma_range);

#if CV_AVX
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX);
#endif
		if (cn == 1)
		{
			const double* sptr = temp->ptr<double>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					const __m256d mcw = _mm256_set1_pd(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m256d exp_arg_min = _mm256_set1_pd(EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
#if __GRF_POSTVENTION__
					static const __m256d double_min = _mm256_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrj = sptr + j;
						const __m256d sval0 = _mm256_load_pd(sptrj);

						__m256d tval = _mm256_setzero_pd();
						__m256d wval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));

							const __m256d diff = _mm256_sub_pd(sval0, sref);
							const __m256d difft = _mm256_mul_pd(diff, diff);

							__m256d aw = _mm256_mul_pd(difft, mcw);

#if __GRF_PREVENTION__
							aw = _mm256_max_pd(aw, exp_arg_min);
#endif

							__m256d _w =
#if __USE_INTEL_EXP__
								_mm256_exp_pd(aw);
#else
								//fmath::exp_pd256(aw); //TODO: exp_pd
								_mm256_set1_pd(1);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_pd(sref, _w, tval);
#else
							const __m256d sref_mul = _mm256_mul_pd(sref, _w);
							tval = _mm256_add_pd(tval, sref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						tval = _mm256_div_pd(tval, wval);
						_mm256_stream_pd((dptr + j), tval);
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
						const double w = exp(gauss_range_coeff * ((val - val0) * (val - val0)));
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
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start) + 4 * (radiusH / 4 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					const __m256d mcw = _mm256_set1_pd(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m256d exp_arg_min = _mm256_set1_pd(EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
#if __GRF_POSTVENTION__
					static const __m256d double_min = _mm256_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrrj = sptrr + j;
						const double* sptrgj = sptrg + j;
						const double* sptrbj = sptrb + j;

						const __m256d bval0 = _mm256_load_pd((sptrbj));
						const __m256d gval0 = _mm256_load_pd((sptrgj));
						const __m256d rval0 = _mm256_load_pd((sptrrj));

						__m256d wval = _mm256_setzero_pd();
						__m256d bval = _mm256_setzero_pd();
						__m256d gval = _mm256_setzero_pd();
						__m256d rval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d bref = _mm256_loadu_pd((sptrbj + *ofs));
							const __m256d gref = _mm256_loadu_pd((sptrgj + *ofs));
							const __m256d rref = _mm256_loadu_pd((sptrrj + *ofs));

							const __m256d bdiff = _mm256_sub_pd(bval0, bref);
							const __m256d gdiff = _mm256_sub_pd(gval0, gref);
							const __m256d rdiff = _mm256_sub_pd(rval0, rref);

							__m256d difft = _mm256_mul_pd(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm256_fmadd_pd(rdiff, rdiff, difft);
#else
							const __m256d gdiff_mul = _mm256_mul_pd(gdiff, gdiff);
							difft = _mm256_add_pd(difft, gdiff_mul);
							const __m256d rdiff_mul = _mm256_mul_pd(rdiff, rdiff);
							difft = _mm256_add_pd(difft, rdiff_mul);
#endif
							__m256d aw = _mm256_mul_pd(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_pd(aw, exp_arg_min);
#endif

							__m256d _w =
#if __USE_INTEL_EXP__
								_mm256_exp_pd(aw);
#else
								//fmath::exp_pd256(aw); //TODO: exp_pd
								_mm256_set1_pd(1);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_pd(_w, double_min);
#endif


#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_pd(bref, _w, bval);
							gval = _mm256_fmadd_pd(gref, _w, gval);
							rval = _mm256_fmadd_pd(rref, _w, rval);
#else
							const __m256d bref_mul = _mm256_mul_pd(bref, _w);
							bval = _mm256_add_pd(bval, bref_mul);
							const __m256d gref_mul = _mm256_mul_pd(gref, _w);
							gval = _mm256_add_pd(gval, gref_mul);
							const __m256d rref_mul = _mm256_mul_pd(rref, _w);
							rval = _mm256_add_pd(rval, rref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						bval = _mm256_div_pd(bval, wval);
						gval = _mm256_div_pd(gval, wval);
						rval = _mm256_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm256_stream_pd_color(dptrc, bval, gval, rval);
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
						const double w = exp(gauss_range_coeff * ((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0)));
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int *space_ofs;
	double sigma_range;
};

class GaussianRangeFilter_EXP_32f_InvokerAVX : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_EXP_32f_InvokerAVX(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float _sigma_range) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), sigma_range(_sigma_range)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float gauss_range_coeff = -0.5f / (sigma_range * sigma_range);

#if CV_AVX
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX);
#endif
		if (cn == 1)
		{
			float* sptr = const_cast<float*>(temp->ptr<float>(range.start + radiusV)) + 8 * (radiusH / 8 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					const __m256 mcw = _mm256_set1_ps(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m256 exp_arg_min = _mm256_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __GRF_POSTVENTION__
					static const __m256 float_min = _mm256_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrj = sptr + j;
						const __m256 sval0 = _mm256_load_ps(sptrj);

						__m256 tval = _mm256_setzero_ps();
						__m256 wval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 sref = _mm256_loadu_ps((sptrj + *ofs));

							const __m256 diff = _mm256_sub_ps(sval0, sref);
							const __m256 difft = _mm256_mul_ps(diff, diff);

							__m256 aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif
							__m256 _w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_ps(sref, _w, tval);
#else
							const __m256 sref_mul = _mm256_mul_ps(sref, _w);
							tval = _mm256_add_ps(tval, sref_mul);
#endif

							wval = _mm256_add_ps(wval, _w);
						}
						tval = _mm256_div_ps(tval, wval);
						_mm256_stream_ps((dptr + j), tval);
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
						const float w = exp(gauss_range_coeff * ((val - val0) * (val - val0)));
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
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					const __m256 mcw = _mm256_set1_ps(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m256 exp_arg_min = _mm256_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __GRF_POSTVENTION__
					static const __m256 float_min = _mm256_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrrj = sptrr + j;
						const float* sptrgj = sptrg + j;
						const float* sptrbj = sptrb + j;

						const __m256 bval0 = _mm256_load_ps((sptrbj));
						const __m256 gval0 = _mm256_load_ps((sptrgj));
						const __m256 rval0 = _mm256_load_ps((sptrrj));

						__m256 wval = _mm256_setzero_ps();
						__m256 bval = _mm256_setzero_ps();
						__m256 gval = _mm256_setzero_ps();
						__m256 rval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 bref = _mm256_loadu_ps((sptrbj + *ofs));
							const __m256 gref = _mm256_loadu_ps((sptrgj + *ofs));
							const __m256 rref = _mm256_loadu_ps((sptrrj + *ofs));

							const __m256 bdiff = _mm256_sub_ps(bval0, bref);
							const __m256 gdiff = _mm256_sub_ps(gval0, gref);
							const __m256 rdiff = _mm256_sub_ps(rval0, rref);

							__m256 difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							const __m256 gdiff_mul = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff_mul);
							const __m256 rdiff_mul = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff_mul);
#endif
							__m256 aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif
							__m256 _w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_ps(bref, _w, bval);
							gval = _mm256_fmadd_ps(gref, _w, gval);
							rval = _mm256_fmadd_ps(rref, _w, rval);
#else
							const __m256 bref_mul = _mm256_mul_ps(bref, _w);
							bval = _mm256_add_ps(bval, bref_mul);
							const __m256 gref_mul = _mm256_mul_ps(gref, _w);
							gval = _mm256_add_ps(gval, gref_mul);
							const __m256 rref_mul = _mm256_mul_ps(rref, _w);
							rval = _mm256_add_ps(rval, rref_mul);
#endif
							wval = _mm256_add_ps(wval, _w);
						}
						bval = _mm256_div_ps(bval, wval);
						gval = _mm256_div_ps(gval, wval);
						rval = _mm256_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm256_stream_ps_color(dptrc, bval, gval, rval);
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
						const float w = exp(gauss_range_coeff * ((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0)));
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int *space_ofs;
	float sigma_range;
};

class GaussianRangeFilter_EXP_8u_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_EXP_8u_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float _sigma_range) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), sigma_range(_sigma_range)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
		const float gauss_range_coeff = -0.5f / (sigma_range * sigma_range);

#if CV_AVX2
		const bool haveAVX2 = checkHardwareSupport(CV_CPU_AVX2);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 32 * (radiusH / 32 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
					const __m256 mcw = _mm256_set1_ps(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m256 exp_arg_min = _mm256_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __GRF_POSTVENTION__
					static const __m256 float_min = _mm256_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrj = sptr + j;
						const __m256i sval0_32elem = _mm256_load_si256((__m256i*)sptrj);

						__m256 tval1 = _mm256_setzero_ps();
						__m256 wval1 = _mm256_setzero_ps();
						__m256 tval2 = _mm256_setzero_ps();
						__m256 wval2 = _mm256_setzero_ps();
						__m256 tval3 = _mm256_setzero_ps();
						__m256 wval3 = _mm256_setzero_ps();
						__m256 tval4 = _mm256_setzero_ps();
						__m256 wval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i sref_32elem = _mm256_loadu_si256((__m256i*)(sptrj + *ofs));
							const __m256i diff_32elem = _mm256_add_epi8(_mm256_subs_epu8(sval0_32elem, sref_32elem), _mm256_subs_epu8(sref_32elem, sval0_32elem));

							// lower bit
							__m256i sref_16elem = _mm256_unpacklo_epi8(sref_32elem, zero);
							__m256i diff_16elem = _mm256_unpacklo_epi8(diff_32elem, zero);

							__m256 diff = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(diff_16elem, zero));
							__m256 difft = _mm256_mul_ps(diff, diff);
							__m256 aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif
							__m256 _w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

							__m256 sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm256_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval1 = _mm256_add_ps(tval1, sref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


							diff = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(diff_16elem, zero));
							difft = _mm256_mul_ps(diff, diff);
							aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif

							_w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm256_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval2 = _mm256_add_ps(tval2, sref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// higher bit
							sref_16elem = _mm256_unpackhi_epi8(sref_32elem, zero);
							diff_16elem = _mm256_unpackhi_epi8(diff_32elem, zero);

							diff = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(diff_16elem, zero));
							difft = _mm256_mul_ps(diff, diff);
							aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif

							_w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

							sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm256_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval3 = _mm256_add_ps(tval3, sref);
#endif

							wval3 = _mm256_add_ps(wval3, _w);

							diff = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(diff_16elem, zero));
							difft = _mm256_mul_ps(diff, diff);
							aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif

							_w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm256_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval4 = _mm256_add_ps(tval4, sref);
#endif

							wval4 = _mm256_add_ps(wval4, _w);
						}
						tval1 = _mm256_div_ps(tval1, wval1);
						tval2 = _mm256_div_ps(tval2, wval2);
						tval3 = _mm256_div_ps(tval3, wval3);
						tval4 = _mm256_div_ps(tval4, wval4);
						_mm256_stream_si256((__m256i*)(dptr + j), _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(tval1), _mm256_cvtps_epi32(tval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(tval3), _mm256_cvtps_epi32(tval4))));
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

						const float w = exp(gauss_range_coeff * ((val - val0) * (val - val0)));
						sum += val * w;
						wsum += w;
					}
					dptr[j] = (char)(sum / wsum);
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 32 * (radiusH / 32 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 32 * (radiusH / 32 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 32 * (radiusH / 32 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
					const __m256 mcw = _mm256_set1_ps(gauss_range_coeff);
#if __GRF_PREVENTION__
					static const __m256 exp_arg_min = _mm256_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __GRF_POSTVENTION__
					static const __m256 float_min = _mm256_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m256i bval0_32elem = _mm256_load_si256((__m256i*)(sptrbj));
						const __m256i gval0_32elem = _mm256_load_si256((__m256i*)(sptrgj));
						const __m256i rval0_32elem = _mm256_load_si256((__m256i*)(sptrrj));

						__m256 wval1 = _mm256_setzero_ps();
						__m256 bval1 = _mm256_setzero_ps();
						__m256 gval1 = _mm256_setzero_ps();
						__m256 rval1 = _mm256_setzero_ps();

						__m256 wval2 = _mm256_setzero_ps();
						__m256 bval2 = _mm256_setzero_ps();
						__m256 gval2 = _mm256_setzero_ps();
						__m256 rval2 = _mm256_setzero_ps();

						__m256 wval3 = _mm256_setzero_ps();
						__m256 bval3 = _mm256_setzero_ps();
						__m256 gval3 = _mm256_setzero_ps();
						__m256 rval3 = _mm256_setzero_ps();

						__m256 wval4 = _mm256_setzero_ps();
						__m256 bval4 = _mm256_setzero_ps();
						__m256 gval4 = _mm256_setzero_ps();
						__m256 rval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i bref_32elem = _mm256_loadu_si256((__m256i*)(sptrbj + *ofs));
							const __m256i gref_32elem = _mm256_loadu_si256((__m256i*)(sptrgj + *ofs));
							const __m256i rref_32elem = _mm256_loadu_si256((__m256i*)(sptrrj + *ofs));

							const __m256i bdiff_32elem = _mm256_add_epi8(_mm256_subs_epu8(bval0_32elem, bref_32elem), _mm256_subs_epu8(bref_32elem, bval0_32elem));
							const __m256i gdiff_32elem = _mm256_add_epi8(_mm256_subs_epu8(gval0_32elem, gref_32elem), _mm256_subs_epu8(gref_32elem, gval0_32elem));
							const __m256i rdiff_32elem = _mm256_add_epi8(_mm256_subs_epu8(rval0_32elem, rref_32elem), _mm256_subs_epu8(rref_32elem, rval0_32elem));

							// lower bit
							__m256i bref_16elem = _mm256_unpacklo_epi8(bref_32elem, zero);
							__m256i gref_16elem = _mm256_unpacklo_epi8(gref_32elem, zero);
							__m256i rref_16elem = _mm256_unpacklo_epi8(rref_32elem, zero);
							__m256i bdiff_16elem = _mm256_unpacklo_epi8(bdiff_32elem, zero);
							__m256i gdiff_16elem = _mm256_unpacklo_epi8(gdiff_32elem, zero);
							__m256i rdiff_16elem = _mm256_unpacklo_epi8(rdiff_32elem, zero);

							__m256 bdiff = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bdiff_16elem, zero));
							__m256 gdiff = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gdiff_16elem, zero));
							__m256 rdiff = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rdiff_16elem, zero));
							__m256 difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, gdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							__m256 aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif

							__m256 _w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

							__m256 bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							__m256 gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							__m256 rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm256_fmadd_ps(bref, _w, bval1);
							gval1 = _mm256_fmadd_ps(gref, _w, gval1);
							rval1 = _mm256_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm256_mul_ps(bref, _w);
							gref = _mm256_mul_ps(gref, _w);
							rref = _mm256_mul_ps(rref, _w);
							bval1 = _mm256_add_ps(bval1, bref);
							gval1 = _mm256_add_ps(gval1, gref);
							rval1 = _mm256_add_ps(rval1, rref);
#endif

							wval1 = _mm256_add_ps(wval1, _w);

							bdiff = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bdiff_16elem, zero));
							gdiff = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gdiff_16elem, zero));
							rdiff = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rdiff_16elem, zero));

							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, gdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm256_fmadd_ps(bref, _w, bval2);
							gval2 = _mm256_fmadd_ps(gref, _w, gval2);
							rval2 = _mm256_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm256_mul_ps(bref, _w);
							gref = _mm256_mul_ps(gref, _w);
							rref = _mm256_mul_ps(rref, _w);
							bval2 = _mm256_add_ps(bval2, bref);
							gval2 = _mm256_add_ps(gval2, gref);
							rval2 = _mm256_add_ps(rval2, rref);
#endif

							wval2 = _mm256_add_ps(wval2, _w);

							// high
							bref_16elem = _mm256_unpackhi_epi8(bref_32elem, zero);
							gref_16elem = _mm256_unpackhi_epi8(gref_32elem, zero);
							rref_16elem = _mm256_unpackhi_epi8(rref_32elem, zero);
							bdiff_16elem = _mm256_unpackhi_epi8(bdiff_32elem, zero);
							gdiff_16elem = _mm256_unpackhi_epi8(gdiff_32elem, zero);
							rdiff_16elem = _mm256_unpackhi_epi8(rdiff_32elem, zero);

							bdiff = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bdiff_16elem, zero));
							gdiff = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gdiff_16elem, zero));
							rdiff = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rdiff_16elem, zero));
							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, gdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

							bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm256_fmadd_ps(bref, _w, bval3);
							gval3 = _mm256_fmadd_ps(gref, _w, gval3);
							rval3 = _mm256_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm256_mul_ps(bref, _w);
							gref = _mm256_mul_ps(gref, _w);
							rref = _mm256_mul_ps(rref, _w);
							bval3 = _mm256_add_ps(bval3, bref);
							gval3 = _mm256_add_ps(gval3, gref);
							rval3 = _mm256_add_ps(rval3, rref);
#endif

							wval3 = _mm256_add_ps(wval3, _w);

							bdiff = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bdiff_16elem, zero));
							gdiff = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gdiff_16elem, zero));
							rdiff = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rdiff_16elem, zero));

							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, gdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							aw = _mm256_mul_ps(difft, mcw);
#if __GRF_PREVENTION__
							aw = _mm256_max_ps(aw, exp_arg_min);
#endif
							_w =
#if __USE_INTEL_EXP__
								_mm256_exp_ps(aw);
#else
								fmath::exp_ps256(aw);
#endif
#if __GRF_POSTVENTION__
							_w = _mm256_max_ps(_w, float_min);
#endif

							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm256_fmadd_ps(bref, _w, bval4);
							gval4 = _mm256_fmadd_ps(gref, _w, gval4);
							rval4 = _mm256_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm256_mul_ps(bref, _w);
							gref = _mm256_mul_ps(gref, _w);
							rref = _mm256_mul_ps(rref, _w);
							bval4 = _mm256_add_ps(bval4, bref);
							gval4 = _mm256_add_ps(gval4, gref);
							rval4 = _mm256_add_ps(rval4, rref);
#endif

							wval4 = _mm256_add_ps(wval4, _w);
						}
						bval1 = _mm256_div_ps(bval1, wval1);
						bval2 = _mm256_div_ps(bval2, wval2);
						bval3 = _mm256_div_ps(bval3, wval3);
						bval4 = _mm256_div_ps(bval4, wval4);
						const __m256i a = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(bval1), _mm256_cvtps_epi32(bval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(bval3), _mm256_cvtps_epi32(bval4)));
						gval1 = _mm256_div_ps(gval1, wval1);
						gval2 = _mm256_div_ps(gval2, wval2);
						gval3 = _mm256_div_ps(gval3, wval3);
						gval4 = _mm256_div_ps(gval4, wval4);
						const __m256i b = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(gval1), _mm256_cvtps_epi32(gval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(gval3), _mm256_cvtps_epi32(gval4)));
						rval1 = _mm256_div_ps(rval1, wval1);
						rval2 = _mm256_div_ps(rval2, wval2);
						rval3 = _mm256_div_ps(rval3, wval3);
						rval4 = _mm256_div_ps(rval4, wval4);
						const __m256i c = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(rval1), _mm256_cvtps_epi32(rval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(rval3), _mm256_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm256_stream_epi8_color(dptrc, a, b, c);
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
						const float w = exp(gauss_range_coeff * ((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0)));
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j + 0] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}

private:
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int *space_ofs;
	float sigma_range;
};


/*************************************************
	using LUT with "set instruction" x 3
*************************************************/
class GaussianRangeFilter_LUT_Setx3_64f_InvokerAVX : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Setx3_64f_InvokerAVX(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double* _range_weight, const double offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX || CV_AVX2
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX);
		static const long long CV_DECL_ALIGNED(32) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff,
		};
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(32) buf[4];

			const double* sptr = temp->ptr<double>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrj = sptr + j;
						const __m256d sval0 = _mm256_load_pd(sptrj);

						__m256d tval = _mm256_setzero_pd();
						__m256d wval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
							_mm_store_si128((__m128i*)buf, _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(sval0, sref), *(const __m256d*)v64f_absmask)));

							__m256d _w = _mm256_set_pd(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);

#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_pd(sref, _w, tval);
#else
							const __m256d sref_mul = _mm256_mul_pd(sref, _w);
							tval = _mm256_add_pd(tval, sref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						tval = _mm256_div_pd(tval, wval);
						_mm256_stream_pd((dptr + j), tval);
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
						const double w = range_weight[(char)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(32) rbuf[4];
			int CV_DECL_ALIGNED(32) gbuf[4];
			int CV_DECL_ALIGNED(32) bbuf[4];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start + 0) + 4 * (radiusH / 4 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
#if  __GRF_PREVENTION__	
					const __m256d moffset = _mm256_set1_pd(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m256d double_min = _mm256_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m256d bval0 = _mm256_load_pd(sptrbj);
						const __m256d gval0 = _mm256_load_pd(sptrgj);
						const __m256d rval0 = _mm256_load_pd(sptrrj);

						__m256d wval = _mm256_setzero_pd();
						__m256d bval = _mm256_setzero_pd();
						__m256d gval = _mm256_setzero_pd();
						__m256d rval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d bref = _mm256_load_pd(sptrbj + *ofs);
							const __m256d gref = _mm256_load_pd(sptrgj + *ofs);
							const __m256d rref = _mm256_load_pd(sptrrj + *ofs);
							_mm_store_si128((__m128i*)bbuf, _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(bval0, bref), *(const __m256d*)v64f_absmask)));
							_mm_store_si128((__m128i*)gbuf, _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(gval0, gref), *(const __m256d*)v64f_absmask)));
							_mm_store_si128((__m128i*)rbuf, _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(rval0, rref), *(const __m256d*)v64f_absmask)));

#if __GRF_PREVENTION__
							__m256d _w = _mm256_mul_pd(
								moffset,
								_mm256_set_pd(range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]])
							);
#else
							__m256d _w = _mm256_set_pd(range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]]);
#endif
							_w = _mm256_mul_pd(_w, _mm256_set_pd(range_weight[gbuf[3]], range_weight[gbuf[2]], range_weight[gbuf[1]], range_weight[gbuf[0]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_pd(_w, double_min);
#endif
							_w = _mm256_mul_pd(_w, _mm256_set_pd(range_weight[rbuf[3]], range_weight[rbuf[2]], range_weight[rbuf[1]], range_weight[rbuf[0]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_pd(bref, _w, bval);
							gval = _mm256_fmadd_pd(gref, _w, gval);
							rval = _mm256_fmadd_pd(rref, _w, rval);
#else
							const __m256d bref_mul = _mm256_mul_pd(bref, _w);
							bval = _mm256_add_pd(bval, bref_mul);
							const __m256d gref_mul = _mm256_mul_pd(gref, _w);
							gval = _mm256_add_pd(gval, gref_mul);
							const __m256d rref_mul = _mm256_mul_pd(rref, _w);
							rval = _mm256_add_pd(rval, rref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						bval = _mm256_div_pd(bval, wval);
						gval = _mm256_div_pd(gval, wval);
						rval = _mm256_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm256_stream_pd_color(dptrc, bval, gval, rval);
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
						const double w = range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)abs(r - r0)];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j + 0] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}

private:
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *range_weight;
	const double offset;
};

class GaussianRangeFilter_LUT_Setx3_32f_InvokerAVX : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Setx3_32f_InvokerAVX(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float* _range_weight, const float offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX || CV_AVX2
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX);
		static const int CV_DECL_ALIGNED(32) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(32) buf[8];

			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrj = sptr + j;
						const __m256 sval0 = _mm256_load_ps(sptrj);

						__m256 tval = _mm256_setzero_ps();
						__m256 wval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 sref = _mm256_loadu_ps((sptrj + *ofs));
							_mm256_store_si256((__m256i*)buf, _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(sval0, sref), *(const __m256*)v32f_absmask)));

							__m256 _w = _mm256_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_ps(sref, _w, tval);
#else
							const __m256 sref_mul = _mm256_mul_ps(sref, _w);
							tval = _mm256_add_ps(tval, sref_mul);
#endif
							wval = _mm256_add_ps(wval, _w);
						}
						tval = _mm256_div_ps(tval, wval);
						_mm256_stream_ps((dptr + j), tval);
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
						const float w = range_weight[(char)abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
					dptr[j] = sum / wsum;
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(32) rbuf[8];
			int CV_DECL_ALIGNED(32) gbuf[8];
			int CV_DECL_ALIGNED(32) bbuf[8];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start + 0) + 8 * (radiusH / 8 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			float* dptr = dest->ptr<float>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
#if  __GRF_PREVENTION__	
					const __m256 moffset = _mm256_set1_ps(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m256 float_min = _mm256_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m256 bval0 = _mm256_load_ps(sptrbj);
						const __m256 gval0 = _mm256_load_ps(sptrgj);
						const __m256 rval0 = _mm256_load_ps(sptrrj);

						__m256 wval = _mm256_setzero_ps();
						__m256 bval = _mm256_setzero_ps();
						__m256 gval = _mm256_setzero_ps();
						__m256 rval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 bref = _mm256_load_ps(sptrbj + *ofs);
							const __m256 gref = _mm256_load_ps(sptrgj + *ofs);
							const __m256 rref = _mm256_load_ps(sptrrj + *ofs);
							_mm256_store_si256((__m256i*)bbuf, _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(bval0, bref), *(const __m256*)v32f_absmask)));
							_mm256_store_si256((__m256i*)gbuf, _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(gval0, gref), *(const __m256*)v32f_absmask)));
							_mm256_store_si256((__m256i*)rbuf, _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(rval0, rref), *(const __m256*)v32f_absmask)));

#if  __GRF_PREVENTION__	
							__m256 _w = _mm256_mul_ps(
								moffset,
								_mm256_set_ps(range_weight[bbuf[7]], range_weight[bbuf[6]], range_weight[bbuf[5]], range_weight[bbuf[4]], range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]])
							);
#else
							__m256 _w = _mm256_set_ps(range_weight[bbuf[7]], range_weight[bbuf[6]], range_weight[bbuf[5]], range_weight[bbuf[4]], range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]]);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[gbuf[7]], range_weight[gbuf[6]], range_weight[gbuf[5]], range_weight[gbuf[4]], range_weight[gbuf[3]], range_weight[gbuf[2]], range_weight[gbuf[1]], range_weight[gbuf[0]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[rbuf[7]], range_weight[rbuf[6]], range_weight[rbuf[5]], range_weight[rbuf[4]], range_weight[rbuf[3]], range_weight[rbuf[2]], range_weight[rbuf[1]], range_weight[rbuf[0]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_ps(bref, _w, bval);
							gval = _mm256_fmadd_ps(gref, _w, gval);
							rval = _mm256_fmadd_ps(rref, _w, rval);
#else
							const __m256 bref_mul = _mm256_mul_ps(bref, _w);
							bval = _mm256_add_ps(bval, bref_mul);
							const __m256 gref_mul = _mm256_mul_ps(gref, _w);
							gval = _mm256_add_ps(gval, gref_mul);
							const __m256 rref_mul = _mm256_mul_ps(rref, _w);
							rval = _mm256_add_ps(rval, rref_mul);
#endif

							wval = _mm256_add_ps(wval, _w);
						}
						bval = _mm256_div_ps(bval, wval);
						gval = _mm256_div_ps(gval, wval);
						rval = _mm256_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm256_stream_ps_color(dptrc, bval, gval, rval);
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
						const float w = range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)abs(r - r0)];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j + 0] = sum_b / wsum;
					dptr[3 * j + 1] = sum_g / wsum;
					dptr[3 * j + 2] = sum_r / wsum;
				}
			}
		}
	}

private:
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *range_weight;
	const float offset;
};

class GaussianRangeFilter_LUT_Setx3_8u_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Setx3_8u_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float* _range_weight, const float offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX2
		const bool haveAVX2 = checkHardwareSupport(CV_CPU_AVX2);
#endif
		if (cn == 1)
		{
			uchar CV_DECL_ALIGNED(32) buf[32];

			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 32 * (radiusH / 32 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrj = sptr + j;
						const __m256i sval0_32elem = _mm256_load_si256((__m256i*)sptrj);

						__m256 tval1 = _mm256_setzero_ps();
						__m256 wval1 = _mm256_setzero_ps();
						__m256 tval2 = _mm256_setzero_ps();
						__m256 wval2 = _mm256_setzero_ps();
						__m256 tval3 = _mm256_setzero_ps();
						__m256 wval3 = _mm256_setzero_ps();
						__m256 tval4 = _mm256_setzero_ps();
						__m256 wval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i sref_32elem = _mm256_loadu_si256((__m256i*)(sptrj + *ofs));
							_mm256_store_si256((__m256i*)buf, _mm256_add_epi8(_mm256_subs_epu8(sval0_32elem, sref_32elem), _mm256_subs_epu8(sref_32elem, sval0_32elem)));

							// lower bit
							__m256i sref_16elem = _mm256_unpacklo_epi8(sref_32elem, zero);

							__m256 _w = _mm256_set_ps(range_weight[buf[19]], range_weight[buf[18]], range_weight[buf[17]], range_weight[buf[16]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							__m256 sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm256_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval1 = _mm256_add_ps(tval1, sref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


							_w = _mm256_set_ps(range_weight[buf[23]], range_weight[buf[22]], range_weight[buf[21]], range_weight[buf[20]], range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]]);
							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm256_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval2 = _mm256_add_ps(tval2, sref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// higher bit
							sref_16elem = _mm256_unpackhi_epi8(sref_32elem, zero);

							_w = _mm256_set_ps(range_weight[buf[27]], range_weight[buf[26]], range_weight[buf[25]], range_weight[buf[24]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]]);
							sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm256_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval3 = _mm256_add_ps(tval3, sref);
#endif
							wval3 = _mm256_add_ps(wval3, _w);

							_w = _mm256_set_ps(range_weight[buf[31]], range_weight[buf[30]], range_weight[buf[29]], range_weight[buf[28]], range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]]);
							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm256_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval4 = _mm256_add_ps(tval4, sref);
#endif
							wval4 = _mm256_add_ps(wval4, _w);
						}
						tval1 = _mm256_div_ps(tval1, wval1);
						tval2 = _mm256_div_ps(tval2, wval2);
						tval3 = _mm256_div_ps(tval3, wval3);
						tval4 = _mm256_div_ps(tval4, wval4);
						_mm256_stream_si256((__m256i*)(dptr + j), _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(tval1), _mm256_cvtps_epi32(tval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(tval3), _mm256_cvtps_epi32(tval4))));
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
					dptr[j] = (char)(sum / wsum);
				}
			}
		}
		else
		{
			uchar CV_DECL_ALIGNED(32) rbuf[32];
			uchar CV_DECL_ALIGNED(32) gbuf[32];
			uchar CV_DECL_ALIGNED(32) bbuf[32];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 32 * (radiusH / 32 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 32 * (radiusH / 32 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 32 * (radiusH / 32 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
#if  __GRF_PREVENTION__	
					const __m256 moffset = _mm256_set1_ps(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m256 float_min = _mm256_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m256i bval0_32elem = _mm256_load_si256((__m256i*)(sptrbj));
						const __m256i gval0_32elem = _mm256_load_si256((__m256i*)(sptrgj));
						const __m256i rval0_32elem = _mm256_load_si256((__m256i*)(sptrrj));

						__m256 wval1 = _mm256_setzero_ps();
						__m256 bval1 = _mm256_setzero_ps();
						__m256 gval1 = _mm256_setzero_ps();
						__m256 rval1 = _mm256_setzero_ps();

						__m256 wval2 = _mm256_setzero_ps();
						__m256 bval2 = _mm256_setzero_ps();
						__m256 gval2 = _mm256_setzero_ps();
						__m256 rval2 = _mm256_setzero_ps();

						__m256 wval3 = _mm256_setzero_ps();
						__m256 bval3 = _mm256_setzero_ps();
						__m256 gval3 = _mm256_setzero_ps();
						__m256 rval3 = _mm256_setzero_ps();

						__m256 wval4 = _mm256_setzero_ps();
						__m256 bval4 = _mm256_setzero_ps();
						__m256 gval4 = _mm256_setzero_ps();
						__m256 rval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i bref_32elem = _mm256_loadu_si256((__m256i*)(sptrbj + *ofs));
							const __m256i gref_32elem = _mm256_loadu_si256((__m256i*)(sptrgj + *ofs));
							const __m256i rref_32elem = _mm256_loadu_si256((__m256i*)(sptrrj + *ofs));
							_mm256_store_si256((__m256i*)bbuf, _mm256_add_epi8(_mm256_subs_epu8(bval0_32elem, bref_32elem), _mm256_subs_epu8(bref_32elem, bval0_32elem)));
							_mm256_store_si256((__m256i*)gbuf, _mm256_add_epi8(_mm256_subs_epu8(gval0_32elem, gref_32elem), _mm256_subs_epu8(gref_32elem, gval0_32elem)));
							_mm256_store_si256((__m256i*)rbuf, _mm256_add_epi8(_mm256_subs_epu8(rval0_32elem, rref_32elem), _mm256_subs_epu8(rref_32elem, rval0_32elem)));

							// lower bit
							__m256i bref_16elem = _mm256_unpacklo_epi8(bref_32elem, zero);
							__m256i gref_16elem = _mm256_unpacklo_epi8(gref_32elem, zero);
							__m256i rref_16elem = _mm256_unpacklo_epi8(rref_32elem, zero);

#if __GRF_PREVENTION__
							__m256 _w = _mm256_mul_ps(
								moffset,
								_mm256_set_ps(range_weight[bbuf[19]], range_weight[bbuf[18]], range_weight[bbuf[17]], range_weight[bbuf[16]], range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]])
							);
#else
							__m256 _w = _mm256_set_ps(range_weight[bbuf[19]], range_weight[bbuf[18]], range_weight[bbuf[17]], range_weight[bbuf[16]], range_weight[bbuf[3]], range_weight[bbuf[2]], range_weight[bbuf[1]], range_weight[bbuf[0]]);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[gbuf[19]], range_weight[gbuf[18]], range_weight[gbuf[17]], range_weight[gbuf[16]], range_weight[gbuf[3]], range_weight[gbuf[2]], range_weight[gbuf[1]], range_weight[gbuf[0]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[rbuf[19]], range_weight[rbuf[18]], range_weight[rbuf[17]], range_weight[rbuf[16]], range_weight[rbuf[3]], range_weight[rbuf[2]], range_weight[rbuf[1]], range_weight[rbuf[0]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

							__m256 bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							__m256 gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							__m256 rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm256_fmadd_ps(bref, _w, bval1);
							gval1 = _mm256_fmadd_ps(gref, _w, gval1);
							rval1 = _mm256_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval1 = _mm256_add_ps(bval1, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval1 = _mm256_add_ps(gval1, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval1 = _mm256_add_ps(rval1, rref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


#if __GRF_PREVENTION__
							_w = _mm256_mul_ps(
								moffset,
								_mm256_set_ps(range_weight[bbuf[23]], range_weight[bbuf[22]], range_weight[bbuf[21]], range_weight[bbuf[20]], range_weight[bbuf[7]], range_weight[bbuf[6]], range_weight[bbuf[5]], range_weight[bbuf[4]])
							);
#else
							_w = _mm256_set_ps(range_weight[bbuf[23]], range_weight[bbuf[22]], range_weight[bbuf[21]], range_weight[bbuf[20]], range_weight[bbuf[7]], range_weight[bbuf[6]], range_weight[bbuf[5]], range_weight[bbuf[4]]);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[gbuf[23]], range_weight[gbuf[22]], range_weight[gbuf[21]], range_weight[gbuf[20]], range_weight[gbuf[7]], range_weight[gbuf[6]], range_weight[gbuf[5]], range_weight[gbuf[4]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[rbuf[23]], range_weight[rbuf[22]], range_weight[rbuf[21]], range_weight[rbuf[20]], range_weight[rbuf[7]], range_weight[rbuf[6]], range_weight[rbuf[5]], range_weight[rbuf[4]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm256_fmadd_ps(bref, _w, bval2);
							gval2 = _mm256_fmadd_ps(gref, _w, gval2);
							rval2 = _mm256_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval2 = _mm256_add_ps(bval2, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval2 = _mm256_add_ps(gval2, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval2 = _mm256_add_ps(rval2, rref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// higher bit
							bref_16elem = _mm256_unpackhi_epi8(bref_32elem, zero);
							gref_16elem = _mm256_unpackhi_epi8(gref_32elem, zero);
							rref_16elem = _mm256_unpackhi_epi8(rref_32elem, zero);

#if __GRF_PREVENTION__
							_w = _mm256_mul_ps(
								moffset,
								_mm256_set_ps(range_weight[bbuf[27]], range_weight[bbuf[26]], range_weight[bbuf[25]], range_weight[bbuf[24]], range_weight[bbuf[11]], range_weight[bbuf[10]], range_weight[bbuf[9]], range_weight[bbuf[8]])
							);
#else
							_w = _mm256_set_ps(range_weight[bbuf[27]], range_weight[bbuf[26]], range_weight[bbuf[25]], range_weight[bbuf[24]], range_weight[bbuf[11]], range_weight[bbuf[10]], range_weight[bbuf[9]], range_weight[bbuf[8]]);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[gbuf[27]], range_weight[gbuf[26]], range_weight[gbuf[25]], range_weight[gbuf[24]], range_weight[gbuf[11]], range_weight[gbuf[10]], range_weight[gbuf[9]], range_weight[gbuf[8]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[rbuf[27]], range_weight[rbuf[26]], range_weight[rbuf[25]], range_weight[rbuf[24]], range_weight[rbuf[11]], range_weight[rbuf[10]], range_weight[rbuf[9]], range_weight[rbuf[8]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

							bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm256_fmadd_ps(bref, _w, bval3);
							gval3 = _mm256_fmadd_ps(gref, _w, gval3);
							rval3 = _mm256_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval3 = _mm256_add_ps(bval3, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval3 = _mm256_add_ps(gval3, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval3 = _mm256_add_ps(rval3, rref);
#endif
							wval3 = _mm256_add_ps(wval3, _w);

#if __GRF_PREVENTION__
							_w = _mm256_mul_ps(
								moffset,
								_mm256_set_ps(range_weight[bbuf[31]], range_weight[bbuf[30]], range_weight[bbuf[29]], range_weight[bbuf[28]], range_weight[bbuf[15]], range_weight[bbuf[14]], range_weight[bbuf[13]], range_weight[bbuf[12]])
							);
#else
							_w = _mm256_set_ps(range_weight[bbuf[31]], range_weight[bbuf[30]], range_weight[bbuf[29]], range_weight[bbuf[28]], range_weight[bbuf[15]], range_weight[bbuf[14]], range_weight[bbuf[13]], range_weight[bbuf[12]]);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[gbuf[31]], range_weight[gbuf[30]], range_weight[gbuf[29]], range_weight[gbuf[28]], range_weight[gbuf[15]], range_weight[gbuf[14]], range_weight[gbuf[13]], range_weight[gbuf[12]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_set_ps(range_weight[rbuf[31]], range_weight[rbuf[30]], range_weight[rbuf[29]], range_weight[rbuf[28]], range_weight[rbuf[15]], range_weight[rbuf[14]], range_weight[rbuf[13]], range_weight[rbuf[12]]));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm256_fmadd_ps(bref, _w, bval4);
							gval4 = _mm256_fmadd_ps(gref, _w, gval4);
							rval4 = _mm256_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval4 = _mm256_add_ps(bval4, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval4 = _mm256_add_ps(gval4, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval4 = _mm256_add_ps(rval4, rref);
#endif
							wval4 = _mm256_add_ps(wval4, _w);
						}
						bval1 = _mm256_div_ps(bval1, wval1);
						bval2 = _mm256_div_ps(bval2, wval2);
						bval3 = _mm256_div_ps(bval3, wval3);
						bval4 = _mm256_div_ps(bval4, wval4);
						const __m256i a = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(bval1), _mm256_cvtps_epi32(bval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(bval3), _mm256_cvtps_epi32(bval4)));
						gval1 = _mm256_div_ps(gval1, wval1);
						gval2 = _mm256_div_ps(gval2, wval2);
						gval3 = _mm256_div_ps(gval3, wval3);
						gval4 = _mm256_div_ps(gval4, wval4);
						const __m256i b = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(gval1), _mm256_cvtps_epi32(gval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(gval3), _mm256_cvtps_epi32(gval4)));
						rval1 = _mm256_div_ps(rval1, wval1);
						rval2 = _mm256_div_ps(rval2, wval2);
						rval3 = _mm256_div_ps(rval3, wval3);
						rval4 = _mm256_div_ps(rval4, wval4);
						const __m256i c = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(rval1), _mm256_cvtps_epi32(rval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(rval3), _mm256_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm256_stream_epi8_color(dptrc, a, b, c);
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *range_weight;
	const float offset;
};


/*************************************************
	using LUT with "gather instruction" x 3
**************************************************/
class GaussianRangeFilter_LUT_Gatherx3_64f_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Gatherx3_64f_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double* _range_weight, const double offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX);
		static const long long CV_DECL_ALIGNED(32) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff,
		};
#endif
		if (cn == 1)
		{
			const double* sptr = temp->ptr<double>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrj = sptr + j;
						const __m256d sval0 = _mm256_load_pd(sptrj);

						__m256d tval = _mm256_setzero_pd();
						__m256d wval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
							const __m128i diff = _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(sval0, sref), *(const __m256d*)v64f_absmask));

							__m256d _w = _mm256_i32gather_pd(range_weight, diff, 8);
#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_pd(sref, _w, tval);
#else
							const __m256d sref_mul = _mm256_mul_pd(sref, _w);
							tval = _mm256_add_pd(tval, sref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						tval = _mm256_div_pd(tval, wval);
						_mm256_stream_pd((dptr + j), tval);
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
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start + 0) + 4 * (radiusH / 4 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

			double* dptr = dest->ptr<double>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
#if  __GRF_PREVENTION__	
					const __m256d moffset = _mm256_set1_pd(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m256d double_min = _mm256_set1_pd(DBL_MIN);
#endif
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m256d bval0 = _mm256_load_pd(sptrbj);
						const __m256d gval0 = _mm256_load_pd(sptrgj);
						const __m256d rval0 = _mm256_load_pd(sptrrj);

						__m256d wval = _mm256_setzero_pd();
						__m256d bval = _mm256_setzero_pd();
						__m256d gval = _mm256_setzero_pd();
						__m256d rval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d bref = _mm256_load_pd(sptrbj + *ofs);
							const __m256d gref = _mm256_load_pd(sptrgj + *ofs);
							const __m256d rref = _mm256_load_pd(sptrrj + *ofs);
							const __m128i bdiff = _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(bval0, bref), *(const __m256d*)v64f_absmask));
							const __m128i gdiff = _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(gval0, gref), *(const __m256d*)v64f_absmask));
							const __m128i rdiff = _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(rval0, rref), *(const __m256d*)v64f_absmask));

#if __GRF_PREVENTION__
							__m256d _w = _mm256_mul_pd(
								moffset,
								_mm256_i32gather_pd(range_weight, bdiff, 8)
							);
#else
							__m256d _w = _mm256_i32gather_pd(range_weight, bdiff, 8);
#endif
							_w = _mm256_mul_pd(_w, _mm256_i32gather_pd(range_weight, gdiff, 8));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_pd(_w, double_min);
#endif
							_w = _mm256_mul_pd(_w, _mm256_i32gather_pd(range_weight, rdiff, 8));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_pd(_w, double_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_pd(bref, _w, bval);
							gval = _mm256_fmadd_pd(gref, _w, gval);
							rval = _mm256_fmadd_pd(rref, _w, rval);
#else
							const __m256d bref_mul = _mm256_mul_pd(bref, _w);
							bval = _mm256_add_pd(bval, bref_mul);
							const __m256d gref_mul = _mm256_mul_pd(gref, _w);
							gval = _mm256_add_pd(gval, gref_mul);
							const __m256d rref_mul = _mm256_mul_pd(rref, _w);
							rval = _mm256_add_pd(rval, rref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						bval = _mm256_div_pd(bval, wval);
						gval = _mm256_div_pd(gval, wval);
						rval = _mm256_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm256_stream_pd_color(dptrc, bval, gval, rval);
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
						const double w = range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)abs(r - r0)];
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *range_weight;
	const double offset;
};

class GaussianRangeFilter_LUT_Gatherx3_32f_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Gatherx3_32f_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float* _range_weight, const float offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX);
		static const int CV_DECL_ALIGNED(32) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
#endif
		if (cn == 1)
		{
			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrj = sptr + j;
						const __m256 sval0 = _mm256_load_ps(sptrj);

						__m256 tval = _mm256_setzero_ps();
						__m256 wval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 sref = _mm256_loadu_ps((sptrj + *ofs));
							const __m256i diff = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(sval0, sref), *(const __m256*)v32f_absmask));

							__m256 _w = _mm256_i32gather_ps(range_weight, diff, 4);
#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_ps(sref, _w, tval);
#else
							const __m256 sref_mul = _mm256_mul_ps(sref, _w);
							tval = _mm256_add_ps(tval, sref_mul);
#endif
							wval = _mm256_add_ps(wval, _w);
						}
						tval = _mm256_div_ps(tval, wval);
						_mm256_stream_ps((dptr + j), tval);
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
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			float* dptr = dest->ptr<float>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
#if  __GRF_PREVENTION__	
					const __m256 moffset = _mm256_set1_ps(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m256 float_min = _mm256_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m256 bval0 = _mm256_load_ps(sptrbj);
						const __m256 gval0 = _mm256_load_ps(sptrgj);
						const __m256 rval0 = _mm256_load_ps(sptrrj);

						__m256 wval = _mm256_setzero_ps();
						__m256 bval = _mm256_setzero_ps();
						__m256 gval = _mm256_setzero_ps();
						__m256 rval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 bref = _mm256_load_ps(sptrbj + *ofs);
							const __m256 gref = _mm256_load_ps(sptrgj + *ofs);
							const __m256 rref = _mm256_load_ps(sptrrj + *ofs);
							const __m256i bdiff = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(bval0, bref), *(const __m256*)v32f_absmask));
							const __m256i gdiff = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(gval0, gref), *(const __m256*)v32f_absmask));
							const __m256i rdiff = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(rval0, rref), *(const __m256*)v32f_absmask));

#if __GRF_PREVENTION__
							__m256 _w = _mm256_mul_ps(
								moffset,
								_mm256_i32gather_ps(range_weight, bdiff, 4)
							);
#else
							__m256 _w = _mm256_i32gather_ps(range_weight, bdiff, 4);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, gdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, rdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_ps(bref, _w, bval);
							gval = _mm256_fmadd_ps(gref, _w, gval);
							rval = _mm256_fmadd_ps(rref, _w, rval);
#else
							const __m256 bref_mul = _mm256_mul_ps(bref, _w);
							bval = _mm256_add_ps(bval, bref_mul);
							const __m256 gref_mul = _mm256_mul_ps(gref, _w);
							gval = _mm256_add_ps(gval, gref_mul);
							const __m256 rref_mul = _mm256_mul_ps(rref, _w);
							rval = _mm256_add_ps(rval, rref_mul);
#endif

							wval = _mm256_add_ps(wval, _w);
						}
						bval = _mm256_div_ps(bval, wval);
						gval = _mm256_div_ps(gval, wval);
						rval = _mm256_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm256_stream_ps_color(dptrc, bval, gval, rval);
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
						const float w = range_weight[(uchar)abs(b - b0)] * range_weight[(uchar)abs(g - g0)] * range_weight[(uchar)abs(r - r0)];
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *range_weight;
	const float offset;
};

class GaussianRangeFilter_LUT_Gatherx3_8u_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_LUT_Gatherx3_8u_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float* _range_weight, const float offset = 1) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight), offset(offset)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX2
		const bool haveAVX2 = checkHardwareSupport(CV_CPU_AVX2);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 32 * (radiusH / 32 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrj = sptr + j;
						const __m256i sval0_32elem = _mm256_load_si256((__m256i*)sptrj);

						__m256 tval1 = _mm256_setzero_ps();
						__m256 wval1 = _mm256_setzero_ps();
						__m256 tval2 = _mm256_setzero_ps();
						__m256 wval2 = _mm256_setzero_ps();
						__m256 tval3 = _mm256_setzero_ps();
						__m256 wval3 = _mm256_setzero_ps();
						__m256 tval4 = _mm256_setzero_ps();
						__m256 wval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i sref_32elem = _mm256_loadu_si256((__m256i*)(sptrj + *ofs));
							const __m256i diff_32elem = _mm256_add_epi8(_mm256_subs_epu8(sval0_32elem, sref_32elem), _mm256_subs_epu8(sref_32elem, sval0_32elem));

							// lower bit
							__m256i sref_16elem = _mm256_unpacklo_epi8(sref_32elem, zero);
							__m256i diff_16elem = _mm256_unpacklo_epi8(diff_32elem, zero);

							__m256i diff = _mm256_unpacklo_epi16(diff_16elem, zero);
							__m256 _w = _mm256_i32gather_ps(range_weight, diff, 4);
							__m256 sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm256_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval1 = _mm256_add_ps(tval1, sref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


							diff = _mm256_unpackhi_epi16(diff_16elem, zero);
							_w = _mm256_i32gather_ps(range_weight, diff, 4);
							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm256_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval2 = _mm256_add_ps(tval2, sref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// higher bit
							sref_16elem = _mm256_unpackhi_epi8(sref_32elem, zero);
							diff_16elem = _mm256_unpackhi_epi8(diff_32elem, zero);

							diff = _mm256_unpacklo_epi16(diff_16elem, zero);
							_w = _mm256_i32gather_ps(range_weight, diff, 4);
							sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm256_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval3 = _mm256_add_ps(tval3, sref);
#endif
							wval3 = _mm256_add_ps(wval3, _w);


							diff = _mm256_unpackhi_epi16(diff_16elem, zero);
							_w = _mm256_i32gather_ps(range_weight, diff, 4);
							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm256_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval4 = _mm256_add_ps(tval4, sref);
#endif
							wval4 = _mm256_add_ps(wval4, _w);
						}
						tval1 = _mm256_div_ps(tval1, wval1);
						tval2 = _mm256_div_ps(tval2, wval2);
						tval3 = _mm256_div_ps(tval3, wval3);
						tval4 = _mm256_div_ps(tval4, wval4);
						_mm256_stream_si256((__m256i*)(dptr + j), _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(tval1), _mm256_cvtps_epi32(tval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(tval3), _mm256_cvtps_epi32(tval4))));
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
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 32 * (radiusH / 32 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 32 * (radiusH / 32 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 32 * (radiusH / 32 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
#if  __GRF_PREVENTION__	
					const __m256 moffset = _mm256_set1_ps(offset);
#endif
#if  __GRF_POSTVENTION__	
					static const __m256 float_min = _mm256_set1_ps(FLT_MIN);
#endif
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m256i bval0_32elem = _mm256_load_si256((__m256i*)(sptrbj));
						const __m256i gval0_32elem = _mm256_load_si256((__m256i*)(sptrgj));
						const __m256i rval0_32elem = _mm256_load_si256((__m256i*)(sptrrj));

						__m256 wval1 = _mm256_setzero_ps();
						__m256 bval1 = _mm256_setzero_ps();
						__m256 gval1 = _mm256_setzero_ps();
						__m256 rval1 = _mm256_setzero_ps();

						__m256 wval2 = _mm256_setzero_ps();
						__m256 bval2 = _mm256_setzero_ps();
						__m256 gval2 = _mm256_setzero_ps();
						__m256 rval2 = _mm256_setzero_ps();

						__m256 wval3 = _mm256_setzero_ps();
						__m256 bval3 = _mm256_setzero_ps();
						__m256 gval3 = _mm256_setzero_ps();
						__m256 rval3 = _mm256_setzero_ps();

						__m256 wval4 = _mm256_setzero_ps();
						__m256 bval4 = _mm256_setzero_ps();
						__m256 gval4 = _mm256_setzero_ps();
						__m256 rval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i bref_32elem = _mm256_loadu_si256((__m256i*)(sptrbj + *ofs));
							const __m256i gref_32elem = _mm256_loadu_si256((__m256i*)(sptrgj + *ofs));
							const __m256i rref_32elem = _mm256_loadu_si256((__m256i*)(sptrrj + *ofs));

							const __m256i bdiff_32elem = _mm256_add_epi8(_mm256_subs_epu8(bval0_32elem, bref_32elem), _mm256_subs_epu8(bref_32elem, bval0_32elem));
							const __m256i gdiff_32elem = _mm256_add_epi8(_mm256_subs_epu8(gval0_32elem, gref_32elem), _mm256_subs_epu8(gref_32elem, gval0_32elem));
							const __m256i rdiff_32elem = _mm256_add_epi8(_mm256_subs_epu8(rval0_32elem, rref_32elem), _mm256_subs_epu8(rref_32elem, rval0_32elem));

							// lower bit
							__m256i bref_16elem = _mm256_unpacklo_epi8(bref_32elem, zero);
							__m256i gref_16elem = _mm256_unpacklo_epi8(gref_32elem, zero);
							__m256i rref_16elem = _mm256_unpacklo_epi8(rref_32elem, zero);
							__m256i bdiff_16elem = _mm256_unpacklo_epi8(bdiff_32elem, zero);
							__m256i gdiff_16elem = _mm256_unpacklo_epi8(gdiff_32elem, zero);
							__m256i rdiff_16elem = _mm256_unpacklo_epi8(rdiff_32elem, zero);

							__m256i bdiff = _mm256_unpacklo_epi16(bdiff_16elem, zero);
							__m256i gdiff = _mm256_unpacklo_epi16(gdiff_16elem, zero);
							__m256i rdiff = _mm256_unpacklo_epi16(rdiff_16elem, zero);
#if __GRF_PREVENTION__
							__m256 _w = _mm256_mul_ps(
								moffset,
								_mm256_i32gather_ps(range_weight, bdiff, 4)
							);
#else
							__m256 _w = _mm256_i32gather_ps(range_weight, bdiff, 4);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, gdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, rdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

							__m256 bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							__m256 gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							__m256 rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm256_fmadd_ps(bref, _w, bval1);
							gval1 = _mm256_fmadd_ps(gref, _w, gval1);
							rval1 = _mm256_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval1 = _mm256_add_ps(bval1, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval1 = _mm256_add_ps(gval1, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval1 = _mm256_add_ps(rval1, rref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


							bdiff = _mm256_unpackhi_epi16(bdiff_16elem, zero);
							gdiff = _mm256_unpackhi_epi16(gdiff_16elem, zero);
							rdiff = _mm256_unpackhi_epi16(rdiff_16elem, zero);

#if __GRF_PREVENTION__
							_w = _mm256_mul_ps(
								moffset,
								_mm256_i32gather_ps(range_weight, bdiff, 4)
							);
#else
							_w = _mm256_i32gather_ps(range_weight, bdiff, 4);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, gdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, rdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm256_fmadd_ps(bref, _w, bval2);
							gval2 = _mm256_fmadd_ps(gref, _w, gval2);
							rval2 = _mm256_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval2 = _mm256_add_ps(bval2, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval2 = _mm256_add_ps(gval2, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval2 = _mm256_add_ps(rval2, rref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// higher bit
							bref_16elem = _mm256_unpackhi_epi8(bref_32elem, zero);
							gref_16elem = _mm256_unpackhi_epi8(gref_32elem, zero);
							rref_16elem = _mm256_unpackhi_epi8(rref_32elem, zero);
							bdiff_16elem = _mm256_unpackhi_epi8(bdiff_32elem, zero);
							gdiff_16elem = _mm256_unpackhi_epi8(gdiff_32elem, zero);
							rdiff_16elem = _mm256_unpackhi_epi8(rdiff_32elem, zero);

							bdiff = _mm256_unpacklo_epi16(bdiff_16elem, zero);
							gdiff = _mm256_unpacklo_epi16(gdiff_16elem, zero);
							rdiff = _mm256_unpacklo_epi16(rdiff_16elem, zero);

#if __GRF_PREVENTION__
							_w = _mm256_mul_ps(
								moffset,
								_mm256_i32gather_ps(range_weight, bdiff, 4)
							);
#else
							_w = _mm256_i32gather_ps(range_weight, bdiff, 4);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, gdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, rdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm256_fmadd_ps(bref, _w, bval3);
							gval3 = _mm256_fmadd_ps(gref, _w, gval3);
							rval3 = _mm256_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval3 = _mm256_add_ps(bval3, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval3 = _mm256_add_ps(gval3, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval3 = _mm256_add_ps(rval3, rref);
#endif
							wval3 = _mm256_add_ps(wval3, _w);


							bdiff = _mm256_unpackhi_epi16(bdiff_16elem, zero);
							gdiff = _mm256_unpackhi_epi16(gdiff_16elem, zero);
							rdiff = _mm256_unpackhi_epi16(rdiff_16elem, zero);

#if __GRF_PREVENTION__
							_w = _mm256_mul_ps(
								moffset,
								_mm256_i32gather_ps(range_weight, bdiff, 4)
							);
#else
							_w = _mm256_i32gather_ps(range_weight, bdiff, 4);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, gdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif
							_w = _mm256_mul_ps(_w, _mm256_i32gather_ps(range_weight, rdiff, 4));
#if  __GRF_POSTVENTION__	
							_w = _mm256_max_ps(_w, float_min);
#endif

							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm256_fmadd_ps(bref, _w, bval4);
							gval4 = _mm256_fmadd_ps(gref, _w, gval4);
							rval4 = _mm256_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval4 = _mm256_add_ps(bval4, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval4 = _mm256_add_ps(gval4, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval4 = _mm256_add_ps(rval4, rref);
#endif
							wval4 = _mm256_add_ps(wval4, _w);
						}
						bval1 = _mm256_div_ps(bval1, wval1);
						bval2 = _mm256_div_ps(bval2, wval2);
						bval3 = _mm256_div_ps(bval3, wval3);
						bval4 = _mm256_div_ps(bval4, wval4);
						const __m256i a = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(bval1), _mm256_cvtps_epi32(bval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(bval3), _mm256_cvtps_epi32(bval4)));
						gval1 = _mm256_div_ps(gval1, wval1);
						gval2 = _mm256_div_ps(gval2, wval2);
						gval3 = _mm256_div_ps(gval3, wval3);
						gval4 = _mm256_div_ps(gval4, wval4);
						const __m256i b = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(gval1), _mm256_cvtps_epi32(gval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(gval3), _mm256_cvtps_epi32(gval4)));
						rval1 = _mm256_div_ps(rval1, wval1);
						rval2 = _mm256_div_ps(rval2, wval2);
						rval3 = _mm256_div_ps(rval3, wval3);
						rval4 = _mm256_div_ps(rval4, wval4);
						const __m256i c = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(rval1), _mm256_cvtps_epi32(rval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(rval3), _mm256_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm256_stream_epi8_color(dptrc, a, b, c);
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *range_weight;
	const float offset;
};


/*************************************************
	using quantized range LUT with "set instruction" x 1
**************************************************/
class GaussianRangeFilter_QuantizationLUT_Setx1_64f_InvokerAVX : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationLUT_Setx1_64f_InvokerAVX(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double* _range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX
		static const long long CV_DECL_ALIGNED(32) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff,
		};
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(32) buf[4];

			const double* sptr = temp->ptr<double>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrj = sptr + j;
						const __m256d sval0 = _mm256_load_pd(sptrj);

						__m256d tval = _mm256_setzero_pd();
						__m256d wval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
							_mm_store_si128((__m128i*)buf, _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(sval0, sref), *(const __m256d*)v64f_absmask)));

							__m256d _w = _mm256_set_pd(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_pd(sref, _w, tval);
#else
							const __m256d sref_mul = _mm256_mul_pd(sref, _w);
							tval = _mm256_add_pd(tval, sref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						tval = _mm256_div_pd(tval, wval);
						_mm256_stream_pd((dptr + j), tval);
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
			int CV_DECL_ALIGNED(32) buf[4];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start + 0) + 4 * (radiusH / 4 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

			double* dptr = dest->ptr<double>(range.start);

			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m256d bval0 = _mm256_load_pd(sptrbj);
						const __m256d gval0 = _mm256_load_pd(sptrgj);
						const __m256d rval0 = _mm256_load_pd(sptrrj);

						__m256d wval = _mm256_setzero_pd();
						__m256d bval = _mm256_setzero_pd();
						__m256d gval = _mm256_setzero_pd();
						__m256d rval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d bref = _mm256_load_pd(sptrbj + *ofs);
							const __m256d gref = _mm256_load_pd(sptrgj + *ofs);
							const __m256d rref = _mm256_load_pd(sptrrj + *ofs);

							const __m256d bdiff = _mm256_sub_pd(bval0, bref);
							const __m256d gdiff = _mm256_sub_pd(gval0, gref);
							const __m256d rdiff = _mm256_sub_pd(rval0, rref);
							__m256d difft = _mm256_mul_pd(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm256_fmadd_pd(rdiff, rdiff, difft);
#else
							const __m256d gdiff_mul = _mm256_mul_pd(gdiff, gdiff);
							difft = _mm256_add_pd(difft, gdiff_mul);
							const __m256d rdiff_mul = _mm256_mul_pd(rdiff, rdiff);
							difft = _mm256_add_pd(difft, rdiff_mul);
#endif
							difft = _mm256_sqrt_pd(difft);
							_mm_store_si128((__m128i*)buf, _mm256_cvtpd_epi32(difft));
							__m256d _w = _mm256_set_pd(range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_pd(bref, _w, bval);
							gval = _mm256_fmadd_pd(gref, _w, gval);
							rval = _mm256_fmadd_pd(rref, _w, rval);
#else
							const __m256d bref_mul = _mm256_mul_pd(bref, _w);
							bval = _mm256_add_pd(bval, bref_mul);
							const __m256d gref_mul = _mm256_mul_pd(gref, _w);
							gval = _mm256_add_pd(gval, gref_mul);
							const __m256d rref_mul = _mm256_mul_pd(rref, _w);
							rval = _mm256_add_pd(rval, rref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						bval = _mm256_div_pd(bval, wval);
						gval = _mm256_div_pd(gval, wval);
						rval = _mm256_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm256_stream_pd_color(dptrc, bval, gval, rval);
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
						const double w = range_weight[(int)sqrt((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *range_weight;
};

class GaussianRangeFilter_QuantizationLUT_Setx1_32f_InvokerAVX : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationLUT_Setx1_32f_InvokerAVX(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float* _range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX
		static const int CV_DECL_ALIGNED(32) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX);
#endif
		if (cn == 1)
		{
			int CV_DECL_ALIGNED(32) buf[8];

			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrj = sptr + j;
						const __m256 sval0 = _mm256_load_ps(sptrj);

						__m256 tval = _mm256_setzero_ps();
						__m256 wval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 sref = _mm256_loadu_ps((sptrj + *ofs));
							_mm256_store_si256((__m256i*)buf, _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(sval0, sref), *(const __m256*)v32f_absmask)));

							__m256 _w = _mm256_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_ps(sref, _w, tval);
#else
							const __m256 sref_mul = _mm256_mul_ps(sref, _w);
							tval = _mm256_add_ps(tval, sref_mul);
#endif
							wval = _mm256_add_ps(wval, _w);
						}
						tval = _mm256_div_ps(tval, wval);
						_mm256_stream_ps((dptr + j), tval);
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
			int CV_DECL_ALIGNED(32) buf[8];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			float* dptr = dest->ptr<float>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX
				if (haveAVX)
				{
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m256 bval0 = _mm256_load_ps(sptrbj);
						const __m256 gval0 = _mm256_load_ps(sptrgj);
						const __m256 rval0 = _mm256_load_ps(sptrrj);

						__m256 wval = _mm256_setzero_ps();
						__m256 bval = _mm256_setzero_ps();
						__m256 gval = _mm256_setzero_ps();
						__m256 rval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 bref = _mm256_load_ps(sptrbj + *ofs);
							const __m256 gref = _mm256_load_ps(sptrgj + *ofs);
							const __m256 rref = _mm256_load_ps(sptrrj + *ofs);

							const __m256 bdiff = _mm256_sub_ps(bval0, bref);
							const __m256 gdiff = _mm256_sub_ps(gval0, gref);
							const __m256 rdiff = _mm256_sub_ps(rval0, rref);
							__m256 difft = _mm256_mul_ps(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							const __m256 gdiff_mul = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff_mul);
							const __m256 rdiff_mul = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff_mul);
#endif

							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							_mm256_store_si256((__m256i*)buf, _mm256_cvtps_epi32(difft));
							__m256 _w = _mm256_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_ps(bref, _w, bval);
							gval = _mm256_fmadd_ps(gref, _w, gval);
							rval = _mm256_fmadd_ps(rref, _w, rval);
#else
							const __m256 bref_mul = _mm256_mul_ps(bref, _w);
							bval = _mm256_add_ps(bval, bref_mul);
							const __m256 gref_mul = _mm256_mul_ps(gref, _w);
							gval = _mm256_add_ps(gval, gref_mul);
							const __m256 rref_mul = _mm256_mul_ps(rref, _w);
							rval = _mm256_add_ps(rval, rref_mul);
#endif
							wval = _mm256_add_ps(wval, _w);
						}
						bval = _mm256_div_ps(bval, wval);
						gval = _mm256_div_ps(gval, wval);
						rval = _mm256_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm256_stream_ps_color(dptrc, bval, gval, rval);
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
						const float w = range_weight[(int)sqrt((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *range_weight;
};

class GaussianRangeFilter_QuantizationLUT_Setx1_8u_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationLUT_Setx1_8u_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float* _range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
#if CV_AVX2
		const bool haveAVX2 = checkHardwareSupport(CV_CPU_AVX2);
#endif
		if (cn == 1)
		{
			uchar CV_DECL_ALIGNED(32) buf[32];

			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 32 * (radiusH / 32 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrj = sptr + j;
						const __m256i sval0_32elem = _mm256_load_si256((__m256i*)sptrj);

						__m256 tval1 = _mm256_setzero_ps();
						__m256 wval1 = _mm256_setzero_ps();
						__m256 tval2 = _mm256_setzero_ps();
						__m256 wval2 = _mm256_setzero_ps();
						__m256 tval3 = _mm256_setzero_ps();
						__m256 wval3 = _mm256_setzero_ps();
						__m256 tval4 = _mm256_setzero_ps();
						__m256 wval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i sref_32elem = _mm256_loadu_si256((__m256i*)(sptrj + *ofs));
							_mm256_store_si256((__m256i*)buf, _mm256_add_epi8(_mm256_subs_epu8(sval0_32elem, sref_32elem), _mm256_subs_epu8(sref_32elem, sval0_32elem)));

							// lower bit
							__m256i sref_16elem = _mm256_unpacklo_epi8(sref_32elem, zero);

							__m256 _w = _mm256_set_ps(range_weight[buf[19]], range_weight[buf[18]], range_weight[buf[17]], range_weight[buf[16]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
							__m256 sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm256_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval1 = _mm256_add_ps(tval1, sref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


							_w = _mm256_set_ps(range_weight[buf[23]], range_weight[buf[22]], range_weight[buf[21]], range_weight[buf[20]], range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]]);
							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm256_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval2 = _mm256_add_ps(tval2, sref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// higher bit
							sref_16elem = _mm256_unpackhi_epi8(sref_32elem, zero);

							_w = _mm256_set_ps(range_weight[buf[27]], range_weight[buf[26]], range_weight[buf[25]], range_weight[buf[24]], range_weight[buf[11]], range_weight[buf[10]], range_weight[buf[9]], range_weight[buf[8]]);
							sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm256_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval3 = _mm256_add_ps(tval3, sref);
#endif
							wval3 = _mm256_add_ps(wval3, _w);

							_w = _mm256_set_ps(range_weight[buf[31]], range_weight[buf[30]], range_weight[buf[29]], range_weight[buf[28]], range_weight[buf[15]], range_weight[buf[14]], range_weight[buf[13]], range_weight[buf[12]]);
							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm256_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval4 = _mm256_add_ps(tval4, sref);
#endif
							wval4 = _mm256_add_ps(wval4, _w);
						}
						tval1 = _mm256_div_ps(tval1, wval1);
						tval2 = _mm256_div_ps(tval2, wval2);
						tval3 = _mm256_div_ps(tval3, wval3);
						tval4 = _mm256_div_ps(tval4, wval4);
						_mm256_stream_si256((__m256i*)(dptr + j), _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(tval1), _mm256_cvtps_epi32(tval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(tval3), _mm256_cvtps_epi32(tval4))));
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
					dptr[j] = (int)(sum / wsum);
				}
			}
		}
		else
		{
			int CV_DECL_ALIGNED(32) buf[8];

			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 32 * (radiusH / 32 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 32 * (radiusH / 32 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 32 * (radiusH / 32 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m256i bval0_32elem = _mm256_load_si256((__m256i*)(sptrbj));
						const __m256i gval0_32elem = _mm256_load_si256((__m256i*)(sptrgj));
						const __m256i rval0_32elem = _mm256_load_si256((__m256i*)(sptrrj));

						__m256i temp = _mm256_unpacklo_epi8(bval0_32elem, zero);
						const __m256 bval0_1 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 bval0_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpackhi_epi8(bval0_32elem, zero);
						const __m256 bval0_3 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 bval0_4 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpacklo_epi8(gval0_32elem, zero);
						const __m256 gval0_1 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 gval0_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpackhi_epi8(gval0_32elem, zero);
						const __m256 gval0_3 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 gval0_4 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpacklo_epi8(rval0_32elem, zero);
						const __m256 rval0_1 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 rval0_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpackhi_epi8(rval0_32elem, zero);
						const __m256 rval0_3 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 rval0_4 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));

						__m256 wval1 = _mm256_setzero_ps();
						__m256 bval1 = _mm256_setzero_ps();
						__m256 gval1 = _mm256_setzero_ps();
						__m256 rval1 = _mm256_setzero_ps();

						__m256 wval2 = _mm256_setzero_ps();
						__m256 bval2 = _mm256_setzero_ps();
						__m256 gval2 = _mm256_setzero_ps();
						__m256 rval2 = _mm256_setzero_ps();

						__m256 wval3 = _mm256_setzero_ps();
						__m256 bval3 = _mm256_setzero_ps();
						__m256 gval3 = _mm256_setzero_ps();
						__m256 rval3 = _mm256_setzero_ps();

						__m256 wval4 = _mm256_setzero_ps();
						__m256 bval4 = _mm256_setzero_ps();
						__m256 gval4 = _mm256_setzero_ps();
						__m256 rval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i bref_32elem = _mm256_loadu_si256((__m256i*)(sptrbj + *ofs));
							const __m256i gref_32elem = _mm256_loadu_si256((__m256i*)(sptrgj + *ofs));
							const __m256i rref_32elem = _mm256_loadu_si256((__m256i*)(sptrrj + *ofs));

							// low
							__m256i bref_16elem = _mm256_unpacklo_epi8(bref_32elem, zero);
							__m256i gref_16elem = _mm256_unpacklo_epi8(gref_32elem, zero);
							__m256i rref_16elem = _mm256_unpacklo_epi8(rref_32elem, zero);

							__m256 bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							__m256 gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							__m256 rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));

							__m256 bdiff = _mm256_sub_ps(bval0_1, bref);
							__m256 gdiff = _mm256_sub_ps(gval0_1, gref);
							__m256 rdiff = _mm256_sub_ps(rval0_1, rref);
							__m256 difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif

							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							_mm256_store_si256((__m256i*)buf, _mm256_cvtps_epi32(difft));
							__m256 _w = _mm256_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm256_fmadd_ps(bref, _w, bval1);
							gval1 = _mm256_fmadd_ps(gref, _w, gval1);
							rval1 = _mm256_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval1 = _mm256_add_ps(bval1, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval1 = _mm256_add_ps(gval1, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval1 = _mm256_add_ps(rval1, rref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));

							bdiff = _mm256_sub_ps(bval0_2, bref);
							gdiff = _mm256_sub_ps(gval0_2, gref);
							rdiff = _mm256_sub_ps(rval0_2, rref);
							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							_mm256_store_si256((__m256i*)buf, _mm256_cvtps_epi32(difft));
							_w = _mm256_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm256_fmadd_ps(bref, _w, bval2);
							gval2 = _mm256_fmadd_ps(gref, _w, gval2);
							rval2 = _mm256_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval2 = _mm256_add_ps(bval2, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval2 = _mm256_add_ps(gval2, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval2 = _mm256_add_ps(rval2, rref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// high
							bref_16elem = _mm256_unpackhi_epi8(bref_32elem, zero);
							gref_16elem = _mm256_unpackhi_epi8(gref_32elem, zero);
							rref_16elem = _mm256_unpackhi_epi8(rref_32elem, zero);

							bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));

							bdiff = _mm256_sub_ps(bval0_3, bref);
							gdiff = _mm256_sub_ps(gval0_3, gref);
							rdiff = _mm256_sub_ps(rval0_3, rref);
							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif

							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							_mm256_store_si256((__m256i*)buf, _mm256_cvtps_epi32(difft));
							_w = _mm256_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm256_fmadd_ps(bref, _w, bval3);
							gval3 = _mm256_fmadd_ps(gref, _w, gval3);
							rval3 = _mm256_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval3 = _mm256_add_ps(bval3, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval3 = _mm256_add_ps(gval3, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval3 = _mm256_add_ps(rval3, rref);
#endif
							wval3 = _mm256_add_ps(wval3, _w);


							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));

							bdiff = _mm256_sub_ps(bval0_4, bref);
							gdiff = _mm256_sub_ps(gval0_4, gref);
							rdiff = _mm256_sub_ps(rval0_4, rref);
							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							_mm256_store_si256((__m256i*)buf, _mm256_cvtps_epi32(difft));
							_w = _mm256_set_ps(range_weight[buf[7]], range_weight[buf[6]], range_weight[buf[5]], range_weight[buf[4]], range_weight[buf[3]], range_weight[buf[2]], range_weight[buf[1]], range_weight[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm256_fmadd_ps(bref, _w, bval4);
							gval4 = _mm256_fmadd_ps(gref, _w, gval4);
							rval4 = _mm256_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval4 = _mm256_add_ps(bval4, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval4 = _mm256_add_ps(gval4, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval4 = _mm256_add_ps(rval4, rref);
#endif
							wval4 = _mm256_add_ps(wval4, _w);
						}
						bval1 = _mm256_div_ps(bval1, wval1);
						bval2 = _mm256_div_ps(bval2, wval2);
						bval3 = _mm256_div_ps(bval3, wval3);
						bval4 = _mm256_div_ps(bval4, wval4);
						const __m256i a = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(bval1), _mm256_cvtps_epi32(bval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(bval3), _mm256_cvtps_epi32(bval4)));
						gval1 = _mm256_div_ps(gval1, wval1);
						gval2 = _mm256_div_ps(gval2, wval2);
						gval3 = _mm256_div_ps(gval3, wval3);
						gval4 = _mm256_div_ps(gval4, wval4);
						const __m256i b = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(gval1), _mm256_cvtps_epi32(gval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(gval3), _mm256_cvtps_epi32(gval4)));
						rval1 = _mm256_div_ps(rval1, wval1);
						rval2 = _mm256_div_ps(rval2, wval2);
						rval3 = _mm256_div_ps(rval3, wval3);
						rval4 = _mm256_div_ps(rval4, wval4);
						const __m256i c = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(rval1), _mm256_cvtps_epi32(rval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(rval3), _mm256_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm256_stream_epi8_color(dptrc, a, b, c);
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
						const float w = range_weight[(int)sqrt((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j + 0] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}

private:
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *range_weight;
};


/*************************************************
	using quantized range LUT with "gather instruction" x 1
**************************************************/
class GaussianRangeFilter_QuantizationLUT_Gatherx1_64f_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationLUT_Gatherx1_64f_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const double* _range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX2
		static const long long CV_DECL_ALIGNED(32) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff,	0x7fffffffffffffff,
		};
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX2);
#endif
		if (cn == 1)
		{
			const double* sptr = temp->ptr<double>(range.start + radiusV) + 4 * (radiusH / 4 + 1);
			double* dptr = dest->ptr<double>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrj = sptr + j;
						const __m256d sval0 = _mm256_load_pd(sptrj);

						__m256d tval = _mm256_setzero_pd();
						__m256d wval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d sref = _mm256_loadu_pd((sptrj + *ofs));
							const __m128i midx = _mm256_cvtpd_epi32(_mm256_and_pd(_mm256_sub_pd(sval0, sref), *(const __m256d*)v64f_absmask));

							__m256d _w = _mm256_i32gather_pd(range_weight, midx, 8);
#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_pd(sref, _w, tval);
#else
							const __m256d sref_mul = _mm256_mul_pd(sref, _w);
							tval = _mm256_add_pd(tval, sref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						tval = _mm256_div_pd(tval, wval);
						_mm256_stream_pd((dptr + j), tval);
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
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			const double* sptrb = temp->ptr<double>(3 * radiusV + 3 * range.start + 0) + 4 * (radiusH / 4 + 1);
			const double* sptrg = temp->ptr<double>(3 * radiusV + 3 * range.start + 1) + 4 * (radiusH / 4 + 1);
			const double* sptrr = temp->ptr<double>(3 * radiusV + 3 * range.start + 2) + 4 * (radiusH / 4 + 1);

			double* dptr = dest->ptr<double>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX)
				{
					for (; j < size.width; j += 4)//4 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const double* sptrbj = sptrb + j;
						const double* sptrgj = sptrg + j;
						const double* sptrrj = sptrr + j;

						const __m256d bval0 = _mm256_load_pd(sptrbj);
						const __m256d gval0 = _mm256_load_pd(sptrgj);
						const __m256d rval0 = _mm256_load_pd(sptrrj);

						__m256d wval = _mm256_setzero_pd();
						__m256d bval = _mm256_setzero_pd();
						__m256d gval = _mm256_setzero_pd();
						__m256d rval = _mm256_setzero_pd();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256d bref = _mm256_load_pd(sptrbj + *ofs);
							const __m256d gref = _mm256_load_pd(sptrgj + *ofs);
							const __m256d rref = _mm256_load_pd(sptrrj + *ofs);

							const __m256d bdiff = _mm256_sub_pd(bval0, bref);
							const __m256d gdiff = _mm256_sub_pd(gval0, gref);
							const __m256d rdiff = _mm256_sub_pd(rval0, rref);
							__m256d difft = _mm256_mul_pd(bdiff, bdiff);

#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_pd(gdiff, gdiff, difft);
							difft = _mm256_fmadd_pd(rdiff, rdiff, difft);
#else
							const __m256d gdiff_mul = _mm256_mul_pd(gdiff, gdiff);
							difft = _mm256_add_pd(difft, gdiff_mul);
							const __m256d rdiff_mul = _mm256_mul_pd(rdiff, rdiff);
							difft = _mm256_add_pd(difft, rdiff_mul);
#endif
							difft = _mm256_sqrt_pd(difft);
							const __m128i midx = _mm256_cvtpd_epi32(difft);
							__m256d _w = _mm256_i32gather_pd(range_weight, midx, 8);
#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_pd(bref, _w, bval);
							gval = _mm256_fmadd_pd(gref, _w, gval);
							rval = _mm256_fmadd_pd(rref, _w, rval);
#else
							const __m256d bref_mul = _mm256_mul_pd(bref, _w);
							bval = _mm256_add_pd(bval, bref_mul);
							const __m256d gref_mul = _mm256_mul_pd(gref, _w);
							gval = _mm256_add_pd(gval, gref_mul);
							const __m256d rref_mul = _mm256_mul_pd(rref, _w);
							rval = _mm256_add_pd(rval, rref_mul);
#endif
							wval = _mm256_add_pd(wval, _w);
						}
						bval = _mm256_div_pd(bval, wval);
						gval = _mm256_div_pd(gval, wval);
						rval = _mm256_div_pd(rval, wval);

						double* dptrc = dptr + 3 * j;
						_mm256_stream_pd_color(dptrc, bval, gval, rval);
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
						const double w = range_weight[(int)sqrt((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const double *range_weight;
};

class GaussianRangeFilter_QuantizationLUT_Gatherx1_32f_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationLUT_Gatherx1_32f_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float* _range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();

#if CV_AVX2
		static const int CV_DECL_ALIGNED(32) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		const bool haveAVX = checkHardwareSupport(CV_CPU_AVX2);
#endif
		if (cn == 1)
		{
			float* sptr = (float*)temp->ptr<float>(range.start + radiusV) + 8 * (radiusH / 8 + 1);
			float* dptr = dest->ptr<float>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX)
				{
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrj = sptr + j;
						const __m256 sval0 = _mm256_load_ps(sptrj);

						__m256 tval = _mm256_setzero_ps();
						__m256 wval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 sref = _mm256_loadu_ps((sptrj + *ofs));
							const __m256i midx = _mm256_cvtps_epi32(_mm256_and_ps(_mm256_sub_ps(sval0, sref), *(const __m256*)v32f_absmask));

							__m256 _w = _mm256_i32gather_ps(range_weight, midx, 4);
#if __USE_FMA_INSTRUCTION__
							tval = _mm256_fmadd_ps(sref, _w, tval);
#else
							const __m256 sref_mul = _mm256_mul_ps(sref, _w);
							tval = _mm256_add_ps(tval, sref_mul);
#endif
							wval = _mm256_add_ps(wval, _w);
						}
						tval = _mm256_div_ps(tval, wval);
						_mm256_stream_ps((dptr + j), tval);
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
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			float* sptrb = (float*)temp->ptr(3 * radiusV + 3 * range.start) + 8 * (radiusH / 8 + 1);
			float* sptrg = (float*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 8 * (radiusH / 8 + 1);
			float* sptrr = (float*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 8 * (radiusH / 8 + 1);

			float* dptr = dest->ptr<float>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX)
				{
					for (; j < size.width; j += 8)//8 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const float* sptrbj = sptrb + j;
						const float* sptrgj = sptrg + j;
						const float* sptrrj = sptrr + j;

						const __m256 bval0 = _mm256_load_ps(sptrbj);
						const __m256 gval0 = _mm256_load_ps(sptrgj);
						const __m256 rval0 = _mm256_load_ps(sptrrj);

						__m256 wval = _mm256_setzero_ps();
						__m256 bval = _mm256_setzero_ps();
						__m256 gval = _mm256_setzero_ps();
						__m256 rval = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256 bref = _mm256_load_ps(sptrbj + *ofs);
							const __m256 gref = _mm256_load_ps(sptrgj + *ofs);
							const __m256 rref = _mm256_load_ps(sptrrj + *ofs);

							const __m256 bdiff = _mm256_sub_ps(bval0, bref);
							const __m256 gdiff = _mm256_sub_ps(gval0, gref);
							const __m256 rdiff = _mm256_sub_ps(rval0, rref);
							__m256 difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							const __m256 gdiff_mul = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff_mul);
							const __m256 rdiff_mul = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff_mul);
#endif
							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							const __m256i midx = _mm256_cvtps_epi32(difft);
							__m256 _w = _mm256_i32gather_ps(range_weight, midx, 4);
#if __USE_FMA_INSTRUCTION__
							bval = _mm256_fmadd_ps(bref, _w, bval);
							gval = _mm256_fmadd_ps(gref, _w, gval);
							rval = _mm256_fmadd_ps(rref, _w, rval);
#else
							const __m256 bref_mul = _mm256_mul_ps(bref, _w);
							bval = _mm256_add_ps(bval, bref_mul);
							const __m256 gref_mul = _mm256_mul_ps(gref, _w);
							gval = _mm256_add_ps(gval, gref_mul);
							const __m256 rref_mul = _mm256_mul_ps(rref, _w);
							rval = _mm256_add_ps(rval, rref_mul);
#endif
							wval = _mm256_add_ps(wval, _w);
						}
						bval = _mm256_div_ps(bval, wval);
						gval = _mm256_div_ps(gval, wval);
						rval = _mm256_div_ps(rval, wval);

						float* dptrc = dptr + 3 * j;
						_mm256_stream_ps_color(dptrc, bval, gval, rval);
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
						const float w = range_weight[(int)sqrt((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
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
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *range_weight;
};

class GaussianRangeFilter_QuantizationLUT_Gatherx1_8u_InvokerAVX2 : public cv::ParallelLoopBody
{
public:
	GaussianRangeFilter_QuantizationLUT_Gatherx1_8u_InvokerAVX2(Mat& _dest, const Mat& _temp, const int _radiusH, const int _radiusV, const int _maxk, const int* _space_ofs, const float* _range_weight) :
		temp(&_temp), dest(&_dest), radiusH(_radiusH), radiusV(_radiusV), maxk(_maxk), space_ofs(_space_ofs), range_weight(_range_weight)
	{
	}

	void operator()(const Range& range) const override
	{
		int i, j, k;
		const int cn = dest->channels();
		const Size size = dest->size();
#if CV_AVX2
		const bool haveAVX2 = checkHardwareSupport(CV_CPU_AVX2);
#endif
		if (cn == 1)
		{
			uchar* sptr = (uchar*)temp->ptr<uchar>(range.start + radiusV) + 32 * (radiusH / 32 + 1);
			uchar* dptr = dest->ptr<uchar>(range.start);

			const int sstep = temp->cols;
			const int dstep = dest->cols;

			for (i = range.start; i != range.end; i++, dptr += dstep, sptr += sstep)
			{
				j = 0;
#if CV_AVX2
				uchar CV_DECL_ALIGNED(32) buf[32];
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrj = sptr + j;
						const __m256i sval0_32elem = _mm256_load_si256((__m256i*)sptrj);

						__m256 tval1 = _mm256_setzero_ps();
						__m256 wval1 = _mm256_setzero_ps();
						__m256 tval2 = _mm256_setzero_ps();
						__m256 wval2 = _mm256_setzero_ps();
						__m256 tval3 = _mm256_setzero_ps();
						__m256 wval3 = _mm256_setzero_ps();
						__m256 tval4 = _mm256_setzero_ps();
						__m256 wval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i sref_32elem = _mm256_loadu_si256((__m256i*)(sptrj + *ofs));
							const __m256i diff_32elem = _mm256_add_epi8(_mm256_subs_epu8(sval0_32elem, sref_32elem), _mm256_subs_epu8(sref_32elem, sval0_32elem));
							_mm256_store_si256((__m256i*)buf, _mm256_add_epi8(_mm256_subs_epu8(sval0_32elem, sref_32elem), _mm256_subs_epu8(sref_32elem, sval0_32elem)));

							// lower bit
							__m256i sref_16elem = _mm256_unpacklo_epi8(sref_32elem, zero);
							__m256i diff_16elem = _mm256_unpacklo_epi8(diff_32elem, zero);
							__m256i midx = _mm256_unpacklo_epi16(diff_16elem, zero);
							__m256 _w = _mm256_i32gather_ps(range_weight, midx, 4);
							__m256 sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval1 = _mm256_fmadd_ps(sref, _w, tval1);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval1 = _mm256_add_ps(tval1, sref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


							midx = _mm256_unpackhi_epi16(diff_16elem, zero);
							_w = _mm256_i32gather_ps(range_weight, midx, 4);
							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval2 = _mm256_fmadd_ps(sref, _w, tval2);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval2 = _mm256_add_ps(tval2, sref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// higher bit
							diff_16elem = _mm256_unpackhi_epi8(diff_32elem, zero);
							midx = _mm256_unpacklo_epi16(diff_16elem, zero);
							_w = _mm256_i32gather_ps(range_weight, midx, 4);
							sref_16elem = _mm256_unpackhi_epi8(sref_32elem, zero);
							sref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval3 = _mm256_fmadd_ps(sref, _w, tval3);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval3 = _mm256_add_ps(tval3, sref);
#endif
							wval3 = _mm256_add_ps(wval3, _w);

							midx = _mm256_unpackhi_epi16(diff_16elem, zero);
							_w = _mm256_i32gather_ps(range_weight, midx, 4);
							sref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(sref_16elem, zero));
#if __USE_FMA_INSTRUCTION__
							tval4 = _mm256_fmadd_ps(sref, _w, tval4);
#else
							sref = _mm256_mul_ps(sref, _w);
							tval4 = _mm256_add_ps(tval4, sref);
#endif
							wval4 = _mm256_add_ps(wval4, _w);
						}
						tval1 = _mm256_div_ps(tval1, wval1);
						tval2 = _mm256_div_ps(tval2, wval2);
						tval3 = _mm256_div_ps(tval3, wval3);
						tval4 = _mm256_div_ps(tval4, wval4);
						_mm256_stream_si256((__m256i*)(dptr + j), _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(tval1), _mm256_cvtps_epi32(tval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(tval3), _mm256_cvtps_epi32(tval4))));
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
					dptr[j] = (int)(sum / wsum);
				}
			}
		}
		else
		{
			const int sstep = 3 * temp->cols;
			const int dstep = 3 * dest->cols;
			uchar* sptrb = (uchar*)temp->ptr(3 * radiusV + 3 * range.start) + 32 * (radiusH / 32 + 1);
			uchar* sptrg = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 1) + 32 * (radiusH / 32 + 1);
			uchar* sptrr = (uchar*)temp->ptr(3 * radiusV + 3 * range.start + 2) + 32 * (radiusH / 32 + 1);

			uchar* dptr = dest->ptr<uchar>(range.start);
			for (i = range.start; i != range.end; i++, sptrr += sstep, sptrg += sstep, sptrb += sstep, dptr += dstep)
			{
				j = 0;
#if CV_AVX2
				if (haveAVX2)
				{
					static const __m256i zero = _mm256_setzero_si256();
					for (; j < size.width; j += 32)//32 pixel unit
					{
						const int* ofs = &space_ofs[0];

						const uchar* sptrbj = sptrb + j;
						const uchar* sptrgj = sptrg + j;
						const uchar* sptrrj = sptrr + j;

						const __m256i bval0_32elem = _mm256_load_si256((__m256i*)(sptrbj));
						const __m256i gval0_32elem = _mm256_load_si256((__m256i*)(sptrgj));
						const __m256i rval0_32elem = _mm256_load_si256((__m256i*)(sptrrj));

						__m256i temp = _mm256_unpacklo_epi8(bval0_32elem, zero);
						const __m256 bval0_1 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 bval0_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpackhi_epi8(bval0_32elem, zero);
						const __m256 bval0_3 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 bval0_4 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpacklo_epi8(gval0_32elem, zero);
						const __m256 gval0_1 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 gval0_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpackhi_epi8(gval0_32elem, zero);
						const __m256 gval0_3 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 gval0_4 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpacklo_epi8(rval0_32elem, zero);
						const __m256 rval0_1 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 rval0_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));
						temp = _mm256_unpackhi_epi8(rval0_32elem, zero);
						const __m256 rval0_3 = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(temp, zero));
						const __m256 rval0_4 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(temp, zero));

						__m256 wval1 = _mm256_setzero_ps();
						__m256 bval1 = _mm256_setzero_ps();
						__m256 gval1 = _mm256_setzero_ps();
						__m256 rval1 = _mm256_setzero_ps();

						__m256 wval2 = _mm256_setzero_ps();
						__m256 bval2 = _mm256_setzero_ps();
						__m256 gval2 = _mm256_setzero_ps();
						__m256 rval2 = _mm256_setzero_ps();

						__m256 wval3 = _mm256_setzero_ps();
						__m256 bval3 = _mm256_setzero_ps();
						__m256 gval3 = _mm256_setzero_ps();
						__m256 rval3 = _mm256_setzero_ps();

						__m256 wval4 = _mm256_setzero_ps();
						__m256 bval4 = _mm256_setzero_ps();
						__m256 gval4 = _mm256_setzero_ps();
						__m256 rval4 = _mm256_setzero_ps();

						for (k = 0; k < maxk; k++, ofs++)
						{
							const __m256i bref_32elem = _mm256_loadu_si256((__m256i*)(sptrbj + *ofs));
							const __m256i gref_32elem = _mm256_loadu_si256((__m256i*)(sptrgj + *ofs));
							const __m256i rref_32elem = _mm256_loadu_si256((__m256i*)(sptrrj + *ofs));

							// low
							__m256i bref_16elem = _mm256_unpacklo_epi8(bref_32elem, zero);
							__m256i gref_16elem = _mm256_unpacklo_epi8(gref_32elem, zero);
							__m256i rref_16elem = _mm256_unpacklo_epi8(rref_32elem, zero);

							__m256 bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							__m256 gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							__m256 rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));

							__m256 bdiff = _mm256_sub_ps(bval0_1, bref);
							__m256 gdiff = _mm256_sub_ps(gval0_1, gref);
							__m256 rdiff = _mm256_sub_ps(rval0_1, rref);
							__m256 difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							__m256i midx = _mm256_cvtps_epi32(difft);

							__m256 _w = _mm256_i32gather_ps(range_weight, midx, 4);
#if __USE_FMA_INSTRUCTION__
							bval1 = _mm256_fmadd_ps(bref, _w, bval1);
							gval1 = _mm256_fmadd_ps(gref, _w, gval1);
							rval1 = _mm256_fmadd_ps(rref, _w, rval1);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval1 = _mm256_add_ps(bval1, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval1 = _mm256_add_ps(gval1, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval1 = _mm256_add_ps(rval1, rref);
#endif
							wval1 = _mm256_add_ps(wval1, _w);


							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));

							bdiff = _mm256_sub_ps(bval0_2, bref);
							gdiff = _mm256_sub_ps(gval0_2, gref);
							rdiff = _mm256_sub_ps(rval0_2, rref);
							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							midx = _mm256_cvtps_epi32(difft);
							_w = _mm256_i32gather_ps(range_weight, midx, 4);
#if __USE_FMA_INSTRUCTION__
							bval2 = _mm256_fmadd_ps(bref, _w, bval2);
							gval2 = _mm256_fmadd_ps(gref, _w, gval2);
							rval2 = _mm256_fmadd_ps(rref, _w, rval2);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval2 = _mm256_add_ps(bval2, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval2 = _mm256_add_ps(gval2, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval2 = _mm256_add_ps(rval2, rref);
#endif
							wval2 = _mm256_add_ps(wval2, _w);


							// high
							bref_16elem = _mm256_unpackhi_epi8(bref_32elem, zero);
							gref_16elem = _mm256_unpackhi_epi8(gref_32elem, zero);
							rref_16elem = _mm256_unpackhi_epi8(rref_32elem, zero);

							bref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(rref_16elem, zero));

							bdiff = _mm256_sub_ps(bval0_3, bref);
							gdiff = _mm256_sub_ps(gval0_3, gref);
							rdiff = _mm256_sub_ps(rval0_3, rref);
							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							midx = _mm256_cvtps_epi32(difft);
							_w = _mm256_i32gather_ps(range_weight, midx, 4);
#if __USE_FMA_INSTRUCTION__
							bval3 = _mm256_fmadd_ps(bref, _w, bval3);
							gval3 = _mm256_fmadd_ps(gref, _w, gval3);
							rval3 = _mm256_fmadd_ps(rref, _w, rval3);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval3 = _mm256_add_ps(bval3, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval3 = _mm256_add_ps(gval3, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval3 = _mm256_add_ps(rval3, rref);
#endif
							wval3 = _mm256_add_ps(wval3, _w);


							bref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(bref_16elem, zero));
							gref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(gref_16elem, zero));
							rref = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(rref_16elem, zero));

							bdiff = _mm256_sub_ps(bval0_4, bref);
							gdiff = _mm256_sub_ps(gval0_4, gref);
							rdiff = _mm256_sub_ps(rval0_4, rref);
							difft = _mm256_mul_ps(bdiff, bdiff);
#if __USE_FMA_INSTRUCTION__
							difft = _mm256_fmadd_ps(gdiff, gdiff, difft);
							difft = _mm256_fmadd_ps(rdiff, rdiff, difft);
#else
							gdiff = _mm256_mul_ps(gdiff, gdiff);
							difft = _mm256_add_ps(difft, gdiff);
							rdiff = _mm256_mul_ps(rdiff, rdiff);
							difft = _mm256_add_ps(difft, rdiff);
#endif
							difft = _mm256_rcp_ps(_mm256_rsqrt_ps(difft));
							midx = _mm256_cvtps_epi32(difft);
							_w = _mm256_i32gather_ps(range_weight, midx, 4);
#if __USE_FMA_INSTRUCTION__
							bval4 = _mm256_fmadd_ps(bref, _w, bval4);
							gval4 = _mm256_fmadd_ps(gref, _w, gval4);
							rval4 = _mm256_fmadd_ps(rref, _w, rval4);
#else
							bref = _mm256_mul_ps(bref, _w);
							bval4 = _mm256_add_ps(bval4, bref);
							gref = _mm256_mul_ps(gref, _w);
							gval4 = _mm256_add_ps(gval4, gref);
							rref = _mm256_mul_ps(rref, _w);
							rval4 = _mm256_add_ps(rval4, rref);
#endif
							wval4 = _mm256_add_ps(wval4, _w);
						}
						bval1 = _mm256_div_ps(bval1, wval1);
						bval2 = _mm256_div_ps(bval2, wval2);
						bval3 = _mm256_div_ps(bval3, wval3);
						bval4 = _mm256_div_ps(bval4, wval4);
						const __m256i a = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(bval1), _mm256_cvtps_epi32(bval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(bval3), _mm256_cvtps_epi32(bval4)));
						gval1 = _mm256_div_ps(gval1, wval1);
						gval2 = _mm256_div_ps(gval2, wval2);
						gval3 = _mm256_div_ps(gval3, wval3);
						gval4 = _mm256_div_ps(gval4, wval4);
						const __m256i b = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(gval1), _mm256_cvtps_epi32(gval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(gval3), _mm256_cvtps_epi32(gval4)));
						rval1 = _mm256_div_ps(rval1, wval1);
						rval2 = _mm256_div_ps(rval2, wval2);
						rval3 = _mm256_div_ps(rval3, wval3);
						rval4 = _mm256_div_ps(rval4, wval4);
						const __m256i c = _mm256_packus_epi16(_mm256_packs_epi32(_mm256_cvtps_epi32(rval1), _mm256_cvtps_epi32(rval2)), _mm256_packs_epi32(_mm256_cvtps_epi32(rval3), _mm256_cvtps_epi32(rval4)));

						uchar* dptrc = dptr + 3 * j;
						_mm256_stream_epi8_color(dptrc, a, b, c);
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
						const float w = range_weight[(int)sqrt((b - b0) * (b - b0) + (g - g0) * (g - g0) + (r - r0) * (r - r0))];
						sum_b += b * w;
						sum_g += g * w;
						sum_r += r * w;
						wsum += w;
					}
					dptr[3 * j + 0] = cvRound(sum_b / wsum);
					dptr[3 * j + 1] = cvRound(sum_g / wsum);
					dptr[3 * j + 2] = cvRound(sum_r / wsum);
				}
			}
		}
	}

private:
	const Mat* temp;

	Mat* dest;
	int radiusH, radiusV, maxk;
	const int* space_ofs;
	const float *range_weight;
};


namespace grf
{
	///////////////////////////////////////////////
	void gaussianRangeFilter_AVX_64f(const Mat& src, Mat& dst, Size kernelSize, double sigma_range, const int borderType, const bool isRectangle, const grf::WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0)
		{
			src.copyTo(dst);
			return;
		}
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_64FC1 || src.type() == CV_64FC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
		{
			sigma_range = 1.f;
		}

		const double gauss_range_coeff = -0.5 / (sigma_range * sigma_range);

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
			splitBGRLineInterleaveAVX(temp2, temp);
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
			const GaussianRangeFilter_EXP_64f_InvokerAVX body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range);
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

			const GaussianRangeFilter_LUT_Setx3_64f_InvokerAVX body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
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

			const GaussianRangeFilter_LUT_Gatherx3_64f_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 442;
			vector<double> _weight(value_range);
			double* range_weight = &_weight[0];

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

			const GaussianRangeFilter_QuantizationLUT_Setx1_64f_InvokerAVX body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHER:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 442;
			vector<double> _weight(value_range);
			double* range_weight = &_weight[0];

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

			const GaussianRangeFilter_QuantizationLUT_Gatherx1_64f_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		default:
			break;
		}
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void gaussianRangeFilter_AVX_32f(const Mat& src, Mat& dst, Size kernelSize, float sigma_range, const int borderType, const bool isRectangle, const WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0)
		{
			src.copyTo(dst);
			return;
		}
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_32FC1 || src.type() == CV_32FC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
		{
			sigma_range = 1.f;
		}

		const float gauss_range_coeff = -0.5f / (sigma_range * sigma_range);

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

		vector<int> _space_ofs(kernelSize.area());
		int* space_ofs = &_space_ofs[0];
		int maxk = 0;
		setSpaceKernel(space_ofs, maxk, radiusH, radiusV, temp.cols * cn, isRectangle);

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), dst.type());
		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			const GaussianRangeFilter_EXP_32f_InvokerAVX body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range);
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

			const GaussianRangeFilter_LUT_Setx3_32f_InvokerAVX body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
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

			const GaussianRangeFilter_LUT_Gatherx3_32f_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 442;
			vector<float> _weight(value_range);
			float* range_weight = &_weight[0];

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

			const GaussianRangeFilter_QuantizationLUT_Setx1_32f_InvokerAVX body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHER:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 442;
			vector<float> _weight(value_range);
			float* range_weight = &_weight[0];

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

			const GaussianRangeFilter_QuantizationLUT_Gatherx1_32f_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		default:
			break;
		}
		Mat(dest(Rect(0, 0, dst.cols, dst.rows))).copyTo(dst);
	}

	void gaussianRangeFilter_AVX_8u(const Mat& src, Mat& dst, const Size kernelSize, float sigma_range, const int borderType, const bool isRectangle, const WEIGHT_MODE weightingMethod)
	{
		if (kernelSize.width == 0 || kernelSize.height == 0)
		{
			src.copyTo(dst);
			return;
		}
		const int cn = src.channels();
		const Size size = src.size();

		CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) && src.type() == dst.type() && src.size() == dst.size());

		if (sigma_range <= 0)
			sigma_range = 1.;

		const float gauss_range_coeff = -0.5f / (sigma_range * sigma_range);

		const int radiusH = kernelSize.width >> 1;
		const int radiusV = kernelSize.height >> 1;

		Mat temp;

		const int dpad = (32 - src.cols % 32) % 32;
		int spad = dpad + (32 - (2 * radiusH) % 32) % 32;
		if (spad < 32) spad += 32;
		const int lpad = 32 * (radiusH / 32 + 1) - radiusH;
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

		vector<int> _space_ofs(kernelSize.area());
		int* space_ofs = &_space_ofs[0];
		int maxk = 0;
		setSpaceKernel(space_ofs, maxk, radiusH, radiusV, temp.cols * cn, isRectangle);

		Mat dest = Mat::zeros(Size(src.cols + dpad, src.rows), src.type());
		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			const GaussianRangeFilter_EXP_8u_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, sigma_range);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 256;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			// initialize color-related gaussianRange filter coefficients
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

			const GaussianRangeFilter_LUT_Setx3_8u_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 256;
			vector<float> _range_weight(value_range);
			float* range_weight = &_range_weight[0];

			// initialize color-related gaussianRange filter coefficients
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

			const GaussianRangeFilter_LUT_Gatherx3_8u_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight, bias);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SET:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 442;
			vector<float> _weight(value_range);
			float* range_weight = &_weight[0];

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

			const GaussianRangeFilter_QuantizationLUT_Setx1_8u_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
			parallel_for_(Range(0, size.height), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHER:
		{
			// initialize space-related gaussianRange filter coefficients
			const int value_range = 442;
			vector<float> _weight(value_range);
			float* range_weight = &_weight[0];

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

			const GaussianRangeFilter_QuantizationLUT_Gatherx1_8u_InvokerAVX2 body(dest, temp, radiusH, radiusV, maxk, space_ofs, range_weight);
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
	void gaussianRangeFilter_AVX(cv::InputArray src_, cv::OutputArray dest, const cv::Size kernelSize, const double sigma_range, const int kernel_type, const int borderType, const WEIGHT_MODE weightingMethod)
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
				gaussianRangeFilter_AVX_8u(src, dst, kernelSize, sigma_range, borderType, true, weightingMethod);
			}
			else if (src.depth() == CV_32F)
			{
				gaussianRangeFilter_AVX_32f(src, dst, kernelSize, sigma_range, borderType, true, weightingMethod);
			}
			else if (src.depth() == CV_64F)
			{
				gaussianRangeFilter_AVX_64f(src, dst, kernelSize, sigma_range, borderType, true, weightingMethod);
			}

			break;
		}
		case FILTER_CIRCLE:
		{
			if (src.depth() == CV_8U)
			{
				gaussianRangeFilter_AVX_8u(src, dst, kernelSize, sigma_range, borderType, false, weightingMethod);
			}
			else if (src.depth() == CV_32F)
			{
				gaussianRangeFilter_AVX_32f(src, dst, kernelSize, sigma_range, borderType, false, weightingMethod);
			}
			else if (src.depth() == CV_64F)
			{
				gaussianRangeFilter_AVX_64f(src, dst, kernelSize, sigma_range, borderType, false, weightingMethod);
			}
			break;
		}
		case FILTER_SEPARABLE:
		default:
			break;
		}
	}
}
