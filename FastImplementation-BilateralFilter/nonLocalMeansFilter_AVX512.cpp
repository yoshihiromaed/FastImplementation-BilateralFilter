#include "color.h"
#include "filter.h"
#include "fmath.hpp"
#include "simd_util.h"

using namespace std;
using namespace cv;


/*************************************************
	using exp function
*************************************************/
class NonlocalMeansFilterInvorker_EXP_64f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const double h;

public:
	NonlocalMeansFilterInvorker_EXP_64f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const double h, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_maxk(search_maxk), search_ofs(search_ofs), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), h(h)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

		const double gauss_color_coeff = -(1.0 / (h*h));

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		__m512d* tVec = (__m512d*)_mm_malloc(sizeof(__m512d)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d gcc = _mm512_set1_pd(gauss_color_coeff);
#if __NLMF_PREVENTION__
					static const __m512d exp_arg_min = _mm512_set1_pd(EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mb = _mm512_setzero_pd();
						__m512d mg = _mm512_setzero_pd();
						__m512d mr = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d me = _mm512_setzero_pd();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512d s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								s0 = _mm512_mul_pd(s0, s0);
								me = _mm512_add_pd(me, s0);
#endif

								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								s0 = _mm512_mul_pd(s0, s0);
								me = _mm512_add_pd(me, s0);
#endif

								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								s0 = _mm512_mul_pd(s0, s0);
								me = _mm512_add_pd(me, s0);
#endif
							}
							__m512d aw = _mm512_mul_pd(me, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_pd(aw, exp_arg_min);
#endif
							__m512d mw =
#if __USE_INTEL_EXP__
								_mm512_exp_pd(aw);
#else
								//fmath::exp_pd(aw); //TODO: exp_pd
								_mm512_set1_pd(1);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_pd(mw, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mb);
							mg = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep), mg);
							mr = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep2), mr);
#else
							mb = _mm512_add_pd(mb, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
							mg = _mm512_add_pd(mg, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep)));
							mr = _mm512_add_pd(mr, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep2)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}

						mb = _mm512_div_pd(mb, mtweight);
						mg = _mm512_div_pd(mg, mtweight);
						mr = _mm512_div_pd(mr, mtweight);

						_mm512_stream_pd_color(d, mb, mg, mr);
						d += 24;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sumb = 0;
					double sumg = 0;
					double sumr = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							double s0 = *(s + t_ofs[n]) - tV[count++];
							e += s0 * s0;
							s0 = *(s + t_ofs[n] + colorstep) - tV[count++];
							e += s0 * s0;
							s0 = *(s + t_ofs[n] + colorstep2) - tV[count++];
							e += s0 * s0;
						}
						double w = exp(e*gauss_color_coeff);
						sumb += *(s)* w;
						sumg += *(s + colorstep) * w;
						sumr += *(s + colorstep2) * w;
						sumw += w;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512d gcc = _mm512_set1_pd(gauss_color_coeff);
#if __NLMF_PREVENTION__
					static const __m512d exp_arg_min = _mm512_set1_pd(EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mval = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d me = _mm512_setzero_pd();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512d s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								s0 = _mm512_mul_pd(s0, s0);
								me = _mm512_add_pd(me, s0);
#endif
							}
							__m512d aw = _mm512_mul_pd(me, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_pd(aw, exp_arg_min);
#endif
							__m512d mw =
#if __USE_INTEL_EXP__
								_mm512_exp_pd(aw);
#else
								//fmath::exp_pd(aw); //TODO: exp_pd
								_mm512_set1_pd(1);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_pd(mw, double_min);
#endif
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mval);
#else
							mval = _mm512_add_pd(mval, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}
						_mm512_stream_pd(d, _mm512_div_pd(mval, mtweight));
						d += 8;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sum = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							double s0 = *(s + t_ofs[n]) - tV[count++];
							e += s0 * s0;
						}
						const double w = exp(e*gauss_color_coeff);
						sum += *(s)* w;
						sumw += w;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_EXP_32f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float h;

public:
	NonlocalMeansFilterInvorker_EXP_32f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float h, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), h(h)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;
		const float gauss_color_coeff = -(1.0 / (h*h));

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 gcc = _mm512_set1_ps(gauss_color_coeff);
#if __NLMF_PREVENTION__
					static const __m512 exp_arg_min = _mm512_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mb = _mm512_setzero_ps();
						__m512 mg = _mm512_setzero_ps();
						__m512 mr = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512 s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								s0 = _mm512_mul_ps(s0, s0);
								me = _mm512_add_ps(me, s0);
#endif

								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								s0 = _mm512_mul_ps(s0, s0);
								me = _mm512_add_ps(me, s0);
#endif

								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								s0 = _mm512_mul_ps(s0, s0);
								me = _mm512_add_ps(me, s0);
#endif
							}
							__m512 aw = _mm512_mul_ps(me, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mb);
							mg = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep), mg);
							mr = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep2), mr);
#else
							mb = _mm512_add_ps(mb, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
							mg = _mm512_add_ps(mg, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep)));
							mr = _mm512_add_ps(mr, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep2)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}

						mb = _mm512_div_ps(mb, mtweight);
						mg = _mm512_div_ps(mg, mtweight);
						mr = _mm512_div_ps(mr, mtweight);

						_mm512_stream_ps_color(d, mb, mg, mr);
						d += 48;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = *(s + t_ofs[n]) - tV[count++];
							e += s0 * s0;
							s0 = *(s + t_ofs[n] + colorstep) - tV[count++];
							e += s0 * s0;
							s0 = *(s + t_ofs[n] + colorstep2) - tV[count++];
							e += s0 * s0;
						}
						float w = exp(e*gauss_color_coeff);
						sumb += *(s)* w;
						sumg += *(s + colorstep) * w;
						sumr += *(s + colorstep2) * w;
						sumw += w;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 gcc = _mm512_set1_ps(gauss_color_coeff);
#if __NLMF_PREVENTION__
					static const __m512 exp_arg_min = _mm512_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mval = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512 s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								s0 = _mm512_mul_ps(s0, s0);
								me = _mm512_add_ps(me, s0);
#endif
							}
							__m512 aw = _mm512_mul_ps(me, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mval);
#else
							mval = _mm512_add_ps(mval, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}
						_mm512_stream_ps(d, _mm512_div_ps(mval, mtweight));
						d += 16;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sum = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = *(s + t_ofs[n]) - tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff);
						sum += *(s)* w;
						sumw += w;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_EXP_8u_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float h;

public:
	NonlocalMeansFilterInvorker_EXP_8u_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float h, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), h(h)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;
		const float gauss_color_coeff = -(1.0 / (h*h));

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels() * 4, 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 gcc = _mm512_set1_ps(gauss_color_coeff);
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_PREVENTION__
					static const __m512 exp_arg_min = _mm512_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mb0 = _mm512_setzero_ps();
						__m512 mb1 = _mm512_setzero_ps();
						__m512 mb2 = _mm512_setzero_ps();
						__m512 mb3 = _mm512_setzero_ps();

						__m512 mg0 = _mm512_setzero_ps();
						__m512 mg1 = _mm512_setzero_ps();
						__m512 mg2 = _mm512_setzero_ps();
						__m512 mg3 = _mm512_setzero_ps();

						__m512 mr0 = _mm512_setzero_ps();
						__m512 mr1 = _mm512_setzero_ps();
						__m512 mr2 = _mm512_setzero_ps();
						__m512 mr3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							__m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep2));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me0 = _mm512_setzero_ps();
							__m512 me1 = _mm512_setzero_ps();
							__m512 me2 = _mm512_setzero_ps();
							__m512 me3 = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								//b
								__m512i s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								__m512 s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif

								//g
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif

								//r
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep2));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif
							}

							//low low
							__m512 aw = _mm512_mul_ps(me0, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
							//b
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mb0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw, mss, mb0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw, mss), mb0);
#endif
							//g
							const __m512i ssg_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep));
							__m512i ssg_32elem = _mm512_unpacklo_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg0 = _mm512_fmadd_ps(mw, mss, mg0);
#else
							mg0 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg0);
#endif
							//r
							const __m512i ssr_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep2));
							__m512i ssr_32elem = _mm512_unpacklo_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr0 = _mm512_fmadd_ps(mw, mss, mr0);
#else
							mr0 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw);


							//low high
							aw = _mm512_mul_ps(me1, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb1 = _mm512_fmadd_ps(mw, mss, mb1);
#else
							mb1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb1);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg1 = _mm512_fmadd_ps(mw, mss, mg1);
#else
							mg1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg1);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr1 = _mm512_fmadd_ps(mw, mss, mr1);
#else
							mr1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw);


							//high low
							aw = _mm512_mul_ps(me2, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
							//b
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb2 = _mm512_fmadd_ps(mw, mss, mb2);
#else
							mb2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb2);
#endif
							//g
							ssg_32elem = _mm512_unpackhi_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg2 = _mm512_fmadd_ps(mw, mss, mg2);
#else
							mg2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg2);
#endif
							//r
							ssr_32elem = _mm512_unpackhi_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr2 = _mm512_fmadd_ps(mw, mss, mr2);
#else
							mr2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw);


							//high high
							aw = _mm512_mul_ps(me3, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb3 = _mm512_fmadd_ps(mw, mss, mb3);
#else
							mb3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb3);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg3 = _mm512_fmadd_ps(mw, mss, mg3);
#else
							mg3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg3);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr3 = _mm512_fmadd_ps(mw, mss, mr3);
#else
							mr3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw);
						}
						mb0 = _mm512_div_ps(mb0, mtweight0);
						mb1 = _mm512_div_ps(mb1, mtweight1);
						mb2 = _mm512_div_ps(mb2, mtweight2);
						mb3 = _mm512_div_ps(mb3, mtweight3);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mb0), _mm512_cvtps_epi32(mb1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mb2), _mm512_cvtps_epi32(mb3)));
						mg0 = _mm512_div_ps(mg0, mtweight0);
						mg1 = _mm512_div_ps(mg1, mtweight1);
						mg2 = _mm512_div_ps(mg2, mtweight2);
						mg3 = _mm512_div_ps(mg3, mtweight3);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mg0), _mm512_cvtps_epi32(mg1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mg2), _mm512_cvtps_epi32(mg3)));
						mr0 = _mm512_div_ps(mr0, mtweight0);
						mr1 = _mm512_div_ps(mr1, mtweight1);
						mr2 = _mm512_div_ps(mr2, mtweight2);
						mr3 = _mm512_div_ps(mr3, mtweight3);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mr0), _mm512_cvtps_epi32(mr1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mr2), _mm512_cvtps_epi32(mr3)));

						_mm512_stream_epi8_color(d, a, b, c);
						d += 192;
					}
				}
#endif
				for (; i < dest->cols; i++)
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep) - tV[count++];
							e += s0 * s0;
							s0 = (float)*(s + t_ofs[n] + colorstep2) - tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff);
						sumb += *(s)* w;
						sumg += *(s + colorstep) * w;
						sumr += *(s + colorstep2) * w;
						sumw += w;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					const __m512 gcc = _mm512_set1_ps(gauss_color_coeff);
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_PREVENTION__
					static const __m512 exp_arg_min = _mm512_set1_ps(EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mval0 = _mm512_setzero_ps();
						__m512 mval1 = _mm512_setzero_ps();
						__m512 mval2 = _mm512_setzero_ps();
						__m512 mval3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							const __m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me0 = _mm512_setzero_ps();
							__m512 me1 = _mm512_setzero_ps();
							__m512 me2 = _mm512_setzero_ps();
							__m512 me3 = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512i s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								__m512 s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif
							}

							//low low
							__m512 aw = _mm512_mul_ps(me0, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							__m512 mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mval0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw, mss, mval0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw, mss), mval0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw);

							//low high
							aw = _mm512_mul_ps(me1, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval1 = _mm512_fmadd_ps(mw, mss, mval1);
#else
							mval1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw);

							//high low
							aw = _mm512_mul_ps(me2, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval2 = _mm512_fmadd_ps(mw, mss, mval2);
#else
							mval2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw);


							//high high
							aw = _mm512_mul_ps(me3, gcc);
#if __NLMF_PREVENTION__
							aw = _mm512_max_ps(aw, exp_arg_min);
#endif
							mw =
#if __USE_INTEL_EXP__
								_mm512_exp_ps(aw);
#else
								fmath::exp_ps512(aw);
#endif
#if __NLMF_POSTVENTION__
							mw = _mm512_max_ps(mw, float_min);
#endif
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval3 = _mm512_fmadd_ps(mw, mss, mval3);
#else
							mval3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw);
						}
						mval0 = _mm512_div_ps(mval0, mtweight0);
						mval1 = _mm512_div_ps(mval1, mtweight1);
						mval2 = _mm512_div_ps(mval2, mtweight2);
						mval3 = _mm512_div_ps(mval3, mtweight3);
						_mm512_stream_si512((__m512i*)d, _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mval0), _mm512_cvtps_epi32(mval1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mval2), _mm512_cvtps_epi32(mval3))));
						d += 64;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							float s0 = (float)*(s + t_ofs[n]) - tV[count++];
							e += s0 * s0;
						}
						const float w = exp(e*gauss_color_coeff);
						sumb += *(s)* w;
						sumw += w;
					}
					d[0] = sumb / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};


/*************************************************
	using LUT with "set instruction" x 3
*************************************************/
class NonlocalMeansFilterInvorker_LUT_Setx3_64f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const double* weight;
	const double offset;

public:
	NonlocalMeansFilterInvorker_LUT_Setx3_64f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const double* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const double offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_maxk(search_maxk), search_ofs(search_ofs), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const long long CV_DECL_ALIGNED(64) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff,
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff
		};
		long long CV_DECL_ALIGNED(64) buf[8];
		__m512d* tVec = (__m512d*)_mm_malloc(sizeof(__m512d)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const double* w = weight;
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mb = _mm512_setzero_pd();
						__m512d mg = _mm512_setzero_pd();
						__m512d mr = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d mw = _mm512_set1_pd(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]), *(__m512d const*)v64f_absmask)));
								mw = _mm512_mul_pd(mw, _mm512_set_pd(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep), tVec[count++]), *(__m512d const*)v64f_absmask)));
								mw = _mm512_mul_pd(mw, _mm512_set_pd(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep2), tVec[count++]), *(__m512d const*)v64f_absmask)));
								mw = _mm512_mul_pd(mw, _mm512_set_pd(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mb);
							mg = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep), mg);
							mr = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep2), mr);
#else
							mb = _mm512_add_pd(mb, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
							mg = _mm512_add_pd(mg, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep)));
							mr = _mm512_add_pd(mr, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep2)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}

						mb = _mm512_div_pd(mb, mtweight);
						mg = _mm512_div_pd(mg, mtweight);
						mr = _mm512_div_pd(mr, mtweight);

						_mm512_stream_pd_color(d, mb, mg, mr);
						d += 24;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sumb = 0;
					double sumg = 0;
					double sumr = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
							s0 = abs(*(s + t_ofs[n] + colorstep) - tV[count++]);
							e *= w[s0];
							s0 = abs(*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e *= w[s0];
						}
						sumb += *(s)* e;
						sumg += *(s + colorstep) * e;
						sumr += *(s + colorstep2) * e;
						sumw += e;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const double* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mval = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d mw = _mm512_set1_pd(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]), *(__m512d const*)v64f_absmask)));
								mw = _mm512_mul_pd(mw, _mm512_set_pd(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mval);
#else
							mval = _mm512_add_pd(mval, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}
						_mm512_stream_pd(d, _mm512_div_pd(mval, mtweight));
						d += 8;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sum = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sum += *(s)* e;
						sumw += e;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_LUT_Setx3_32f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;
	const float offset;

public:
	NonlocalMeansFilterInvorker_LUT_Setx3_32f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const float offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		int CV_DECL_ALIGNED(64) buf[16];
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mb = _mm512_setzero_ps();
						__m512 mg = _mm512_setzero_ps();
						__m512 mr = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw = _mm512_mul_ps(mw, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep), tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw = _mm512_mul_ps(mw, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep2), tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw = _mm512_mul_ps(mw, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mb);
							mg = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep), mg);
							mr = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep2), mr);
#else
							mb = _mm512_add_ps(mb, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
							mg = _mm512_add_ps(mg, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep)));
							mr = _mm512_add_ps(mr, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep2)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}

						mb = _mm512_div_ps(mb, mtweight);
						mg = _mm512_div_ps(mg, mtweight);
						mr = _mm512_div_ps(mr, mtweight);

						_mm512_stream_ps_color(d, mb, mg, mr);
						d += 48;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
							s0 = abs(*(s + t_ofs[n] + colorstep) - tV[count++]);
							e *= w[s0];
							s0 = abs(*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e *= w[s0];
						}
						sumb += *(s)* e;
						sumg += *(s + colorstep) * e;
						sumr += *(s + colorstep2) * e;
						sumw += e;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mval = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw = _mm512_mul_ps(mw, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mval);
#else
							mval = _mm512_add_ps(mval, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}
						_mm512_stream_ps(d, _mm512_div_ps(mval, mtweight));
						d += 16;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sum = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sum += *(s)* e;
						sumw += e;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_LUT_Setx3_8u_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;
	const float offset;

public:
	NonlocalMeansFilterInvorker_LUT_Setx3_8u_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const float offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		int CV_DECL_ALIGNED(64) buf[16];
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels() * 4, 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mb0 = _mm512_setzero_ps();
						__m512 mb1 = _mm512_setzero_ps();
						__m512 mb2 = _mm512_setzero_ps();
						__m512 mb3 = _mm512_setzero_ps();

						__m512 mg0 = _mm512_setzero_ps();
						__m512 mg1 = _mm512_setzero_ps();
						__m512 mg2 = _mm512_setzero_ps();
						__m512 mg3 = _mm512_setzero_ps();

						__m512 mr0 = _mm512_setzero_ps();
						__m512 mr1 = _mm512_setzero_ps();
						__m512 mr2 = _mm512_setzero_ps();
						__m512 mr3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							__m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep2));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw0 = _mm512_set1_ps(offset);
							__m512 mw1 = _mm512_set1_ps(offset);
							__m512 mw2 = _mm512_set1_ps(offset);
							__m512 mw3 = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								//b
								__m512i s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw0 = _mm512_mul_ps(mw0, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw1 = _mm512_mul_ps(mw1, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw2 = _mm512_mul_ps(mw2, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw3 = _mm512_mul_ps(mw3, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif

								//g
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw0 = _mm512_mul_ps(mw0, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw1 = _mm512_mul_ps(mw1, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw2 = _mm512_mul_ps(mw2, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw3 = _mm512_mul_ps(mw3, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif

								//r
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep2));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw0 = _mm512_mul_ps(mw0, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw1 = _mm512_mul_ps(mw1, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw2 = _mm512_mul_ps(mw2, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw3 = _mm512_mul_ps(mw3, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif
							}

							//low low
							//b
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mb0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw0, mss, mb0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw0, mss), mb0);
#endif
							//g
							const __m512i ssg_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep));
							__m512i ssg_32elem = _mm512_unpacklo_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg0 = _mm512_fmadd_ps(mw0, mss, mg0);
#else
							mg0 = _mm512_add_ps(_mm512_mul_ps(mw0, mss), mg0);
#endif
							//r
							const __m512i ssr_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep2));
							__m512i ssr_32elem = _mm512_unpacklo_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr0 = _mm512_fmadd_ps(mw0, mss, mr0);
#else
							mr0 = _mm512_add_ps(_mm512_mul_ps(mw0, mss), mr0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw0);


							//low high
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb1 = _mm512_fmadd_ps(mw1, mss, mb1);
#else
							mb1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mb1);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg1 = _mm512_fmadd_ps(mw1, mss, mg1);
#else
							mg1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mg1);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr1 = _mm512_fmadd_ps(mw1, mss, mr1);
#else
							mr1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mr1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw1);


							//high low
							//b
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb2 = _mm512_fmadd_ps(mw2, mss, mb2);
#else
							mb2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mb2);
#endif
							//g
							ssg_32elem = _mm512_unpackhi_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg2 = _mm512_fmadd_ps(mw2, mss, mg2);
#else
							mg2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mg2);
#endif
							//r
							ssr_32elem = _mm512_unpackhi_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr2 = _mm512_fmadd_ps(mw2, mss, mr2);
#else
							mr2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mr2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw2);


							//high high
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb3 = _mm512_fmadd_ps(mw3, mss, mb3);
#else
							mb3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mb3);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg3 = _mm512_fmadd_ps(mw3, mss, mg3);
#else
							mg3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mg3);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr3 = _mm512_fmadd_ps(mw3, mss, mr3);
#else
							mr3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mr3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw3);
						}
						mb0 = _mm512_div_ps(mb0, mtweight0);
						mb1 = _mm512_div_ps(mb1, mtweight1);
						mb2 = _mm512_div_ps(mb2, mtweight2);
						mb3 = _mm512_div_ps(mb3, mtweight3);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mb0), _mm512_cvtps_epi32(mb1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mb2), _mm512_cvtps_epi32(mb3)));
						mg0 = _mm512_div_ps(mg0, mtweight0);
						mg1 = _mm512_div_ps(mg1, mtweight1);
						mg2 = _mm512_div_ps(mg2, mtweight2);
						mg3 = _mm512_div_ps(mg3, mtweight3);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mg0), _mm512_cvtps_epi32(mg1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mg2), _mm512_cvtps_epi32(mg3)));
						mr0 = _mm512_div_ps(mr0, mtweight0);
						mr1 = _mm512_div_ps(mr1, mtweight1);
						mr2 = _mm512_div_ps(mr2, mtweight2);
						mr3 = _mm512_div_ps(mr3, mtweight3);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mr0), _mm512_cvtps_epi32(mr1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mr2), _mm512_cvtps_epi32(mr3)));

						_mm512_stream_epi8_color(d, a, b, c);
						d += 192;
					}
				}
#endif
				for (; i < dest->cols; i++)
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs((float)*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
							s0 = abs((float)*(s + t_ofs[n] + colorstep) - tV[count++]);
							e *= w[s0];
							s0 = abs((float)*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e *= w[s0];
						}
						sumb += *(s)* e;
						sumg += *(s + colorstep) * e;
						sumr += *(s + colorstep2) * e;
						sumw += e;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mval0 = _mm512_setzero_ps();
						__m512 mval1 = _mm512_setzero_ps();
						__m512 mval2 = _mm512_setzero_ps();
						__m512 mval3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							const __m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw0 = _mm512_set1_ps(offset);
							__m512 mw1 = _mm512_set1_ps(offset);
							__m512 mw2 = _mm512_set1_ps(offset);
							__m512 mw3 = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512i ms_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i ms_32elem = _mm512_unpacklo_epi8(ms_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw0 = _mm512_mul_ps(mw0, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw1 = _mm512_mul_ps(mw1, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								ms_32elem = _mm512_unpackhi_epi8(ms_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw2 = _mm512_mul_ps(mw2, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw3 = _mm512_mul_ps(mw3, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif
							}
							//low low
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mval0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw0, mss, mval0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw0, mss), mval0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw0);

							//low high
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval1 = _mm512_fmadd_ps(mw1, mss, mval1);
#else
							mval1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mval1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw1);

							//high low
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval2 = _mm512_fmadd_ps(mw2, mss, mval2);
#else
							mval2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mval2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw2);


							//high high
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval3 = _mm512_fmadd_ps(mw3, mss, mval3);
#else
							mval3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mval3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw3);
						}
						mval0 = _mm512_div_ps(mval0, mtweight0);
						mval1 = _mm512_div_ps(mval1, mtweight1);
						mval2 = _mm512_div_ps(mval2, mtweight2);
						mval3 = _mm512_div_ps(mval3, mtweight3);
						_mm512_stream_si512((__m512i*)d, _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mval0), _mm512_cvtps_epi32(mval1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mval2), _mm512_cvtps_epi32(mval3))));
						d += 64;
					}
				}
#endif
				for (; i < dest->cols; i++)
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs((float)*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sumb += *(s)* e;
						sumw += e;
					}
					d[0] = sumb / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};


/*************************************************
	using Quantization LUT with "set instruction" x N
*************************************************/
class NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_64f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const double* weight;
	const double offset;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_64f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const double* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const double offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_maxk(search_maxk), search_ofs(search_ofs), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const long long CV_DECL_ALIGNED(64) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff,
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff
		};
		long long CV_DECL_ALIGNED(64) buf[8];
		__m512d* tVec = (__m512d*)_mm_malloc(sizeof(__m512d)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const double* w = weight;
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mb = _mm512_setzero_pd();
						__m512d mg = _mm512_setzero_pd();
						__m512d mr = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d mw = _mm512_set1_pd(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512d s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]);
								__m512d me = _mm512_mul_pd(s0, s0);

								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif

								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, me))));
								mw = _mm512_mul_pd(mw, _mm512_set_pd(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mb);
							mg = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep), mg);
							mr = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep2), mr);
#else
							mb = _mm512_add_pd(mb, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
							mg = _mm512_add_pd(mg, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep)));
							mr = _mm512_add_pd(mr, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep2)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}

						mb = _mm512_div_pd(mb, mtweight);
						mg = _mm512_div_pd(mg, mtweight);
						mr = _mm512_div_pd(mr, mtweight);

						_mm512_stream_pd_color(d, mb, mg, mr);
						d += 24;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sumb = 0;
					double sumg = 0;
					double sumr = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double ww = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = (*(s + t_ofs[n]) - tV[count++]);
							double e = s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
							ww *= w[(int)sqrt(e)];
						}
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const double* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mval = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d mw = _mm512_set1_pd(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512d ms = _mm512_loadu_pd(s + t_ofs[n]);
								_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(ms, tVec[count++]), *(__m512d const*)v64f_absmask)));
								mw = _mm512_mul_pd(mw, _mm512_set_pd(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mval);
#else
							mval = _mm512_add_pd(mval, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}
						_mm512_stream_pd(d, _mm512_div_pd(mval, mtweight));
						d += 8;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sum = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sum += *(s)* e;
						sumw += e;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_32f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;
	const float offset;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_32f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const float offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		int CV_DECL_ALIGNED(64) buf[16];
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mb = _mm512_setzero_ps();
						__m512 mg = _mm512_setzero_ps();
						__m512 mr = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512 s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]);
								__m512 me = _mm512_mul_ps(s0, s0);

								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif

								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me))));
								mw = _mm512_mul_ps(mw, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mb);
							mg = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep), mg);
							mr = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep2), mr);
#else
							mb = _mm512_add_ps(mb, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
							mg = _mm512_add_ps(mg, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep)));
							mr = _mm512_add_ps(mr, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep2)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}

						mb = _mm512_div_ps(mb, mtweight);
						mg = _mm512_div_ps(mg, mtweight);
						mr = _mm512_div_ps(mr, mtweight);

						_mm512_stream_ps_color(d, mb, mg, mr);
						d += 48;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float ww = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = (*(s + t_ofs[n]) - tV[count++]);
							double e = s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
							ww *= w[(int)sqrt(e)];
						}
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mval = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw = _mm512_mul_ps(mw, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mval);
#else
							mval = _mm512_add_ps(mval, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}
						_mm512_stream_ps(d, _mm512_div_ps(mval, mtweight));
						d += 16;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sum = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sum += *(s)* e;
						sumw += e;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_8u_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;
	const float offset;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_8u_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const float offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		int CV_DECL_ALIGNED(64) buf[16];
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels() * 4, 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mb0 = _mm512_setzero_ps();
						__m512 mb1 = _mm512_setzero_ps();
						__m512 mb2 = _mm512_setzero_ps();
						__m512 mb3 = _mm512_setzero_ps();

						__m512 mg0 = _mm512_setzero_ps();
						__m512 mg1 = _mm512_setzero_ps();
						__m512 mg2 = _mm512_setzero_ps();
						__m512 mg3 = _mm512_setzero_ps();

						__m512 mr0 = _mm512_setzero_ps();
						__m512 mr1 = _mm512_setzero_ps();
						__m512 mr2 = _mm512_setzero_ps();
						__m512 mr3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							__m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep2));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw0 = _mm512_set1_ps(offset);
							__m512 mw1 = _mm512_set1_ps(offset);
							__m512 mw2 = _mm512_set1_ps(offset);
							__m512 mw3 = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								//computing color L2 norm
								//b
								__m512i s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								__m512 s0 = _mm512_sub_ps(ms, tVec[count++]);
								__m512 me0 = _mm512_mul_ps(s0, s0);

								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
								__m512 me1 = _mm512_mul_ps(s0, s0);

								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
								__m512 me2 = _mm512_mul_ps(s0, s0);

								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
								__m512 me3 = _mm512_mul_ps(s0, s0);

								//g
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif

								//r
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep2));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me0))));
								mw0 = _mm512_mul_ps(mw0, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me1))));
								mw1 = _mm512_mul_ps(mw1, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me2))));
								mw2 = _mm512_mul_ps(mw2, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me3))));
								mw3 = _mm512_mul_ps(mw3, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif
							}

							//low low
							//b
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mb0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw0, mss, mb0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw0, mss), mb0);
#endif
							//g
							const __m512i ssg_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep));
							__m512i ssg_32elem = _mm512_unpacklo_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg0 = _mm512_fmadd_ps(mw0, mss, mg0);
#else
							mg0 = _mm512_add_ps(_mm512_mul_ps(mw0, mss), mg0);
#endif
							//r
							const __m512i ssr_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep2));
							__m512i ssr_32elem = _mm512_unpacklo_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr0 = _mm512_fmadd_ps(mw0, mss, mr0);
#else
							mr0 = _mm512_add_ps(_mm512_mul_ps(mw0, mss), mr0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw0);


							//low high
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb1 = _mm512_fmadd_ps(mw1, mss, mb1);
#else
							mb1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mb1);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg1 = _mm512_fmadd_ps(mw1, mss, mg1);
#else
							mg1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mg1);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr1 = _mm512_fmadd_ps(mw1, mss, mr1);
#else
							mr1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mr1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw1);


							//high low
							//b
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb2 = _mm512_fmadd_ps(mw2, mss, mb2);
#else
							mb2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mb2);
#endif
							//g
							ssg_32elem = _mm512_unpackhi_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg2 = _mm512_fmadd_ps(mw2, mss, mg2);
#else
							mg2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mg2);
#endif
							//r
							ssr_32elem = _mm512_unpackhi_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr2 = _mm512_fmadd_ps(mw2, mss, mr2);
#else
							mr2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mr2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw2);


							//high high
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb3 = _mm512_fmadd_ps(mw3, mss, mb3);
#else
							mb3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mb3);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg3 = _mm512_fmadd_ps(mw3, mss, mg3);
#else
							mg3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mg3);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr3 = _mm512_fmadd_ps(mw3, mss, mr3);
#else
							mr3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mr3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw3);
						}
						mb0 = _mm512_div_ps(mb0, mtweight0);
						mb1 = _mm512_div_ps(mb1, mtweight1);
						mb2 = _mm512_div_ps(mb2, mtweight2);
						mb3 = _mm512_div_ps(mb3, mtweight3);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mb0), _mm512_cvtps_epi32(mb1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mb2), _mm512_cvtps_epi32(mb3)));
						mg0 = _mm512_div_ps(mg0, mtweight0);
						mg1 = _mm512_div_ps(mg1, mtweight1);
						mg2 = _mm512_div_ps(mg2, mtweight2);
						mg3 = _mm512_div_ps(mg3, mtweight3);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mg0), _mm512_cvtps_epi32(mg1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mg2), _mm512_cvtps_epi32(mg3)));
						mr0 = _mm512_div_ps(mr0, mtweight0);
						mr1 = _mm512_div_ps(mr1, mtweight1);
						mr2 = _mm512_div_ps(mr2, mtweight2);
						mr3 = _mm512_div_ps(mr3, mtweight3);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mr0), _mm512_cvtps_epi32(mr1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mr2), _mm512_cvtps_epi32(mr3)));

						_mm512_stream_epi8_color(d, a, b, c);
						d += 192;
					}
				}
#endif
				for (; i < dest->cols; i++)
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float ww = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = ((float)*(s + t_ofs[n]) - tV[count++]);
							double e = s0 * s0;
							s0 = ((float)*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = ((float)*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
							ww *= w[(int)sqrt(e)];
						}
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mval0 = _mm512_setzero_ps();
						__m512 mval1 = _mm512_setzero_ps();
						__m512 mval2 = _mm512_setzero_ps();
						__m512 mval3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							const __m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw0 = _mm512_set1_ps(offset);
							__m512 mw1 = _mm512_set1_ps(offset);
							__m512 mw2 = _mm512_set1_ps(offset);
							__m512 mw3 = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512i ms_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i ms_32elem = _mm512_unpacklo_epi8(ms_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw0 = _mm512_mul_ps(mw0, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw1 = _mm512_mul_ps(mw1, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								ms_32elem = _mm512_unpackhi_epi8(ms_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw2 = _mm512_mul_ps(mw2, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask)));
								mw3 = _mm512_mul_ps(mw3, _mm512_set_ps(
									w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
									w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
								));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif
							}
							//low low
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mval0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw0, mss, mval0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw0, mss), mval0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw0);

							//low high
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval1 = _mm512_fmadd_ps(mw1, mss, mval1);
#else
							mval1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mval1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw1);

							//high low
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval2 = _mm512_fmadd_ps(mw2, mss, mval2);
#else
							mval2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mval2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw2);


							//high high
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval3 = _mm512_fmadd_ps(mw3, mss, mval3);
#else
							mval3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mval3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw3);
						}
						mval0 = _mm512_div_ps(mval0, mtweight0);
						mval1 = _mm512_div_ps(mval1, mtweight1);
						mval2 = _mm512_div_ps(mval2, mtweight2);
						mval3 = _mm512_div_ps(mval3, mtweight3);
						_mm512_stream_si512((__m512i*)d, _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mval0), _mm512_cvtps_epi32(mval1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mval2), _mm512_cvtps_epi32(mval3))));
						d += 64;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumv = 0;
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs((float)*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sumv += *(s)* e;
						sumw += e;
					}
					d[0] = sumv / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};


/*************************************************
	using Quantization LUT with "set instruction" x 1
*************************************************/
class NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_64f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const double* weight;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_64f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const double* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_maxk(search_maxk), search_ofs(search_ofs), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		long long CV_DECL_ALIGNED(64) buf[8];
		__m512d* tVec = (__m512d*)_mm_malloc(sizeof(__m512d)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const double* w = weight;
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					for (; i < dest->cols; i += 8)
					{
						__m512d mb = _mm512_setzero_pd();
						__m512d mg = _mm512_setzero_pd();
						__m512d mr = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d me = _mm512_setzero_pd();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512d s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif
								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif
								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif
							}
							_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, me))));
							__m512d mw = _mm512_set_pd(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mb);
							mg = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep), mg);
							mr = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep2), mr);
#else
							mb = _mm512_add_pd(mb, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
							mg = _mm512_add_pd(mg, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep)));
							mr = _mm512_add_pd(mr, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep2)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}

						mb = _mm512_div_pd(mb, mtweight);
						mg = _mm512_div_pd(mg, mtweight);
						mr = _mm512_div_pd(mr, mtweight);

						_mm512_stream_pd_color(d, mb, mg, mr);
						d += 24;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sumb = 0;
					double sumg = 0;
					double sumr = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = (*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
						}
						const double ww = w[(int)sqrt(e)];
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const double* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					for (; i < dest->cols; i += 8)
					{
						__m512d mval = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d me = _mm512_setzero_pd();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512d ms = _mm512_loadu_pd(s + t_ofs[n]);
								const __m512d diff = _mm512_sub_pd(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(diff, diff, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(diff, diff), me);
#endif
							}
							_mm512_store_si512((__m512i*)buf, _mm512_cvtpd_epi64(_mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, me))));
							__m512d mw = _mm512_set_pd(w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]);
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mval);
#else
							mval = _mm512_add_pd(mval, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}
						_mm512_stream_pd(d, _mm512_div_pd(mval, mtweight));
						d += 8;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sum = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
						}
						const double ww = w[(int)sqrt(e)];
						sum += *(s)* ww;
						sumw += ww;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_32f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_32f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		int CV_DECL_ALIGNED(64) buf[16];
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif
		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					for (; i < dest->cols; i += 16)
					{
						__m512 mb = _mm512_setzero_ps();
						__m512 mg = _mm512_setzero_ps();
						__m512 mr = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512 s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + +t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
							}
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me))));
							__m512 mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mb);
							mg = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep), mg);
							mr = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep2), mr);
#else
							mb = _mm512_add_ps(mb, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
							mg = _mm512_add_ps(mg, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep)));
							mr = _mm512_add_ps(mr, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep2)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}

						mb = _mm512_div_ps(mb, mtweight);
						mg = _mm512_div_ps(mg, mtweight);
						mr = _mm512_div_ps(mr, mtweight);

						_mm512_stream_ps_color(d, mb, mg, mr);
						d += 48;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = (*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
						}
						const float ww = w[(int)sqrt(e)];
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					for (; i < dest->cols; i += 16)
					{
						__m512 mval = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512 s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
							}
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me))));
							__m512 mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mval);
#else
							mval = _mm512_add_ps(mval, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}
						_mm512_stream_ps(d, _mm512_div_ps(mval, mtweight));
						d += 16;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sum = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
						}
						const double ww = w[(int)sqrt(e)];
						sum += *(s)* ww;
						sumw += ww;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_8u_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_8u_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		int CV_DECL_ALIGNED(64) buf[16];
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels() * 4, 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
					for (; i < dest->cols; i += 64)
					{
						__m512 mb0 = _mm512_setzero_ps();
						__m512 mb1 = _mm512_setzero_ps();
						__m512 mb2 = _mm512_setzero_ps();
						__m512 mb3 = _mm512_setzero_ps();

						__m512 mg0 = _mm512_setzero_ps();
						__m512 mg1 = _mm512_setzero_ps();
						__m512 mg2 = _mm512_setzero_ps();
						__m512 mg3 = _mm512_setzero_ps();

						__m512 mr0 = _mm512_setzero_ps();
						__m512 mr1 = _mm512_setzero_ps();
						__m512 mr2 = _mm512_setzero_ps();
						__m512 mr3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							__m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep2));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me0 = _mm512_setzero_ps();
							__m512 me1 = _mm512_setzero_ps();
							__m512 me2 = _mm512_setzero_ps();
							__m512 me3 = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								//b
								__m512i s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								__m512 s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif

								//g
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif

								//r
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep2));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif
							}

							//low low
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me0))));
							__m512 mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
							//b
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mb0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw, mss, mb0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw, mss), mb0);
#endif
							//g
							const __m512i ssg_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep));
							__m512i ssg_32elem = _mm512_unpacklo_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg0 = _mm512_fmadd_ps(mw, mss, mg0);
#else
							mg0 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg0);
#endif
							//r
							const __m512i ssr_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep2));
							__m512i ssr_32elem = _mm512_unpacklo_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr0 = _mm512_fmadd_ps(mw, mss, mr0);
#else
							mr0 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw);


							//low high
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me1))));
							mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb1 = _mm512_fmadd_ps(mw, mss, mb1);
#else
							mb1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb1);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg1 = _mm512_fmadd_ps(mw, mss, mg1);
#else
							mg1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg1);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr1 = _mm512_fmadd_ps(mw, mss, mr1);
#else
							mr1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw);


							//high low
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me2))));
							mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
							//b
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb2 = _mm512_fmadd_ps(mw, mss, mb2);
#else
							mb2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb2);
#endif
							//g
							ssg_32elem = _mm512_unpackhi_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg2 = _mm512_fmadd_ps(mw, mss, mg2);
#else
							mg2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg2);
#endif
							//r
							ssr_32elem = _mm512_unpackhi_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr2 = _mm512_fmadd_ps(mw, mss, mr2);
#else
							mr2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw);


							//high high
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me3))));
							mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb3 = _mm512_fmadd_ps(mw, mss, mb3);
#else
							mb3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb3);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg3 = _mm512_fmadd_ps(mw, mss, mg3);
#else
							mg3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg3);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr3 = _mm512_fmadd_ps(mw, mss, mr3);
#else
							mr3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw);
						}
						mb0 = _mm512_div_ps(mb0, mtweight0);
						mb1 = _mm512_div_ps(mb1, mtweight1);
						mb2 = _mm512_div_ps(mb2, mtweight2);
						mb3 = _mm512_div_ps(mb3, mtweight3);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mb0), _mm512_cvtps_epi32(mb1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mb2), _mm512_cvtps_epi32(mb3)));
						mg0 = _mm512_div_ps(mg0, mtweight0);
						mg1 = _mm512_div_ps(mg1, mtweight1);
						mg2 = _mm512_div_ps(mg2, mtweight2);
						mg3 = _mm512_div_ps(mg3, mtweight3);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mg0), _mm512_cvtps_epi32(mg1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mg2), _mm512_cvtps_epi32(mg3)));
						mr0 = _mm512_div_ps(mr0, mtweight0);
						mr1 = _mm512_div_ps(mr1, mtweight1);
						mr2 = _mm512_div_ps(mr2, mtweight2);
						mr3 = _mm512_div_ps(mr3, mtweight3);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mr0), _mm512_cvtps_epi32(mr1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mr2), _mm512_cvtps_epi32(mr3)));

						_mm512_stream_epi8_color(d, a, b, c);
						d += 192;
					}
				}
#endif
				for (; i < dest->cols; i++)
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = ((float)*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
							s0 = ((float)*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = ((float)*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
						}
						const double ww = w[(int)sqrt(e)];
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
					for (; i < dest->cols; i += 64)
					{
						__m512 mval0 = _mm512_setzero_ps();
						__m512 mval1 = _mm512_setzero_ps();
						__m512 mval2 = _mm512_setzero_ps();
						__m512 mval3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							const __m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me0 = _mm512_setzero_ps();
							__m512 me1 = _mm512_setzero_ps();
							__m512 me2 = _mm512_setzero_ps();
							__m512 me3 = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512i ms_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i ms_32elem = _mm512_unpacklo_epi8(ms_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								__m512 diff = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(diff, diff, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(diff, diff), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								diff = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(diff, diff, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(diff, diff), me1);
#endif
								ms_32elem = _mm512_unpackhi_epi8(ms_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								diff = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(diff, diff, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(diff, diff), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								diff = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(diff, diff, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(diff, diff), me3);
#endif
							}

							//low low
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me0))));
							__m512 mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mval0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw, mss, mval0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw, mss), mval0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw);

							//low high
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me1))));
							mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval1 = _mm512_fmadd_ps(mw, mss, mval1);
#else
							mval1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw);

							//high low
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me2))));
							mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval2 = _mm512_fmadd_ps(mw, mss, mval2);
#else
							mval2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw);


							//high high
							_mm512_store_si512((__m512i*)buf, _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me3))));
							mw = _mm512_set_ps(
								w[buf[15]], w[buf[14]], w[buf[13]], w[buf[12]], w[buf[11]], w[buf[10]], w[buf[9]], w[buf[8]],
								w[buf[7]], w[buf[6]], w[buf[5]], w[buf[4]], w[buf[3]], w[buf[2]], w[buf[1]], w[buf[0]]
							);
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval3 = _mm512_fmadd_ps(mw, mss, mval3);
#else
							mval3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw);
						}
						mval0 = _mm512_div_ps(mval0, mtweight0);
						mval1 = _mm512_div_ps(mval1, mtweight1);
						mval2 = _mm512_div_ps(mval2, mtweight2);
						mval3 = _mm512_div_ps(mval3, mtweight3);
						_mm512_stream_si512((__m512i*)d, _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mval0), _mm512_cvtps_epi32(mval1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mval2), _mm512_cvtps_epi32(mval3))));
						d += 64;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumv = 0;
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs((float)*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
						}
						const float ww = w[(int)sqrt(e)];
						sumv += *(s)* ww;
						sumw += ww;
					}
					d[0] = sumv / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};


/*************************************************
	using LUT with "gather instruction" x 3
*************************************************/
class NonlocalMeansFilterInvorker_LUT_Gatherx3_64f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const double* weight;
	const double offset;

public:
	NonlocalMeansFilterInvorker_LUT_Gatherx3_64f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const double* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const double offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_maxk(search_maxk), search_ofs(search_ofs), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const long long CV_DECL_ALIGNED(64) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff,
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff
		};
		__m512d* tVec = (__m512d*)_mm_malloc(sizeof(__m512d)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const double* w = weight;
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mb = _mm512_setzero_pd();
						__m512d mg = _mm512_setzero_pd();
						__m512d mr = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d mw = _mm512_set1_pd(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512i buf = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]), *(__m512d const*)v64f_absmask));
								mw = _mm512_mul_pd(mw, _mm512_i64gather_pd(buf, w, 8));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
								buf = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep), tVec[count++]), *(__m512d const*)v64f_absmask));
								mw = _mm512_mul_pd(mw, _mm512_i64gather_pd(buf, w, 8));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
								buf = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep2), tVec[count++]), *(__m512d const*)v64f_absmask));
								mw = _mm512_mul_pd(mw, _mm512_i64gather_pd(buf, w, 8));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mb);
							mg = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep), mg);
							mr = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep2), mr);
#else
							mb = _mm512_add_pd(mb, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
							mg = _mm512_add_pd(mg, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep)));
							mr = _mm512_add_pd(mr, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep2)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}

						mb = _mm512_div_pd(mb, mtweight);
						mg = _mm512_div_pd(mg, mtweight);
						mr = _mm512_div_pd(mr, mtweight);

						_mm512_stream_pd_color(d, mb, mg, mr);
						d += 24;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sumb = 0;
					double sumg = 0;
					double sumr = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
							s0 = abs(*(s + t_ofs[n] + colorstep) - tV[count++]);
							e *= w[s0];
							s0 = abs(*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e *= w[s0];
						}
						sumb += *(s)* e;
						sumg += *(s + colorstep) * e;
						sumr += *(s + colorstep2) * e;
						sumw += e;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const double* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mval = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d mw = _mm512_set1_pd(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512i buf = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]), *(__m512d const*)v64f_absmask));
								mw = _mm512_mul_pd(mw, _mm512_i64gather_pd(buf, w, 8));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mval);
#else
							mval = _mm512_add_pd(mval, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}
						_mm512_stream_pd(d, _mm512_div_pd(mval, mtweight));
						d += 8;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sum = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sum += *(s)* e;
						sumw += e;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_LUT_Gatherx3_32f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;
	const float offset;

public:
	NonlocalMeansFilterInvorker_LUT_Gatherx3_32f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const float offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mb = _mm512_setzero_ps();
						__m512 mg = _mm512_setzero_ps();
						__m512 mr = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512i buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]), *(__m512 const*)v32f_absmask));
								mw = _mm512_mul_ps(mw, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep), tVec[count++]), *(__m512 const*)v32f_absmask));
								mw = _mm512_mul_ps(mw, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep2), tVec[count++]), *(__m512 const*)v32f_absmask));
								mw = _mm512_mul_ps(mw, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mb);
							mg = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep), mg);
							mr = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep2), mr);
#else
							mb = _mm512_add_ps(mb, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
							mg = _mm512_add_ps(mg, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep)));
							mr = _mm512_add_ps(mr, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep2)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}

						mb = _mm512_div_ps(mb, mtweight);
						mg = _mm512_div_ps(mg, mtweight);
						mr = _mm512_div_ps(mr, mtweight);

						_mm512_stream_ps_color(d, mb, mg, mr);
						d += 48;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
							s0 = abs(*(s + t_ofs[n] + colorstep) - tV[count++]);
							e *= w[s0];
							s0 = abs(*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e *= w[s0];
						}
						sumb += *(s)* e;
						sumg += *(s + colorstep) * e;
						sumr += *(s + colorstep2) * e;
						sumw += e;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mval = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512i buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]), *(__m512 const*)v32f_absmask));
								mw = _mm512_mul_ps(mw, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mval);
#else
							mval = _mm512_add_ps(mval, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}
						_mm512_stream_ps(d, _mm512_div_ps(mval, mtweight));
						d += 16;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sum = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sum += *(s)* e;
						sumw += e;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_LUT_Gatherx3_8u_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;
	const float offset;

public:
	NonlocalMeansFilterInvorker_LUT_Gatherx3_8u_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const float offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels() * 4, 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mb0 = _mm512_setzero_ps();
						__m512 mb1 = _mm512_setzero_ps();
						__m512 mb2 = _mm512_setzero_ps();
						__m512 mb3 = _mm512_setzero_ps();

						__m512 mg0 = _mm512_setzero_ps();
						__m512 mg1 = _mm512_setzero_ps();
						__m512 mg2 = _mm512_setzero_ps();
						__m512 mg3 = _mm512_setzero_ps();

						__m512 mr0 = _mm512_setzero_ps();
						__m512 mr1 = _mm512_setzero_ps();
						__m512 mr2 = _mm512_setzero_ps();
						__m512 mr3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							__m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep2));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw0 = _mm512_set1_ps(offset);
							__m512 mw1 = _mm512_set1_ps(offset);
							__m512 mw2 = _mm512_set1_ps(offset);
							__m512 mw3 = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								//b
								__m512i s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								__m512i buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw0 = _mm512_mul_ps(mw0, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw1 = _mm512_mul_ps(mw1, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw2 = _mm512_mul_ps(mw2, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw3 = _mm512_mul_ps(mw3, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif

								//g
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw0 = _mm512_mul_ps(mw0, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw1 = _mm512_mul_ps(mw1, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw2 = _mm512_mul_ps(mw2, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw3 = _mm512_mul_ps(mw3, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif

								//r
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep2));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw0 = _mm512_mul_ps(mw0, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw1 = _mm512_mul_ps(mw1, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw2 = _mm512_mul_ps(mw2, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw3 = _mm512_mul_ps(mw3, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif
							}

							//low low
							//b
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mb0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw0, mss, mb0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw0, mss), mb0);
#endif
							//g
							const __m512i ssg_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep));
							__m512i ssg_32elem = _mm512_unpacklo_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg0 = _mm512_fmadd_ps(mw0, mss, mg0);
#else
							mg0 = _mm512_add_ps(_mm512_mul_ps(mw0, mss), mg0);
#endif
							//r
							const __m512i ssr_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep2));
							__m512i ssr_32elem = _mm512_unpacklo_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr0 = _mm512_fmadd_ps(mw0, mss, mr0);
#else
							mr0 = _mm512_add_ps(_mm512_mul_ps(mw0, mss), mr0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw0);


							//low high
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb1 = _mm512_fmadd_ps(mw1, mss, mb1);
#else
							mb1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mb1);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg1 = _mm512_fmadd_ps(mw1, mss, mg1);
#else
							mg1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mg1);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr1 = _mm512_fmadd_ps(mw1, mss, mr1);
#else
							mr1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mr1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw1);


							//high low
							//b
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb2 = _mm512_fmadd_ps(mw2, mss, mb2);
#else
							mb2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mb2);
#endif
							//g
							ssg_32elem = _mm512_unpackhi_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg2 = _mm512_fmadd_ps(mw2, mss, mg2);
#else
							mg2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mg2);
#endif
							//r
							ssr_32elem = _mm512_unpackhi_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr2 = _mm512_fmadd_ps(mw2, mss, mr2);
#else
							mr2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mr2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw2);


							//high high
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb3 = _mm512_fmadd_ps(mw3, mss, mb3);
#else
							mb3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mb3);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg3 = _mm512_fmadd_ps(mw3, mss, mg3);
#else
							mg3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mg3);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr3 = _mm512_fmadd_ps(mw3, mss, mr3);
#else
							mr3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mr3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw3);
						}
						mb0 = _mm512_div_ps(mb0, mtweight0);
						mb1 = _mm512_div_ps(mb1, mtweight1);
						mb2 = _mm512_div_ps(mb2, mtweight2);
						mb3 = _mm512_div_ps(mb3, mtweight3);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mb0), _mm512_cvtps_epi32(mb1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mb2), _mm512_cvtps_epi32(mb3)));
						mg0 = _mm512_div_ps(mg0, mtweight0);
						mg1 = _mm512_div_ps(mg1, mtweight1);
						mg2 = _mm512_div_ps(mg2, mtweight2);
						mg3 = _mm512_div_ps(mg3, mtweight3);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mg0), _mm512_cvtps_epi32(mg1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mg2), _mm512_cvtps_epi32(mg3)));
						mr0 = _mm512_div_ps(mr0, mtweight0);
						mr1 = _mm512_div_ps(mr1, mtweight1);
						mr2 = _mm512_div_ps(mr2, mtweight2);
						mr3 = _mm512_div_ps(mr3, mtweight3);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mr0), _mm512_cvtps_epi32(mr1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mr2), _mm512_cvtps_epi32(mr3)));

						_mm512_stream_epi8_color(d, a, b, c);
						d += 192;
					}
				}
#endif
				for (; i < dest->cols; i++)
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs((float)*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
							s0 = abs((float)*(s + t_ofs[n] + colorstep) - tV[count++]);
							e *= w[s0];
							s0 = abs((float)*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e *= w[s0];
						}
						sumb += *(s)* e;
						sumg += *(s + colorstep) * e;
						sumr += *(s + colorstep2) * e;
						sumw += e;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mval0 = _mm512_setzero_ps();
						__m512 mval1 = _mm512_setzero_ps();
						__m512 mval2 = _mm512_setzero_ps();
						__m512 mval3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							const __m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw0 = _mm512_set1_ps(offset);
							__m512 mw1 = _mm512_set1_ps(offset);
							__m512 mw2 = _mm512_set1_ps(offset);
							__m512 mw3 = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512i ms_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i ms_32elem = _mm512_unpacklo_epi8(ms_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								__m512i buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw0 = _mm512_mul_ps(mw0, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw1 = _mm512_mul_ps(mw1, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								ms_32elem = _mm512_unpackhi_epi8(ms_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw2 = _mm512_mul_ps(mw2, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw3 = _mm512_mul_ps(mw3, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif
							}
							//low low
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mval0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw0, mss, mval0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw0, mss), mval0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw0);

							//low high
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval1 = _mm512_fmadd_ps(mw1, mss, mval1);
#else
							mval1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mval1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw1);

							//high low
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval2 = _mm512_fmadd_ps(mw2, mss, mval2);
#else
							mval2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mval2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw2);


							//high high
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval3 = _mm512_fmadd_ps(mw3, mss, mval3);
#else
							mval3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mval3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw3);
						}
						mval0 = _mm512_div_ps(mval0, mtweight0);
						mval1 = _mm512_div_ps(mval1, mtweight1);
						mval2 = _mm512_div_ps(mval2, mtweight2);
						mval3 = _mm512_div_ps(mval3, mtweight3);
						_mm512_stream_si512((__m512i*)d, _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mval0), _mm512_cvtps_epi32(mval1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mval2), _mm512_cvtps_epi32(mval3))));
						d += 64;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs((float)*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sumb += *(s)* e;
						sumw += e;
					}
					d[0] = sumb / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};


/*************************************************
	using Quantization LUT with "gather instruction" x N
*************************************************/
class NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_64f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const double* weight;
	const double offset;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_64f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const double* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const double offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_maxk(search_maxk), search_ofs(search_ofs), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const long long CV_DECL_ALIGNED(64) v64f_absmask[] = {
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff,
			0x7fffffffffffffff, 0x7fffffffffffffff,	0x7fffffffffffffff, 0x7fffffffffffffff
		};
		__m512d* tVec = (__m512d*)_mm_malloc(sizeof(__m512d)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const double* w = weight;
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mb = _mm512_setzero_pd();
						__m512d mg = _mm512_setzero_pd();
						__m512d mr = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d mw = _mm512_set1_pd(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512d s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]);
								__m512d me = _mm512_mul_pd(s0, s0);

								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif

								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif
								__m512i buf = _mm512_cvtpd_epi64(_mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, me)));
								mw = _mm512_mul_pd(mw, _mm512_i64gather_pd(buf, w, 8));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mb);
							mg = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep), mg);
							mr = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep2), mr);
#else
							mb = _mm512_add_pd(mb, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
							mg = _mm512_add_pd(mg, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep)));
							mr = _mm512_add_pd(mr, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep2)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}

						mb = _mm512_div_pd(mb, mtweight);
						mg = _mm512_div_pd(mg, mtweight);
						mr = _mm512_div_pd(mr, mtweight);

						_mm512_stream_pd_color(d, mb, mg, mr);
						d += 24;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sumb = 0;
					double sumg = 0;
					double sumr = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double ww = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = (*(s + t_ofs[n]) - tV[count++]);
							double e = s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
							ww *= w[(int)sqrt(e)];
						}
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const double* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512d double_min = _mm512_set1_pd(DBL_MIN);
#endif
					for (; i < dest->cols; i += 8)
					{
						__m512d mval = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d mw = _mm512_set1_pd(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512d ms = _mm512_loadu_pd(s + t_ofs[n]);
								__m512i buf = _mm512_cvtpd_epi64(_mm512_and_pd(_mm512_sub_pd(ms, tVec[count++]), *(__m512d const*)v64f_absmask));
								mw = _mm512_mul_pd(mw, _mm512_i64gather_pd(buf, w, 8));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_pd(mw, double_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mval);
#else
							mval = _mm512_add_pd(mval, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}
						_mm512_stream_pd(d, _mm512_div_pd(mval, mtweight));
						d += 8;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sum = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sum += *(s)* e;
						sumw += e;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_32f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;
	const float offset;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_32f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const float offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mb = _mm512_setzero_ps();
						__m512 mg = _mm512_setzero_ps();
						__m512 mr = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512 s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]);
								__m512 me = _mm512_mul_ps(s0, s0);

								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif

								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
								__m512i buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me)));
								mw = _mm512_mul_ps(mw, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mb);
							mg = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep), mg);
							mr = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep2), mr);
#else
							mb = _mm512_add_ps(mb, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
							mg = _mm512_add_ps(mg, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep)));
							mr = _mm512_add_ps(mr, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep2)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}

						mb = _mm512_div_ps(mb, mtweight);
						mg = _mm512_div_ps(mg, mtweight);
						mr = _mm512_div_ps(mr, mtweight);

						_mm512_stream_ps_color(d, mb, mg, mr);
						d += 48;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float ww = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = (*(s + t_ofs[n]) - tV[count++]);
							double e = s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
							ww *= w[(int)sqrt(e)];
						}
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mval = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512i buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]), *(__m512 const*)v32f_absmask));
								mw = _mm512_mul_ps(mw, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw = _mm512_max_ps(mw, float_min);
#endif
							}
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mval);
#else
							mval = _mm512_add_ps(mval, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}
						_mm512_stream_ps(d, _mm512_div_ps(mval, mtweight));
						d += 16;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sum = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sum += *(s)* e;
						sumw += e;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_8u_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;
	const float offset;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_8u_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs, const float offset = 1)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight), offset(offset)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		static const int CV_DECL_ALIGNED(64) v32f_absmask[] = {
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
			0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,	0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
		};
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels() * 4, 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mb0 = _mm512_setzero_ps();
						__m512 mb1 = _mm512_setzero_ps();
						__m512 mb2 = _mm512_setzero_ps();
						__m512 mb3 = _mm512_setzero_ps();

						__m512 mg0 = _mm512_setzero_ps();
						__m512 mg1 = _mm512_setzero_ps();
						__m512 mg2 = _mm512_setzero_ps();
						__m512 mg3 = _mm512_setzero_ps();

						__m512 mr0 = _mm512_setzero_ps();
						__m512 mr1 = _mm512_setzero_ps();
						__m512 mr2 = _mm512_setzero_ps();
						__m512 mr3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							__m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep2));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw0 = _mm512_set1_ps(offset);
							__m512 mw1 = _mm512_set1_ps(offset);
							__m512 mw2 = _mm512_set1_ps(offset);
							__m512 mw3 = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								//computing color L2 norm
								//b
								__m512i s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								__m512 s0 = _mm512_sub_ps(ms, tVec[count++]);
								__m512 me0 = _mm512_mul_ps(s0, s0);

								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
								__m512 me1 = _mm512_mul_ps(s0, s0);

								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
								__m512 me2 = _mm512_mul_ps(s0, s0);

								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
								__m512 me3 = _mm512_mul_ps(s0, s0);

								//g
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif

								//r
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep2));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								__m512i buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me0)));
								mw0 = _mm512_mul_ps(mw0, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me1)));
								mw1 = _mm512_mul_ps(mw1, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me2)));
								mw2 = _mm512_mul_ps(mw2, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif
								buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me3)));
								mw3 = _mm512_mul_ps(mw3, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif
							}

							//low low
							//b
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mb0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw0, mss, mb0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw0, mss), mb0);
#endif
							//g
							const __m512i ssg_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep));
							__m512i ssg_32elem = _mm512_unpacklo_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg0 = _mm512_fmadd_ps(mw0, mss, mg0);
#else
							mg0 = _mm512_add_ps(_mm512_mul_ps(mw0, mss), mg0);
#endif
							//r
							const __m512i ssr_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep2));
							__m512i ssr_32elem = _mm512_unpacklo_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr0 = _mm512_fmadd_ps(mw0, mss, mr0);
#else
							mr0 = _mm512_add_ps(_mm512_mul_ps(mw0, mss), mr0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw0);


							//low high
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb1 = _mm512_fmadd_ps(mw1, mss, mb1);
#else
							mb1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mb1);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg1 = _mm512_fmadd_ps(mw1, mss, mg1);
#else
							mg1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mg1);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr1 = _mm512_fmadd_ps(mw1, mss, mr1);
#else
							mr1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mr1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw1);


							//high low
							//b
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb2 = _mm512_fmadd_ps(mw2, mss, mb2);
#else
							mb2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mb2);
#endif
							//g
							ssg_32elem = _mm512_unpackhi_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg2 = _mm512_fmadd_ps(mw2, mss, mg2);
#else
							mg2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mg2);
#endif
							//r
							ssr_32elem = _mm512_unpackhi_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr2 = _mm512_fmadd_ps(mw2, mss, mr2);
#else
							mr2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mr2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw2);


							//high high
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb3 = _mm512_fmadd_ps(mw3, mss, mb3);
#else
							mb3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mb3);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg3 = _mm512_fmadd_ps(mw3, mss, mg3);
#else
							mg3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mg3);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr3 = _mm512_fmadd_ps(mw3, mss, mr3);
#else
							mr3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mr3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw3);
						}
						mb0 = _mm512_div_ps(mb0, mtweight0);
						mb1 = _mm512_div_ps(mb1, mtweight1);
						mb2 = _mm512_div_ps(mb2, mtweight2);
						mb3 = _mm512_div_ps(mb3, mtweight3);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mb0), _mm512_cvtps_epi32(mb1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mb2), _mm512_cvtps_epi32(mb3)));
						mg0 = _mm512_div_ps(mg0, mtweight0);
						mg1 = _mm512_div_ps(mg1, mtweight1);
						mg2 = _mm512_div_ps(mg2, mtweight2);
						mg3 = _mm512_div_ps(mg3, mtweight3);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mg0), _mm512_cvtps_epi32(mg1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mg2), _mm512_cvtps_epi32(mg3)));
						mr0 = _mm512_div_ps(mr0, mtweight0);
						mr1 = _mm512_div_ps(mr1, mtweight1);
						mr2 = _mm512_div_ps(mr2, mtweight2);
						mr3 = _mm512_div_ps(mr3, mtweight3);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mr0), _mm512_cvtps_epi32(mr1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mr2), _mm512_cvtps_epi32(mr3)));

						_mm512_stream_epi8_color(d, a, b, c);
						d += 192;
					}
				}
#endif
				for (; i < dest->cols; i++)
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float ww = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = ((float)*(s + t_ofs[n]) - tV[count++]);
							double e = s0 * s0;
							s0 = ((float)*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = ((float)*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
							ww *= w[(int)sqrt(e)];
						}
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
#if __NLMF_POSTVENTION__
					static const __m512 float_min = _mm512_set1_ps(FLT_MIN);
#endif
					for (; i < dest->cols; i += 64)
					{
						__m512 mval0 = _mm512_setzero_ps();
						__m512 mval1 = _mm512_setzero_ps();
						__m512 mval2 = _mm512_setzero_ps();
						__m512 mval3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							const __m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 mw0 = _mm512_set1_ps(offset);
							__m512 mw1 = _mm512_set1_ps(offset);
							__m512 mw2 = _mm512_set1_ps(offset);
							__m512 mw3 = _mm512_set1_ps(offset);
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512i ms_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i ms_32elem = _mm512_unpacklo_epi8(ms_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								__m512i buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw0 = _mm512_mul_ps(mw0, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw0 = _mm512_max_ps(mw0, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw1 = _mm512_mul_ps(mw1, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw1 = _mm512_max_ps(mw1, float_min);
#endif
								ms_32elem = _mm512_unpackhi_epi8(ms_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw2 = _mm512_mul_ps(mw2, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw2 = _mm512_max_ps(mw2, float_min);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								buf = _mm512_cvtps_epi32(_mm512_and_ps(_mm512_sub_ps(ms, tVec[count++]), *(__m512 const*)v32f_absmask));
								mw3 = _mm512_mul_ps(mw3, _mm512_i32gather_ps(buf, w, 4));
#if __NLMF_POSTVENTION__
								mw3 = _mm512_max_ps(mw3, float_min);
#endif
							}
							//low low
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mval0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw0, mss, mval0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw0, mss), mval0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw0);

							//low high
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval1 = _mm512_fmadd_ps(mw1, mss, mval1);
#else
							mval1 = _mm512_add_ps(_mm512_mul_ps(mw1, mss), mval1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw1);

							//high low
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval2 = _mm512_fmadd_ps(mw2, mss, mval2);
#else
							mval2 = _mm512_add_ps(_mm512_mul_ps(mw2, mss), mval2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw2);


							//high high
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval3 = _mm512_fmadd_ps(mw3, mss, mval3);
#else
							mval3 = _mm512_add_ps(_mm512_mul_ps(mw3, mss), mval3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw3);
						}
						mval0 = _mm512_div_ps(mval0, mtweight0);
						mval1 = _mm512_div_ps(mval1, mtweight1);
						mval2 = _mm512_div_ps(mval2, mtweight2);
						mval3 = _mm512_div_ps(mval3, mtweight3);
						_mm512_stream_si512((__m512i*)d, _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mval0), _mm512_cvtps_epi32(mval1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mval2), _mm512_cvtps_epi32(mval3))));
						d += 64;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumv = 0;
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 1;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs((float)*(s + t_ofs[n]) - tV[count++]);
							e *= w[s0];
						}
						sumv += *(s)* e;
						sumw += e;
					}
					d[0] = sumv / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};


/*************************************************
	using Quantization LUT with "gather instruction" x 1
*************************************************/
class NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_64f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const double* weight;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_64f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const double* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_maxk(search_maxk), search_ofs(search_ofs), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		__m512d* tVec = (__m512d*)_mm_malloc(sizeof(__m512d)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif

		if (dest->channels() == 3)
		{
			const double* w = weight;
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					for (; i < dest->cols; i += 8)
					{
						__m512d mb = _mm512_setzero_pd();
						__m512d mg = _mm512_setzero_pd();
						__m512d mr = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d me = _mm512_setzero_pd();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512d s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif
								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif
								s0 = _mm512_sub_pd(_mm512_loadu_pd(s + t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(s0, s0, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(s0, s0), me);
#endif
							}
							__m512i buf = _mm512_cvtpd_epi64(_mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, me)));
							__m512d mw = _mm512_i64gather_pd(buf, w, 8);
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mb);
							mg = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep), mg);
							mr = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s + colorstep2), mr);
#else
							mb = _mm512_add_pd(mb, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
							mg = _mm512_add_pd(mg, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep)));
							mr = _mm512_add_pd(mr, _mm512_mul_pd(mw, _mm512_loadu_pd(s + colorstep2)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}

						mb = _mm512_div_pd(mb, mtweight);
						mg = _mm512_div_pd(mg, mtweight);
						mr = _mm512_div_pd(mr, mtweight);

						_mm512_stream_pd_color(d, mb, mg, mr);
						d += 24;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sumb = 0;
					double sumg = 0;
					double sumr = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = (*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
						}
						const double ww = w[(int)sqrt(e)];
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const double* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					for (; i < dest->cols; i += 8)
					{
						__m512d mval = _mm512_setzero_pd();
						__m512d mtweight = _mm512_setzero_pd();

						const double* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_pd(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const double* s = (sptr + s_ofs[l]);
							count = 0;
							__m512d me = _mm512_setzero_pd();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512d ms = _mm512_loadu_pd(s + t_ofs[n]);
								const __m512d diff = _mm512_sub_pd(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_pd(diff, diff, me);
#else
								me = _mm512_add_pd(_mm512_mul_pd(diff, diff), me);
#endif
							}
							__m512i buf = _mm512_cvtpd_epi64(_mm512_maskz_rcp14_pd((__mmask8)_MM_K0_REG, _mm512_maskz_rsqrt14_pd((__mmask8)_MM_K0_REG, me)));
							__m512d mw = _mm512_i64gather_pd(buf, w, 8);
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_pd(mw, _mm512_loadu_pd(s), mval);
#else
							mval = _mm512_add_pd(mval, _mm512_mul_pd(mw, _mm512_loadu_pd(s)));
#endif
							mtweight = _mm512_add_pd(mtweight, mw);
						}
						_mm512_stream_pd(d, _mm512_div_pd(mval, mtweight));
						d += 8;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					double sum = 0;
					double sumw = 0;

					const double* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<double> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const double* s = (sptr + s_ofs[l]);
						count = 0;
						double e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
						}
						const double ww = w[(int)sqrt(e)];
						sum += *(s)* ww;
						sumw += ww;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_32f_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_32f_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels(), 64);
#endif
		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
#if QUANTIZATION_SIZE_MODE == 1 
					const __m512 mQpara = _mm512_set1_ps(QUANTIZATION_SIZE_PRAMETER);
#endif
					for (; i < dest->cols; i += 16)
					{
						__m512 mb = _mm512_setzero_ps();
						__m512 mg = _mm512_setzero_ps();
						__m512 mr = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep);
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n] + colorstep2);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								__m512 s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n] + colorstep), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
								s0 = _mm512_sub_ps(_mm512_loadu_ps(s + +t_ofs[n] + colorstep2), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
							}
							
#if QUANTIZATION_MODE == 0
							me = _mm512_rcp14_ps(_mm512_rsqrt14_ps(me));
#endif
#if QUANTIZATION_SIZE_MODE
							me = _mm512_mul_ps(me, mQpara);
#endif
							__m512i buf = _mm512_cvtps_epi32(me);
							__m512 mw = _mm512_i32gather_ps(buf, w, 4);
#if __USE_FMA_INSTRUCTION__
							mb = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mb);
							mg = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep), mg);
							mr = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s + colorstep2), mr);
#else
							mb = _mm512_add_ps(mb, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
							mg = _mm512_add_ps(mg, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep)));
							mr = _mm512_add_ps(mr, _mm512_mul_ps(mw, _mm512_loadu_ps(s + colorstep2)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}

						mb = _mm512_div_ps(mb, mtweight);
						mg = _mm512_div_ps(mg, mtweight);
						mr = _mm512_div_ps(mr, mtweight);

						_mm512_stream_ps_color(d, mb, mg, mr);
						d += 48;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumb = 0;
					float sumg = 0;
					float sumr = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk * 3);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
						tV[count++] = *(sptr + t_ofs[n] + colorstep);
						tV[count++] = *(sptr + t_ofs[n] + colorstep2);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = (*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = (*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
						}
						const float ww = w[(int)sqrt(e)];
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					for (; i < dest->cols; i += 16)
					{
						__m512 mval = _mm512_setzero_ps();
						__m512 mtweight = _mm512_setzero_ps();

						const float* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							tVec[count++] = _mm512_loadu_ps(sptr + t_ofs[n]);
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const float* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512 s0 = _mm512_sub_ps(_mm512_loadu_ps(s + t_ofs[n]), tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me = _mm512_fmadd_ps(s0, s0, me);
#else
								me = _mm512_add_ps(_mm512_mul_ps(s0, s0), me);
#endif
							}
							__m512i buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me)));
							__m512 mw = _mm512_i32gather_ps(buf, w, 4);
#if __USE_FMA_INSTRUCTION__
							mval = _mm512_fmadd_ps(mw, _mm512_loadu_ps(s), mval);
#else
							mval = _mm512_add_ps(mval, _mm512_mul_ps(mw, _mm512_loadu_ps(s)));
#endif
							mtweight = _mm512_add_ps(mtweight, mw);
						}
						_mm512_stream_ps(d, _mm512_div_ps(mval, mtweight));
						d += 16;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sum = 0;
					float sumw = 0;

					const float* sptr = sptr_ + i;
					int count = 0;
					const int* t_ofs = &template_ofs[0];
					vector<float> tV(template_maxk);
					for (int n = 0; n < template_maxk; n++)
					{
						tV[count++] = *(sptr + t_ofs[n]);
					}

					//search loop
					const int* s_ofs = &search_ofs[0];
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const float* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs(*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
						}
						const double ww = w[(int)sqrt(e)];
						sum += *(s)* ww;
						sumw += ww;
					}
					d[0] = sum / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};

class NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_8u_AVX512 : public cv::ParallelLoopBody
{
private:
	const Mat * src;
	Mat* dest;
	const int* template_ofs;
	const int template_maxk;
	const int* search_ofs;
	const int search_maxk;
	const int templateWindowSizeX;
	const int templateWindowSizeY;
	const int searchWindowSizeX;
	const int searchWindowSizeY;
	const float* weight;

public:
	NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_8u_AVX512(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float* weight, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
		: src(&src_), dest(&dest_), template_ofs(template_ofs), template_maxk(template_maxk), search_ofs(search_ofs), search_maxk(search_maxk), templateWindowSizeX(templateWindowSizeX_), templateWindowSizeY(templateWindowSizeY_), searchWindowSizeX(searchWindowSizeX_), searchWindowSizeY(searchWindowSizeY_), weight(weight)
	{
	}

	void operator()(const cv::Range &range) const override
	{
		const int tr_x = templateWindowSizeX >> 1;
		const int sr_x = searchWindowSizeX >> 1;
		const int tr_y = templateWindowSizeY >> 1;
		const int sr_y = searchWindowSizeY >> 1;

#if CV_AVX512
		const bool haveAVX512 = checkHardwareSupport(CV_CPU_AVX_512F);
		__m512* tVec = (__m512*)_mm_malloc(sizeof(__m512)*templateWindowSizeX*templateWindowSizeY * dest->channels() * 4, 64);
#endif

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
					for (; i < dest->cols; i += 64)
					{
						__m512 mb0 = _mm512_setzero_ps();
						__m512 mb1 = _mm512_setzero_ps();
						__m512 mb2 = _mm512_setzero_ps();
						__m512 mb3 = _mm512_setzero_ps();

						__m512 mg0 = _mm512_setzero_ps();
						__m512 mg1 = _mm512_setzero_ps();
						__m512 mg2 = _mm512_setzero_ps();
						__m512 mg3 = _mm512_setzero_ps();

						__m512 mr0 = _mm512_setzero_ps();
						__m512 mr1 = _mm512_setzero_ps();
						__m512 mr2 = _mm512_setzero_ps();
						__m512 mr3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							__m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));

							temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n] + colorstep2));
							temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me0 = _mm512_setzero_ps();
							__m512 me1 = _mm512_setzero_ps();
							__m512 me2 = _mm512_setzero_ps();
							__m512 me3 = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								//b
								__m512i s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								__m512 s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif

								//g
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif

								//r
								s_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n] + colorstep2));
								s_32elem = _mm512_unpacklo_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(s0, s0, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(s0, s0, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me1);
#endif
								s_32elem = _mm512_unpackhi_epi8(s_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(s0, s0, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(s_32elem, zeroi));
								s0 = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(s0, s0, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(s0, s0), me3);
#endif
							}

							//low low
							__m512i buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me0)));
							__m512 mw = _mm512_i32gather_ps(buf, w, 4);
							//b
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mb0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw, mss, mb0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw, mss), mb0);
#endif
							//g
							const __m512i ssg_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep));
							__m512i ssg_32elem = _mm512_unpacklo_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg0 = _mm512_fmadd_ps(mw, mss, mg0);
#else
							mg0 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg0);
#endif
							//r
							const __m512i ssr_64elem = _mm512_loadu_si512((__m512i const*)(s + colorstep2));
							__m512i ssr_32elem = _mm512_unpacklo_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr0 = _mm512_fmadd_ps(mw, mss, mr0);
#else
							mr0 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw);


							//low high
							buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me1)));
							mw = _mm512_i32gather_ps(buf, w, 4);
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb1 = _mm512_fmadd_ps(mw, mss, mb1);
#else
							mb1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb1);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg1 = _mm512_fmadd_ps(mw, mss, mg1);
#else
							mg1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg1);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr1 = _mm512_fmadd_ps(mw, mss, mr1);
#else
							mr1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw);


							//high low
							buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me2)));
							mw = _mm512_i32gather_ps(buf, w, 4);
							//b
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb2 = _mm512_fmadd_ps(mw, mss, mb2);
#else
							mb2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb2);
#endif
							//g
							ssg_32elem = _mm512_unpackhi_epi8(ssg_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg2 = _mm512_fmadd_ps(mw, mss, mg2);
#else
							mg2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg2);
#endif
							//r
							ssr_32elem = _mm512_unpackhi_epi8(ssr_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr2 = _mm512_fmadd_ps(mw, mss, mr2);
#else
							mr2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw);


							//high high
							buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me3)));
							mw = _mm512_i32gather_ps(buf, w, 4);
							//b
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mb3 = _mm512_fmadd_ps(mw, mss, mb3);
#else
							mb3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mb3);
#endif
							//g
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssg_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mg3 = _mm512_fmadd_ps(mw, mss, mg3);
#else
							mg3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mg3);
#endif
							//r
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssr_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mr3 = _mm512_fmadd_ps(mw, mss, mr3);
#else
							mr3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mr3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw);
						}
						mb0 = _mm512_div_ps(mb0, mtweight0);
						mb1 = _mm512_div_ps(mb1, mtweight1);
						mb2 = _mm512_div_ps(mb2, mtweight2);
						mb3 = _mm512_div_ps(mb3, mtweight3);
						const __m512i a = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mb0), _mm512_cvtps_epi32(mb1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mb2), _mm512_cvtps_epi32(mb3)));
						mg0 = _mm512_div_ps(mg0, mtweight0);
						mg1 = _mm512_div_ps(mg1, mtweight1);
						mg2 = _mm512_div_ps(mg2, mtweight2);
						mg3 = _mm512_div_ps(mg3, mtweight3);
						const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mg0), _mm512_cvtps_epi32(mg1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mg2), _mm512_cvtps_epi32(mg3)));
						mr0 = _mm512_div_ps(mr0, mtweight0);
						mr1 = _mm512_div_ps(mr1, mtweight1);
						mr2 = _mm512_div_ps(mr2, mtweight2);
						mr3 = _mm512_div_ps(mr3, mtweight3);
						const __m512i c = _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mr0), _mm512_cvtps_epi32(mr1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mr2), _mm512_cvtps_epi32(mr3)));

						_mm512_stream_epi8_color(d, a, b, c);
						d += 192;
					}
				}
#endif
				for (; i < dest->cols; i++)
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = ((float)*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
							s0 = ((float)*(s + t_ofs[n] + colorstep) - tV[count++]);
							e += s0 * s0;
							s0 = ((float)*(s + t_ofs[n] + colorstep2) - tV[count++]);
							e += s0 * s0;
						}
						const double ww = w[(int)sqrt(e)];
						sumb += *(s)* ww;
						sumg += *(s + colorstep) * ww;
						sumr += *(s + colorstep2) * ww;
						sumw += ww;
					}
					d[0] = sumb / sumw;
					d[1] = sumg / sumw;
					d[2] = sumr / sumw;
					d += 3;
				}
			}
		}
		else if (dest->channels() == 1)
		{
			const float* w = weight;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				int i = 0;
#if CV_AVX512
				if (haveAVX512)
				{
					static const __m512i zeroi = _mm512_setzero_si512();
					for (; i < dest->cols; i += 64)
					{
						__m512 mval0 = _mm512_setzero_ps();
						__m512 mval1 = _mm512_setzero_ps();
						__m512 mval2 = _mm512_setzero_ps();
						__m512 mval3 = _mm512_setzero_ps();

						__m512 mtweight0 = _mm512_setzero_ps();
						__m512 mtweight1 = _mm512_setzero_ps();
						__m512 mtweight2 = _mm512_setzero_ps();
						__m512 mtweight3 = _mm512_setzero_ps();

						const uchar* sptr = sptr_ + i;
						int count = 0;
						const int* t_ofs = &template_ofs[0];
						for (int n = 0; n < template_maxk; n++)
						{
							const __m512i temp_64elem = _mm512_loadu_si512((__m512i*)(sptr + t_ofs[n]));
							__m512i temp_32elem = _mm512_unpacklo_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
							temp_32elem = _mm512_unpackhi_epi8(temp_64elem, zeroi);
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(temp_32elem, zeroi));
							tVec[count++] = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(temp_32elem, zeroi));
						}

						//search loop
						const int* s_ofs = &search_ofs[0];
						for (int l = 0; l < search_maxk; l++)
						{
							//template loop
							const uchar* s = (sptr + s_ofs[l]);
							count = 0;
							__m512 me0 = _mm512_setzero_ps();
							__m512 me1 = _mm512_setzero_ps();
							__m512 me2 = _mm512_setzero_ps();
							__m512 me3 = _mm512_setzero_ps();
							for (int n = 0; n < template_maxk; n++)
							{
								// computing color L2 norm
								const __m512i ms_64elem = _mm512_loadu_si512((__m512i const*)(s + t_ofs[n]));
								__m512i ms_32elem = _mm512_unpacklo_epi8(ms_64elem, zeroi);
								__m512 ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								__m512 diff = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me0 = _mm512_fmadd_ps(diff, diff, me0);
#else
								me0 = _mm512_add_ps(_mm512_mul_ps(diff, diff), me0);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								diff = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me1 = _mm512_fmadd_ps(diff, diff, me1);
#else
								me1 = _mm512_add_ps(_mm512_mul_ps(diff, diff), me1);
#endif
								ms_32elem = _mm512_unpackhi_epi8(ms_64elem, zeroi);
								ms = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ms_32elem, zeroi));
								diff = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me2 = _mm512_fmadd_ps(diff, diff, me2);
#else
								me2 = _mm512_add_ps(_mm512_mul_ps(diff, diff), me2);
#endif
								ms = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ms_32elem, zeroi));
								diff = _mm512_sub_ps(ms, tVec[count++]);
#if __USE_FMA_INSTRUCTION__
								me3 = _mm512_fmadd_ps(diff, diff, me3);
#else
								me3 = _mm512_add_ps(_mm512_mul_ps(diff, diff), me3);
#endif
							}

							//low low
							__m512i buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me0)));
							__m512 mw = _mm512_i32gather_ps(buf, w, 4);
							const __m512i ssb_64elem = _mm512_loadu_si512((__m512i const*)s);
							__m512i ssb_32elem = _mm512_unpacklo_epi8(ssb_64elem, zeroi);
							__m512 mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
							mval0 =
#if __USE_FMA_INSTRUCTION__
								_mm512_fmadd_ps(mw, mss, mval0);
#else
								_mm512_add_ps(_mm512_mul_ps(mw, mss), mval0);
#endif
							mtweight0 = _mm512_add_ps(mtweight0, mw);

							//low high
							buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me1)));
							mw = _mm512_i32gather_ps(buf, w, 4);
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval1 = _mm512_fmadd_ps(mw, mss, mval1);
#else
							mval1 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval1);
#endif
							mtweight1 = _mm512_add_ps(mtweight1, mw);

							//high low
							buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me2)));
							mw = _mm512_i32gather_ps(buf, w, 4);
							ssb_32elem = _mm512_unpackhi_epi8(ssb_64elem, zeroi);
							mss = _mm512_cvtepi32_ps(_mm512_unpacklo_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval2 = _mm512_fmadd_ps(mw, mss, mval2);
#else
							mval2 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval2);
#endif
							mtweight2 = _mm512_add_ps(mtweight2, mw);


							//high high
							buf = _mm512_cvtps_epi32(_mm512_rcp14_ps(_mm512_rsqrt14_ps(me3)));
							mw = _mm512_i32gather_ps(buf, w, 4);
							mss = _mm512_cvtepi32_ps(_mm512_unpackhi_epi16(ssb_32elem, zeroi));
#if __USE_FMA_INSTRUCTION__
							mval3 = _mm512_fmadd_ps(mw, mss, mval3);
#else
							mval3 = _mm512_add_ps(_mm512_mul_ps(mw, mss), mval3);
#endif
							mtweight3 = _mm512_add_ps(mtweight3, mw);
						}
						mval0 = _mm512_div_ps(mval0, mtweight0);
						mval1 = _mm512_div_ps(mval1, mtweight1);
						mval2 = _mm512_div_ps(mval2, mtweight2);
						mval3 = _mm512_div_ps(mval3, mtweight3);
						_mm512_stream_si512((__m512i*)d, _mm512_packus_epi16(_mm512_packs_epi32(_mm512_cvtps_epi32(mval0), _mm512_cvtps_epi32(mval1)), _mm512_packs_epi32(_mm512_cvtps_epi32(mval2), _mm512_cvtps_epi32(mval3))));
						d += 64;
					}
				}
#endif
				for (; i < dest->cols; i++)
				{
					float sumv = 0;
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
					for (int l = 0; l < search_maxk; l++)
					{
						//template loop
						const uchar* s = (sptr + s_ofs[l]);
						count = 0;
						float e = 0;
						for (int n = 0; n < template_maxk; n++)
						{
							// computing color L2 norm
							int s0 = abs((float)*(s + t_ofs[n]) - tV[count++]);
							e += s0 * s0;
						}
						const float ww = w[(int)sqrt(e)];
						sumv += *(s)* ww;
						sumw += ww;
					}
					d[0] = sumv / sumw;
					d++;
				}
			}
		}
#if CV_AVX512
		_mm_free(tVec);
#endif
	}
};



namespace nlmf
{
	void nonLocalMeansFilter_AVX512_64f(const Mat& src, Mat& dest, const Size templateWindowSize, const Size searchWindowSize, const double h, const int borderType, const WEIGHT_MODE weightingMethod)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int cn = src.channels();
		const int templateH = templateWindowSize.width >> 1;
		const int templateV = templateWindowSize.height >> 1;
		const int searchH = searchWindowSize.width >> 1;
		const int searchV = searchWindowSize.height >> 1;
		const int bbx = templateH + searchH;
		const int bby = templateV + searchV;
		const double gauss_color_coeff = -(1. / (h*h));

		const int dpad = (8 - src.cols % 8) % 8;
		const int spad = (8 - (src.cols + 2 * bbx) % 8) % 8;

		Mat dst = Mat::zeros(Size(src.cols + dpad, src.rows), dest.type());

		Mat im;
		if (cn == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, im);
		}

		vector<int> template_offset(templateWindowSize.area());
		int* template_ofs = &template_offset[0];
		int template_maxk = 0;
		setSpaceKernel(template_ofs, template_maxk, templateH, templateV, im.cols, true);
		vector<int> search_offset(searchWindowSize.area());
		int* search_ofs = &search_offset[0];
		int search_maxk = 0;
		setSpaceKernel(search_ofs, search_maxk, searchH, searchV, im.cols, true);

		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			const NonlocalMeansFilterInvorker_EXP_64f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, h, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
			//weight computation;
			const int value_range = 256;
			vector<double> weight(value_range);
			double* w = &weight[0];

#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area()*cn;
			const double max_digits = floor(log2(DBL_MAX / (255.f*searchWindowSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - tS) / tS);
			const double bias = pow(2, log2(DBL_MAX / (255.f*searchWindowSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#else
			const double bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], DBL_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_LUT_Setx3_64f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SETxN:
		{
			//weight computation;
			const int value_range = 442; //ceil(sqrt(255 ^2 *3));
			vector<double> weight(value_range);
			double* w = &weight[0];
#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area();
			const double max_digits = floor(log2(DBL_MAX / (255.f*searchWindowSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - tS) / tS);
			const double bias = pow(2, log2(DBL_MAX / (255.f*searchWindowSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#else
			const double bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], DBL_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_64f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SETx1:
		{
			//weight computation;
			const int tS = templateWindowSize.area();
			const int tD = tS * src.channels();
			const int value_range = (int)ceil(sqrt(255 * 255 * tD));
			vector<double> weight(value_range);
			double* w = &weight[0];
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
				double v = std::exp(aw);
#if __NLMF_POSTVENTION__
				v = max(v, DBL_MIN);
#endif
				w[i] = v;
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_64f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
		{
			//weight computation;
			const int value_range = 256;
			vector<double> weight(value_range);
			double* w = &weight[0];

#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area()*cn;
			const double max_digits = floor(log2(DBL_MAX / (255.f*searchWindowSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - tS) / tS);
			const double bias = pow(2, log2(DBL_MAX / (255.f*searchWindowSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#else
			const double bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], DBL_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_LUT_Gatherx3_64f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHERxN:
		{
			//weight computation;
			const int value_range = 442; //ceil(sqrt(255 ^2 *3));
			vector<double> weight(value_range);
			double* w = &weight[0];
#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area();
			const double max_digits = floor(log2(DBL_MAX / (255.f*searchWindowSize.area())) - log2(DBL_MIN));
			const double bias_digits = floor((max_digits - tS) / tS);
			const double bias = pow(2, log2(DBL_MAX / (255.f*searchWindowSize.area())));
			const double exp_clip_val = log(pow(2, -bias_digits)) + DBL_EPSILON;
#else
			const double bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], DBL_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_64f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHERx1:
		{
			//weight computation;
			const int tS = templateWindowSize.width * templateWindowSize.height;
			const int tD = tS * src.channels();
			const int value_range = (int)ceil(sqrt(255 * 255 * tD));
			vector<double> weight(value_range);
			double* w = &weight[0];
			for (int i = 0; i < value_range; i++)
			{
				double aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_DP);
#endif
				double v = std::exp(aw);
#if __NLMF_POSTVENTION__
				v = max(v, DBL_MIN);
#endif
				w[i] = v;
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_64f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		default:
			break;
		}
		Mat(dst(Rect(0, 0, dest.cols, dest.rows))).copyTo(dest);
	}

	void nonLocalMeansFilter_AVX512_32f(const Mat& src, Mat& dest, const Size templateWindowSize, const Size searchWindowSize, const float h, const int borderType, const WEIGHT_MODE weightingMethod)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int cn = src.channels();
		const int templateH = templateWindowSize.width >> 1;
		const int templateV = templateWindowSize.height >> 1;
		const int searchH = searchWindowSize.width >> 1;
		const int searchV = searchWindowSize.height >> 1;
		const int bbx = templateH + searchH;
		const int bby = templateV + searchV;
		const float gauss_color_coeff = -(1.f / (h*h));

		const int dpad = (16 - src.cols % 16) % 16;
		const int spad = (16 - (src.cols + 2 * bbx) % 16) % 16;

		Mat dst = Mat::zeros(Size(src.cols + dpad, src.rows), dest.type());

		Mat im;
		if (cn == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, im);
		}

		vector<int> template_offset(templateWindowSize.area());
		int* template_ofs = &template_offset[0];
		int template_maxk = 0;
		setSpaceKernel(template_ofs, template_maxk, templateH, templateV, im.cols, true);
		vector<int> search_offset(searchWindowSize.area());
		int* search_ofs = &search_offset[0];
		int search_maxk = 0;
		setSpaceKernel(search_ofs, search_maxk, searchH, searchV, im.cols, true);

		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			const NonlocalMeansFilterInvorker_EXP_32f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, h, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
			//weight computation;
			const int value_range = 256;
			vector<float> weight(value_range);
			float* w = &weight[0];

#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area()*cn;
			const float max_digits = floor(log2(FLT_MAX / (255.f*searchWindowSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - tS) / tS);
			const float bias = pow(2, log2(FLT_MAX / (255.f*searchWindowSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], FLT_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_LUT_Setx3_32f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SETxN:
		{
			//weight computation;
			const int value_range = 442; //ceil(sqrt(255 ^2 *3));
			vector<float> weight(value_range);
			float* w = &weight[0];
#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area();
			const float max_digits = floor(log2(FLT_MAX / (255.f*searchWindowSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - tS) / tS);
			const float bias = pow(2, log2(FLT_MAX / (255.f*searchWindowSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], FLT_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_32f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SETx1:
		{
			//weight computation;
			const int tS = templateWindowSize.width * templateWindowSize.height;
			const int tD = tS * src.channels();
			const int value_range = (int)ceil(sqrt(255 * 255 * tD));
			vector<float> weight(value_range);
			float* w = &weight[0];
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				float v = std::exp(aw);
#if __NLMF_POSTVENTION__
				v = max(v, FLT_MIN);
#endif
				w[i] = v;
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_32f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
		{
			//weight computation;
			const int value_range = 256;
			vector<float> weight(value_range);
			float* w = &weight[0];

#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area()*cn;
			const float max_digits = floor(log2(FLT_MAX / (255.f*searchWindowSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - tS) / tS);
			const float bias = pow(2, log2(FLT_MAX / (255.f*searchWindowSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], FLT_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_LUT_Gatherx3_32f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHERxN:
		{
			//weight computation;
			const int value_range = 442; //ceil(sqrt(255 ^2 *3));
			vector<float> weight(value_range);
			float* w = &weight[0];
#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area();
			const float max_digits = floor(log2(FLT_MAX / (255.f*searchWindowSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - tS) / tS);
			const float bias = pow(2, log2(FLT_MAX / (255.f*searchWindowSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], FLT_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_32f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHERx1:
		{
			//weight computation;
			const int tS = templateWindowSize.width * templateWindowSize.height;
			const int tD = tS * src.channels();
#if QUANTIZATION_MODE == 0
#if QUANTIZATION_SIZE_MODE == 0
			const int value_range = (int)ceil(sqrt(255 * 255 * tD));
#else
			const int value_range = (int)ceil(sqrt(255 * 255 * tD) * QUANTIZATION_SIZE_PRAMETER);
#endif
#else
#if QUANTIZATION_SIZE_MODE == 0
			const int value_range = (int)ceil((255 * 255 * tD));
#else
			const int value_range = (int)ceil((255 * 255 * tD * QUANTIZATION_SIZE_PRAMETER));
#endif
#endif
			vector<float> weight(value_range);
			float* w = &weight[0];
			for (int i = 0; i < value_range; i++)
			{
#if QUANTIZATION_MODE == 0
#if QUANTIZATION_SIZE_MODE == 0
				float aw = i * i * gauss_color_coeff;
#else
				float aw = (i / QUANTIZATION_SIZE_PRAMETER) * (i / QUANTIZATION_SIZE_PRAMETER) * gauss_color_coeff;
#endif
#else
#if QUANTIZATION_SIZE_MODE == 0
				float aw = i * gauss_color_coeff;
#else
				float aw = (i / QUANTIZATION_SIZE_PRAMETER) * gauss_color_coeff;
#endif
#endif
#if __NLMF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				float v = std::exp(aw);
#if __NLMF_POSTVENTION__
				v = max(v, FLT_MIN);
#endif
				w[i] = v;
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_32f_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		default:
			break;
		}
		Mat(dst(Rect(0, 0, dest.cols, dest.rows))).copyTo(dest);
	}

	void nonLocalMeansFilter_AVX512_8u(const Mat& src, Mat& dest, const Size templateWindowSize, const Size searchWindowSize, const float h, const int borderType, const WEIGHT_MODE weightingMethod)
	{
		if (dest.empty())dest = Mat::zeros(src.size(), src.type());

		const int cn = src.channels();
		const int templateH = templateWindowSize.width >> 1;
		const int templateV = templateWindowSize.height >> 1;
		const int searchH = searchWindowSize.width >> 1;
		const int searchV = searchWindowSize.height >> 1;
		const int bbx = templateH + searchH;
		const int bby = templateV + searchV;
		const float gauss_color_coeff = -(1.f / (h*h));

		const int dpad = (64 - src.cols % 64) % 64;
		const int spad = (64 - (src.cols + 2 * bbx) % 64) % 64;

		Mat dst = Mat::zeros(Size(src.cols + dpad, src.rows), dest.type());

		Mat im;
		if (cn == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx + spad, borderType);
		}
		else if (cn == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx + spad, borderType);
			cvtColorBGR2PLANE(temp, im);
		}

		vector<int> template_offset(templateWindowSize.area());
		int* template_ofs = &template_offset[0];
		int template_maxk = 0;
		setSpaceKernel(template_ofs, template_maxk, templateH, templateV, im.cols, true);
		vector<int> search_offset(searchWindowSize.area());
		int* search_ofs = &search_offset[0];
		int search_maxk = 0;
		setSpaceKernel(search_ofs, search_maxk, searchH, searchV, im.cols, true);

		switch (weightingMethod)
		{
		case WEIGHT_VECTOR_EXP:
		{
			const NonlocalMeansFilterInvorker_EXP_8u_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, h, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_LUT_SET:
		{
			//weight computation;
			const int value_range = 256;
			vector<float> weight(value_range);
			float* w = &weight[0];

#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area()*cn;
			const float max_digits = floor(log2(FLT_MAX / (255.f*searchWindowSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - tS) / tS);
			const float bias = pow(2, log2(FLT_MAX / (255.f*searchWindowSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], FLT_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_LUT_Setx3_8u_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SETxN:
		{
			//weight computation;
			const int value_range = 442; //ceil(sqrt(255 ^2 *3));
			vector<float> weight(value_range);
			float* w = &weight[0];
#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area();
			const float max_digits = floor(log2(FLT_MAX / (255.f*searchWindowSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - tS) / tS);
			const float bias = pow(2, log2(FLT_MAX / (255.f*searchWindowSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], FLT_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_SetxN_8u_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_SETx1:
		{
			//weight computation;
			const int tS = templateWindowSize.width * templateWindowSize.height;
			const int tD = tS * src.channels();
			const int value_range = (int)ceil(sqrt(255 * 255 * tD));
			vector<float> weight(value_range);
			float* w = &weight[0];
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				float v = std::exp(aw);
#if __NLMF_POSTVENTION__
				v = max(v, FLT_MIN);
#endif
				w[i] = v;
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_Setx1_8u_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_LUT_GATHER:
		{
			//weight computation;
			const int value_range = 256;
			vector<float> weight(value_range);
			float* w = &weight[0];

#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area()*cn;
			const float max_digits = floor(log2(FLT_MAX / (255.f*searchWindowSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - tS) / tS);
			const float bias = pow(2, log2(FLT_MAX / (255.f*searchWindowSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], FLT_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_LUT_Gatherx3_8u_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHERxN:
		{
			//weight computation;
			const int value_range = 442; //ceil(sqrt(255 ^2 *3));
			vector<float> weight(value_range);
			float* w = &weight[0];
#if __NLMF_PREVENTION__
			const int tS = templateWindowSize.area();
			const float max_digits = floor(log2(FLT_MAX / (255.f*searchWindowSize.area())) - log2(FLT_MIN));
			const float bias_digits = floor((max_digits - tS) / tS);
			const float bias = pow(2, log2(FLT_MAX / (255.f*searchWindowSize.area())));
			const float exp_clip_val = log(pow(2, -bias_digits)) + FLT_EPSILON;
#else
			const float bias = 1;
#endif
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, exp_clip_val);
#endif
				w[i] = exp(aw);
#if __NLMF_POSTVENTION__
				w[i] = max(w[i], FLT_MIN);
#endif
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_GatherxN_8u_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs, bias);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		case WEIGHT_RANGE_QUANTIZATION_LUT_GATHERx1:
		{
			//weight computation;
			const int tS = templateWindowSize.width * templateWindowSize.height;
			const int tD = tS * src.channels();
			const int value_range = (int)ceil(sqrt(255 * 255 * tD));
			vector<float> weight(value_range);
			float* w = &weight[0];
			for (int i = 0; i < value_range; i++)
			{
				float aw = i * i * gauss_color_coeff;
#if __NLMF_PREVENTION__
				aw = max(aw, EXP_ARGUMENT_CLIP_VALUE_SP);
#endif
				float v = std::exp(aw);
#if __NLMF_POSTVENTION__
				v = max(v, FLT_MIN);
#endif
				w[i] = v;
			}

			const NonlocalMeansFilterInvorker_QuantizationLUT_Gatherx1_8u_AVX512 body(im, dst, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, w, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dst.rows), body);
			break;
		}
		default:
			break;
		}
		Mat(dst(Rect(0, 0, dest.cols, dest.rows))).copyTo(dest);
	}

	void nonLocalMeansFilter_AVX512(InputArray src_, OutputArray dest, const Size templateWindowSize, const Size searchWindowSize, const double h, const int borderType, const WEIGHT_MODE weightingMethod)
	{
		if (dest.empty() || dest.type() != src_.type() || dest.size() != src_.size()) dest.create(src_.size(), src_.type());
		Mat src = src_.getMat();
		Mat dst = dest.getMat();

		if (src.depth() == CV_8U)
		{
			nonLocalMeansFilter_AVX512_8u(src, dst, templateWindowSize, searchWindowSize, (float)h, borderType, weightingMethod);
		}
		else if (src.depth() == CV_32F)
		{
			nonLocalMeansFilter_AVX512_32f(src, dst, templateWindowSize, searchWindowSize, (float)h, borderType, weightingMethod);
		}
		else if (src.depth() == CV_64F)
		{
			nonLocalMeansFilter_AVX512_64f(src, dst, templateWindowSize, searchWindowSize, h, borderType, weightingMethod);
		}
	}
}
