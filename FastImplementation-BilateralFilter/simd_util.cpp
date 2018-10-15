#include "simd_util.h"
#include <opencv2/opencv.hpp>

#if CV_AVX512
void _mm512_stream_epi8_color(void* dst, const __m512i b, const __m512i g, const __m512i r)
{
	static const __m512i mask1 = _mm512_set_epi8(
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0,
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0
	);
	static const __m512i mask2 = _mm512_set_epi8(
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5,
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5
	);
	static const __m512i mask3 = _mm512_set_epi8(
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10,
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10
	);
	static const __m512i pmask1 = _mm512_set_epi64(3, 2, 1, 0, 1, 0, 1, 0);
	static const __m512i pmask2 = _mm512_set_epi64(5, 4, 5, 4, 3, 2, 3, 2);
	static const __m512i pmask3 = _mm512_set_epi64(7, 6, 7, 6, 7, 6, 5, 4);

	static const __mmask64 blendMask1 = 0x4924924924924924;
	static const __mmask64 blendMask2 = 0x2492492492492492;
	static const __mmask64 blendMask3 = 0x9249249249249249;

	const __m512i aa = _mm512_shuffle_epi8(b, mask1);
	const __m512i bb = _mm512_shuffle_epi8(g, mask2);
	const __m512i cc = _mm512_shuffle_epi8(r, mask3);

	__m512i aaa = _mm512_permutexvar_epi64(pmask1, aa);
	__m512i bbb = _mm512_permutexvar_epi64(pmask1, bb);
	__m512i ccc = _mm512_permutexvar_epi64(pmask1, cc);
	_mm512_stream_si512((uchar*)dst + 0, _mm512_mask_blend_epi8(blendMask1, _mm512_mask_blend_epi8(blendMask2, aaa, bbb), ccc));

	aaa = _mm512_permutexvar_epi64(pmask2, aa);
	bbb = _mm512_permutexvar_epi64(pmask2, bb);
	ccc = _mm512_permutexvar_epi64(pmask2, cc);
	_mm512_stream_si512((uchar*)dst + 64, _mm512_mask_blend_epi8(blendMask2, _mm512_mask_blend_epi8(blendMask3, aaa, bbb), ccc));

	aaa = _mm512_permutexvar_epi64(pmask3, aa);
	bbb = _mm512_permutexvar_epi64(pmask3, bb);
	ccc = _mm512_permutexvar_epi64(pmask3, cc);
	_mm512_stream_si512((uchar*)dst + 128, _mm512_mask_blend_epi8(blendMask3, _mm512_mask_blend_epi8(blendMask1, aaa, bbb), ccc));
}

void _mm512_stream_ps_color(void* dst, const __m512 b, const __m512 g, const __m512 r)
{
#if __USE_SCATTER_INSTRUCTION__
	static const __m512i idx = _mm512_set_epi32(45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0);
	_mm512_i32scatter_ps((float*)dst + 0, idx, b, 4);
	_mm512_i32scatter_ps((float*)dst + 1, idx, g, 4);
	_mm512_i32scatter_ps((float*)dst + 2, idx, r, 4);
#else
	static const __m512i permuteIndexB = _mm512_setr_epi32(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	static const __m512i permuteIndexG = _mm512_setr_epi32(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	static const __m512i permuteIndexR = _mm512_setr_epi32(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

	static const __mmask16 blendMask1 = 0x4924;
	static const __mmask16 blendMask2 = 0x2492;
	static const __mmask16 blendMask3 = 0x9249;

	const __m512 aa = _mm512_permutexvar_ps(permuteIndexB, b);
	const __m512 bb = _mm512_permutexvar_ps(permuteIndexG, g);
	const __m512 cc = _mm512_permutexvar_ps(permuteIndexR, r);

	_mm512_stream_ps((float*)dst + 0, _mm512_mask_blend_ps(blendMask1, _mm512_mask_blend_ps(blendMask2, aa, bb), cc));
	_mm512_stream_ps((float*)dst + 16, _mm512_mask_blend_ps(blendMask2, _mm512_mask_blend_ps(blendMask3, aa, bb), cc));
	_mm512_stream_ps((float*)dst + 32, _mm512_mask_blend_ps(blendMask3, _mm512_mask_blend_ps(blendMask1, aa, bb), cc));
#endif
}

void _mm512_stream_pd_color(void* dst, const __m512d b, const __m512d g, const __m512d r)
{
#if __USE_SCATTER_INSTRUCTION__
	static const __m512i idx = _mm512_set_epi64(21, 18, 15, 12, 9, 6, 3, 0);
	_mm512_i64scatter_pd((double*)dst + 0, idx, b, 8);
	_mm512_i64scatter_pd((double*)dst + 1, idx, g, 8);
	_mm512_i64scatter_pd((double*)dst + 2, idx, r, 8);
#else
	static const __m512i permuteIndexB = _mm512_setr_epi64(0, 3, 6, 1, 4, 7, 2, 5);
	static const __m512i permuteIndexG = _mm512_setr_epi64(5, 0, 3, 6, 1, 4, 7, 2);
	static const __m512i permuteIndexR = _mm512_setr_epi64(2, 5, 0, 3, 6, 1, 4, 7);

	static const __mmask16 blendMask1 = 0b00100100;
	static const __mmask16 blendMask2 = 0b10010010;
	static const __mmask16 blendMask3 = 0b01001001;

	const __m512d aa = _mm512_permutexvar_pd(permuteIndexB, b);
	const __m512d bb = _mm512_permutexvar_pd(permuteIndexG, g);
	const __m512d cc = _mm512_permutexvar_pd(permuteIndexR, r);

	_mm512_stream_pd((double*)dst + 0, _mm512_mask_blend_pd(blendMask1, _mm512_mask_blend_pd(blendMask2, aa, bb), cc));
	_mm512_stream_pd((double*)dst + 8, _mm512_mask_blend_pd(blendMask3, _mm512_mask_blend_pd(blendMask1, aa, bb), cc));
	_mm512_stream_pd((double*)dst + 16, _mm512_mask_blend_pd(blendMask2, _mm512_mask_blend_pd(blendMask3, aa, bb), cc));
#endif
}
#endif

#if CV_AVX && !CV_AVX2
void _mm256_stream_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r)
{
	const __m256d b0 = _mm256_permute2f128_pd(b, b, 0b00000000);
	const __m256d b1 = _mm256_permute2f128_pd(b, b, 0b00010001);

	const __m256d g0 = _mm256_shuffle_pd(g, g, 0b0101);

	const __m256d r0 = _mm256_permute2f128_pd(r, r, 0b00000000);
	const __m256d r1 = _mm256_permute2f128_pd(r, r, 0b00010001);

	_mm256_stream_pd(static_cast<double*>(dst) + 0, _mm256_blend_pd(_mm256_blend_pd(b0, g0, 0b0010), r0, 0b0100));
	_mm256_stream_pd(static_cast<double*>(dst) + 4, _mm256_blend_pd(_mm256_blend_pd(b1, g0, 0b1001), r0, 0b0010));
	_mm256_stream_pd(static_cast<double*>(dst) + 8, _mm256_blend_pd(_mm256_blend_pd(b1, g0, 0b0100), r1, 0b1001));
}

void _mm256_stream_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
{
	static const int smask1 = _MM_SHUFFLE(1, 2, 3, 0);
	static const int smask2 = _MM_SHUFFLE(2, 3, 0, 1);
	static const int smask3 = _MM_SHUFFLE(3, 0, 1, 2);

	static const int bmask1 = 0x44;
	static const int bmask2 = 0x22;

	static const int pmask1 = 0x20;
	static const int pmask2 = 0x30;
	static const int pmask3 = 0x31;

	const __m256 aa = _mm256_shuffle_ps(b, b, smask1);
	const __m256 bb = _mm256_shuffle_ps(g, g, smask2);
	const __m256 cc = _mm256_shuffle_ps(r, r, smask3);
	const __m256 bval = _mm256_blend_ps(_mm256_blend_ps(aa, cc, bmask1), bb, bmask2);
	const __m256 gval = _mm256_blend_ps(_mm256_blend_ps(cc, bb, bmask1), aa, bmask2);
	const __m256 rval = _mm256_blend_ps(_mm256_blend_ps(bb, aa, bmask1), cc, bmask2);

	_mm256_stream_ps((float*)dst + 0, _mm256_permute2f128_ps(bval, rval, pmask1));
	_mm256_stream_ps((float*)dst + 8, _mm256_permute2f128_ps(gval, bval, pmask2));
	_mm256_stream_ps((float*)dst + 16, _mm256_permute2f128_ps(rval, gval, pmask3));
}
#endif

#if CV_AVX2
void _mm256_stream_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r)
{
	static const int pmask1 = 0b01101100;
	static const int pmask2 = 0b10110001;
	static const int pmask3 = 0b11000110;

	const __m256d aa = _mm256_permute4x64_pd(b, pmask1);
	const __m256d bb = _mm256_permute4x64_pd(g, pmask2);
	const __m256d cc = _mm256_permute4x64_pd(r, pmask3);

	_mm256_stream_pd(static_cast<double*>(dst) + 0, _mm256_blend_pd(_mm256_blend_pd(aa, bb, 0b0010), cc, 0b0100));
	_mm256_stream_pd(static_cast<double*>(dst) + 4, _mm256_blend_pd(_mm256_blend_pd(aa, bb, 0b1001), cc, 0b0010));
	_mm256_stream_pd(static_cast<double*>(dst) + 8, _mm256_blend_pd(_mm256_blend_pd(aa, bb, 0b0100), cc, 0b1001));
}

void _mm256_stream_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r)
{
#if 1
	static const __m256i pmask1 = _mm256_set_epi32(5, 2, 7, 4, 1, 6, 3, 0);
	static const __m256i pmask2 = _mm256_set_epi32(2, 7, 4, 1, 6, 3, 0, 5);
	static const __m256i pmask3 = _mm256_set_epi32(7, 4, 1, 6, 3, 0, 5, 2);

	const __m256 aa = _mm256_permutevar8x32_ps(b, pmask1);
	const __m256 bb = _mm256_permutevar8x32_ps(g, pmask2);
	const __m256 cc = _mm256_permutevar8x32_ps(r, pmask3);

	_mm256_stream_ps(static_cast<float*>(dst) + 0, _mm256_blend_ps(_mm256_blend_ps(aa, bb, 0b10010010), cc, 0b00100100));
	_mm256_stream_ps(static_cast<float*>(dst) + 8, _mm256_blend_ps(_mm256_blend_ps(aa, bb, 0b00100100), cc, 0b01001001));
	_mm256_stream_ps(static_cast<float*>(dst) + 16, _mm256_blend_ps(_mm256_blend_ps(aa, bb, 0b01001001), cc, 0b10010010));
#else
	static const int smask1 = _MM_SHUFFLE(1, 2, 3, 0);
	static const int smask2 = _MM_SHUFFLE(2, 3, 0, 1);
	static const int smask3 = _MM_SHUFFLE(3, 0, 1, 2);

	static const int bmask1 = 0x44;
	static const int bmask2 = 0x22;

	static const int pmask1 = 0x20;
	static const int pmask2 = 0x30;
	static const int pmask3 = 0x31;

	const __m256 aa = _mm256_shuffle_ps(b, b, smask1);
	const __m256 bb = _mm256_shuffle_ps(g, g, smask2);
	const __m256 cc = _mm256_shuffle_ps(r, r, smask3);
	__m256 bval = _mm256_blend_ps(_mm256_blend_ps(aa, cc, bmask1), bb, bmask2);
	__m256 gval = _mm256_blend_ps(_mm256_blend_ps(cc, bb, bmask1), aa, bmask2);
	__m256 rval = _mm256_blend_ps(_mm256_blend_ps(bb, aa, bmask1), cc, bmask2);

	_mm256_stream_ps((float*)dst + 0, _mm256_permute2f128_ps(bval, rval, pmask1));
	_mm256_stream_ps((float*)dst + 8, _mm256_permute2f128_ps(gval, bval, pmask2));
	_mm256_stream_ps((float*)dst + 16, _mm256_permute2f128_ps(rval, gval, pmask3));
#endif
}

void _mm256_stream_epi8_color(void* dst, const __m256i b, const __m256i g, const __m256i r)
{
	static const __m256i mask1 = _mm256_set_epi8(
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0,
		5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0
	);
	static const __m256i mask2 = _mm256_set_epi8(
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5,
		10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5
	);
	static const __m256i mask3 = _mm256_set_epi8(
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10,
		15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10
	);

	static const __m256i bmask1 = _mm256_set_epi8(
		255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255,
		0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0
	);
	static const __m256i bmask2 = _mm256_set_epi8(
		255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255,
		255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255
	);

	const __m256i aa = _mm256_shuffle_epi8(b, mask1);
	const __m256i bb = _mm256_shuffle_epi8(g, mask2);
	const __m256i cc = _mm256_shuffle_epi8(r, mask3);

	__m256i aaa = _mm256_permute2x128_si256(aa, aa, 0x00);
	__m256i bbb = _mm256_permute2x128_si256(bb, bb, 0x00);
	__m256i ccc = _mm256_permute2x128_si256(cc, cc, 0x00);
	_mm256_stream_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst)), _mm256_blendv_epi8(ccc, _mm256_blendv_epi8(aaa, bbb, bmask1), bmask2));
	_mm256_stream_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst) + 32), _mm256_blendv_epi8(cc, _mm256_blendv_epi8(bb, aa, bmask2), bmask1));
	aaa = _mm256_permute2x128_si256(aa, aa, 0x11);
	bbb = _mm256_permute2x128_si256(bb, bb, 0x11);
	ccc = _mm256_permute2x128_si256(cc, cc, 0x11);
	_mm256_stream_si256(reinterpret_cast<__m256i*>(static_cast<uchar*>(dst) + 64), _mm256_blendv_epi8(aaa, _mm256_blendv_epi8(bbb, ccc, bmask1), bmask2));
}
#endif

#if CV_SSE4_1
void _mm_stream_pd_color(void* dst, const __m128d b, const __m128d g, const __m128d r)
{
	const __m128d g0 = _mm_shuffle_pd(g, g, 0b01);

	_mm_stream_pd((double*)dst + 0, _mm_blend_pd(b, g0, 0b10));
	_mm_stream_pd((double*)dst + 2, _mm_blend_pd(b, r, 0b01));
	_mm_stream_pd((double*)dst + 4, _mm_blend_pd(g0, r, 0b10));
}

void _mm_stream_ps_color(void* dst, const __m128 b, const __m128 g, const __m128 r)
{
	static const int smask1 = _MM_SHUFFLE(3, 0, 1, 2);
	static const int smask2 = _MM_SHUFFLE(1, 2, 3, 0);
	static const int smask3 = _MM_SHUFFLE(2, 3, 0, 1);

	const __m128 aa = _mm_shuffle_ps(r, r, smask1);
	const __m128 bb = _mm_shuffle_ps(b, b, smask2);
	const __m128 cc = _mm_shuffle_ps(g, g, smask3);

	_mm_stream_ps((float*)dst + 0, _mm_blend_ps(_mm_blend_ps(bb, aa, 4), cc, 2));
	_mm_stream_ps((float*)dst + 4, _mm_blend_ps(_mm_blend_ps(cc, bb, 4), aa, 2));
	_mm_stream_ps((float*)dst + 8, _mm_blend_ps(_mm_blend_ps(aa, cc, 4), bb, 2));
}

void _mm_stream_epi8_color(void* dst, const __m128i b, const __m128i g, const  __m128i r)
{
	static const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	static const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	static const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

	static const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
	static const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

	const __m128i aa = _mm_shuffle_epi8(b, mask1);
	const __m128i bb = _mm_shuffle_epi8(g, mask2);
	const __m128i cc = _mm_shuffle_epi8(r, mask3);

	_mm_stream_si128((__m128i*)((uchar*)dst + 0), _mm_blendv_epi8(cc, _mm_blendv_epi8(aa, bb, bmask1), bmask2));
	_mm_stream_si128((__m128i*)((uchar*)dst + 16), _mm_blendv_epi8(bb, _mm_blendv_epi8(aa, cc, bmask2), bmask1));
	_mm_stream_si128((__m128i*)((uchar*)dst + 32), _mm_blendv_epi8(cc, _mm_blendv_epi8(bb, aa, bmask2), bmask1));
}
#endif