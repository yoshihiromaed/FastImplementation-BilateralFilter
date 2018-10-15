#pragma once

#include <immintrin.h>
#include <opencv2/opencv.hpp>

inline void _mm256_transpose32_epi8(__m256i* dst, const __m256i* src)
{
	__m256i tmp[32], tmpp[32];

	for (int i = 0; i < 8; i++)
	{
		tmp[2 * i + 0] = _mm256_unpacklo_epi8(src[i], src[i + 8]);
		tmp[2 * i + 1] = _mm256_unpackhi_epi8(src[i], src[i + 8]);
		tmp[2 * i + 16] = _mm256_unpacklo_epi8(src[i + 16], src[i + 24]);
		tmp[2 * i + 17] = _mm256_unpackhi_epi8(src[i + 16], src[i + 24]);
	}
	for (int i = 0; i < 8; i++)
	{
		tmpp[2 * i + 0] = _mm256_unpacklo_epi8(tmp[i], tmp[i + 8]);
		tmpp[2 * i + 1] = _mm256_unpackhi_epi8(tmp[i], tmp[i + 8]);
		tmpp[2 * i + 16] = _mm256_unpacklo_epi8(tmp[i + 16], tmp[i + 24]);
		tmpp[2 * i + 17] = _mm256_unpackhi_epi8(tmp[i + 16], tmp[i + 24]);
	}
	for (int i = 0; i < 8; i++)
	{
		tmp[2 * i + 0] = _mm256_unpacklo_epi8(tmpp[i], tmpp[i + 8]);
		tmp[2 * i + 1] = _mm256_unpackhi_epi8(tmpp[i], tmpp[i + 8]);
		tmp[2 * i + 16] = _mm256_unpacklo_epi8(tmpp[i + 16], tmpp[i + 24]);
		tmp[2 * i + 17] = _mm256_unpackhi_epi8(tmpp[i + 16], tmpp[i + 24]);
	}
	for (int i = 0; i < 8; i++)
	{
		tmpp[2 * i + 0] = _mm256_unpacklo_epi8(tmp[i], tmp[i + 8]);
		tmpp[2 * i + 1] = _mm256_unpackhi_epi8(tmp[i], tmp[i + 8]);
		tmpp[2 * i + 16] = _mm256_unpacklo_epi8(tmp[i + 16], tmp[i + 24]);
		tmpp[2 * i + 17] = _mm256_unpackhi_epi8(tmp[i + 16], tmp[i + 24]);
	}
	for (int i = 0; i < 16; i++)
	{
		dst[i + 0] = _mm256_permute2x128_si256(tmpp[i], tmpp[i + 16], 0x20);
		dst[i + 16] = _mm256_permute2x128_si256(tmpp[i], tmpp[i + 16], 0x31);
	}
}

/**
* Transposition in AVX of 8 PRECISION vectors
*/
#define ___MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2, out3, out4, out5, out6, out7, __in0, __in1, __in2, __in3, __in4, __in5, __in6, __in7, __out0, __out1, __out2, __out3, __out4, __out5, __out6, __out7, __tmp0, __tmp1, __tmp2, __tmp3, __tmp4, __tmp5, __tmp6, __tmp7, __tmpp0, __tmpp1, __tmpp2, __tmpp3, __tmpp4, __tmpp5, __tmpp6, __tmpp7) \
  do { \
    __m256 __in0 = (in0), __in1 = (in1), __in2 = (in2), __in3 = (in3), __in4 = (in4), __in5 = (in5), __in6 = (in6), __in7 = (in7); \
    __m256 __tmp0, __tmp1, __tmp2, __tmp3, __tmp4, __tmp5, __tmp6, __tmp7; \
    __m256 __tmpp0, __tmpp1, __tmpp2, __tmpp3, __tmpp4, __tmpp5, __tmpp6, __tmpp7; \
    __m256 __out0, __out1, __out2, __out3, __out4, __out5, __out6, __out7; \
    __tmp0  = _mm256_unpacklo_ps(__in0, __in1); \
    __tmp1  = _mm256_unpackhi_ps(__in0, __in1); \
    __tmp2  = _mm256_unpacklo_ps(__in2, __in3); \
    __tmp3  = _mm256_unpackhi_ps(__in2, __in3); \
    __tmp4  = _mm256_unpacklo_ps(__in4, __in5); \
    __tmp5  = _mm256_unpackhi_ps(__in4, __in5); \
    __tmp6  = _mm256_unpacklo_ps(__in6, __in7); \
    __tmp7  = _mm256_unpackhi_ps(__in6, __in7); \
    __tmpp0 = _mm256_shuffle_ps(__tmp0, __tmp2, 0x44); \
    __tmpp1 = _mm256_shuffle_ps(__tmp0, __tmp2, 0xEE); \
    __tmpp2 = _mm256_shuffle_ps(__tmp1, __tmp3, 0x44); \
    __tmpp3 = _mm256_shuffle_ps(__tmp1, __tmp3, 0xEE); \
    __tmpp4 = _mm256_shuffle_ps(__tmp4, __tmp6, 0x44); \
    __tmpp5 = _mm256_shuffle_ps(__tmp4, __tmp6, 0xEE); \
    __tmpp6 = _mm256_shuffle_ps(__tmp5, __tmp7, 0x44); \
    __tmpp7 = _mm256_shuffle_ps(__tmp5, __tmp7, 0xEE); \
    __out0  = _mm256_permute2f128_ps(__tmpp0, __tmpp4, 0x20); \
    __out1  = _mm256_permute2f128_ps(__tmpp1, __tmpp5, 0x20); \
    __out2  = _mm256_permute2f128_ps(__tmpp2, __tmpp6, 0x20); \
    __out3  = _mm256_permute2f128_ps(__tmpp3, __tmpp7, 0x20); \
    __out4  = _mm256_permute2f128_ps(__tmpp0, __tmpp4, 0x31); \
    __out5  = _mm256_permute2f128_ps(__tmpp1, __tmpp5, 0x31); \
    __out6  = _mm256_permute2f128_ps(__tmpp2, __tmpp6, 0x31); \
    __out7  = _mm256_permute2f128_ps(__tmpp3, __tmpp7, 0x31); \
    (out0)  = __out0, (out1) = __out1, (out2) = __out2, (out3) = __out3, (out4) = __out4, (out5) = __out5, (out6) = __out6, (out7) = __out7; \
          } while (0)
#define _MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7) \
      ___MM256_TRANSPOSE8_PS(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4, in5, in6, in7, \
          __in0##__LINE__, __in1##__LINE__, __in2##__LINE__, __in3##__LINE__, __in4##__LINE__, __in5##__LINE__, __in6##__LINE__, __in7##__LINE__, \
          __out0##__LINE__, __out1##__LINE__, __out2##__LINE__, __out3##__LINE__, __out4##__LINE__, __out5##__LINE__, __out6##__LINE__, __out7##__LINE__, \
          __tmp0##__LINE__, __tmp1##__LINE__, __tmp2##__LINE__, __tmp3##__LINE__, __tmp4##__LINE__, __tmp5##__LINE__, __tmp6##__LINE__, __tmp7##__LINE__, \
          __tmpp0##__LINE__, __tmpp1##__LINE__, __tmpp2##__LINE__, __tmpp3##__LINE__, __tmpp4##__LINE__, __tmpp5##__LINE__, __tmpp6##__LINE__, __tmpp7##__LINE__)

/**
* Transposition in AVX of 4 PRECISION vectors
*/
#define ___MM256_TRANSPOSE4_PD(in0, in1, in2, in3, out0, out1, out2, out3, __in0, __in1, __in2, __in3, __out0, __out1, __out2, __out3, __tmp0, __tmp1, __tmp2, __tmp3) \
  do { \
    __m256d __in0 = (in0), __in1 = (in1), __in2 = (in2), __in3 = (in3); \
    __m256d __tmp0, __tmp1, __tmp2, __tmp3; \
    __m256d __out0, __out1, __out2, __out3; \
    __tmp0 = _mm256_shuffle_pd(__in0, __in1, 0x0); \
    __tmp1 = _mm256_shuffle_pd(__in0, __in1, 0xf); \
    __tmp2 = _mm256_shuffle_pd(__in2, __in3, 0x0); \
    __tmp3 = _mm256_shuffle_pd(__in2, __in3, 0xf); \
    __out0 = _mm256_permute2f128_pd(__tmp0, __tmp2, 0x20); \
    __out1 = _mm256_permute2f128_pd(__tmp1, __tmp3, 0x20); \
    __out2 = _mm256_permute2f128_pd(__tmp0, __tmp2, 0x31); \
    __out3 = _mm256_permute2f128_pd(__tmp1, __tmp3, 0x31); \
    (out0) = __out0, (out1) = __out1, (out2) = __out2, (out3) = __out3; \
          } while (0)
#define _MM256_TRANSPOSE4_PD(in0, in1, in2, in3) \
      ___MM256_TRANSPOSE4_PD(in0, in1, in2, in3, in0, in1, in2, in3, \
          __in0##__LINE__, __in1##__LINE__, __in2##__LINE__, __in3##__LINE__, \
          __out0##__LINE__, __out1##__LINE__, __out2##__LINE__, __out3##__LINE__, \
          __tmp0##__LINE__, __tmp1##__LINE__, __tmp2##__LINE__, __tmp3##__LINE__)


// for AVX shift

#ifdef _MSC_VER
#define MIE_ALIGN(x) __declspec(align(x))
#else
#define MIE_ALIGN(x) __attribute__((aligned(x)))
#endif

template<bool> struct RANGE;

// variable shift amount

static const unsigned char MIE_ALIGN(16) shiftPattern[] =
{
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
	0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
};

inline __m256i _mm256i_shift_left(__m256i v, int shift)
{
	assert(0 <= shift && shift <= 32);
	__m256i mask = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 32 + shift)));
	__m256i a0 = _mm256_shuffle_epi8(v, mask);

	__m256i a1 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm256_extracti128_si256(v, 1), _mm_loadu_si128((const __m128i*)(shiftPattern + 16 + shift))));
	__m256i a2 = _mm256_permute2x128_si256(a1, a1, 0x80);

	return _mm256_or_si256(a0, a2);
}

inline __m256i _mm256i_shift_right(__m256i v, int shift)
{
	assert(0 <= shift && shift <= 32);
	__m256i mask = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 32 - shift)));
	__m256i a0 = _mm256_shuffle_epi8(v, mask);

	__m256i a1 = _mm256_castsi128_si256(_mm_shuffle_epi8(_mm256_castsi256_si128(v), _mm_loadu_si128((const __m128i*)(shiftPattern + 48 - shift))));
	__m256i a2 = _mm256_permute2x128_si256(a1, a1, 0x08);

	return _mm256_or_si256(a0, a2);
}

inline __m256i _mm256i_funnel_shift_left(__m256i a, __m256i b, int shift) 
{
	assert(0 <= shift && shift <= 32);

	__m256i ma = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 32 + shift)));
	__m256i a0 = _mm256_shuffle_epi8(a, ma);

	__m256i mb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + shift)));
	__m256i b0 = _mm256_shuffle_epi8(b, mb);

	const __m256i pat = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 16 + shift)));
	__m256i ab = _mm256_permute2x128_si256(a, b, 0x21);
	__m256i ab0 = _mm256_shuffle_epi8(ab, pat);

	return _mm256_or_si256(ab0, _mm256_or_si256(a0, b0));
}

inline __m256i _mm256i_funnel_shift_right(__m256i a, __m256i b, int shift) 
{
	assert(0 <= shift && shift <= 32);

	__m256i mb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 64 - shift)));
	__m256i b0 = _mm256_shuffle_epi8(b, mb);

	__m256i ma = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 32 - shift)));
	__m256i a0 = _mm256_shuffle_epi8(a, ma);

	const __m256i pat = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(shiftPattern + 48 - shift)));
	__m256i ab = _mm256_permute2x128_si256(a, b, 0x03);
	__m256i ab0 = _mm256_shuffle_epi8(ab, pat);

	return _mm256_or_si256(ab0, _mm256_or_si256(a0, b0));
}

inline __m256i _mm256_alignr2_epi8(const __m256i m1, const __m256i m2, const int n)
{
	uchar CV_DECL_ALIGNED(32) buf[64];

	_mm256_store_si256((__m256i *)&buf[0], m1);
	_mm256_store_si256((__m256i *)&buf[32], m2);

	return _mm256_loadu_si256((__m256i *)&buf[n]);
}
