#pragma once
#include "cpu_env.h"
#include <immintrin.h>

#if CV_AVX512
void _mm512_stream_epi8_color(void* dst, const __m512i b, const __m512i g, const __m512i r);
void _mm512_stream_ps_color(void* dst, const __m512 b, const __m512 g, const __m512 r);
void _mm512_stream_pd_color(void* dst, const __m512d b, const __m512d g, const __m512d r);
#endif

#if CV_AVX || CV_AVX2
void _mm256_stream_pd_color(void* dst, const __m256d b, const __m256d g, const __m256d r);
void _mm256_stream_ps_color(void* dst, const __m256 b, const __m256 g, const __m256 r);
#endif
#if CV_AVX2
void _mm256_stream_epi8_color(void* dst, const __m256i b, const __m256i g, const __m256i r);
#endif

#if CV_SSE4_1
void _mm_stream_pd_color(void* dst, const __m128d b, const __m128d g, const __m128d r);
void _mm_stream_ps_color(void* dst, const __m128 b, const __m128 g, const __m128 r);
void _mm_stream_epi8_color(void* dst, const __m128i b, const __m128i g, const  __m128i r);
#endif
