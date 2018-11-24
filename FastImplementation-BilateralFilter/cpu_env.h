#pragma once
#define _CRT_SECURE_NO_WARNINGS

#if defined CV_AVX512
#undef CV_AVX512
#endif

#define CV_AVX512 1
#define __AVX512__

#if defined CV_AVX2
#undef CV_AVX2
#endif
#define CV_AVX2 1

#if defined CV_AVX
#undef CV_AVX
#endif
#define CV_AVX 1

#if defined CV_SSE4_2
#undef CV_SSE4_2
#endif
#define CV_SSE4_2 1

#if defined CV_SSE4_1
#undef CV_SSE4_1
#endif
#define CV_SSE4_1 1

//AVX512
#define __USE_SCATTER_INSTRUCTION__ 1

//AVX512, AVX2
#if defined __USE_FMA_INSTRUCTION__
#undef __USE_FMA_INSTRUCTION__
#endif
#if CV_AVX && !CV_AVX2
#define __USE_FMA_INSTRUCTION__ 0
#else CV_AVX2
#define __USE_FMA_INSTRUCTION__ 1
#endif


#ifdef __INTEL_COMPILER
#define __USE_INTEL_EXP__ 1
#else
#define __USE_INTEL_EXP__ 0
#endif

//quantization mode
//0: n * sqrt(x)
//1: x/n
//TODO: only support impl -> BF AVX512 RangeQLUT gather, BF AVX512 MergedQLUT gather, NLMF AVX512 RangeQLUT gather
#define QUANTIZATION_MODE 0
#define QUANTIZATION_SIZE_MODE 0
//if not 0, size of quantization LUT is changed.
#if QUANTIZATION_SIZE_MODE == 1
#if QUANTIZATION_MODE == 0
#define QUANTIZATION_SIZE_PRAMETER (1.f)
#else
#define QUANTIZATION_SIZE_PRAMETER (1.f)
#endif
#else
#define QUANTIZATION_SIZE_PRAMETER 0
#endif


/*********************
	denormal number 
 *********************/
#define EXP_ARGUMENT_CLIP_VALUE_SP (-87.3f)	//ln(FLT_MIN)
#define EXP_ARGUMENT_CLIP_VALUE_DP (-708.3)	//ln(DBL_MIN)

// Gaussian range filter
#define __GRF_PREVENTION__	1
#define __GRF_POSTVENTION__	0
// bilateral filter
#define __BF_PREVENTION__	1
#define __BF_POSTVENTION__	0
// non-local means filter
#define __NLMF_PREVENTION__	1
#define __NLMF_POSTVENTION__	0
// gaussian non-local means filter
#define __BNLMF_PREVENTION__	1
#define __BNLMF_POSTVENTION__	0

bool show_mxcsr(const bool showState = true, const bool showMask = false, const bool isClaer = false);