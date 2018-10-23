# Fast Implementation of BilateralFilter
This code is implimentation of the following paper:

Y. Maeda, N. Fukushima, and H. Matsuo, "Effective Implementation of Edge-Preserving Filtering on CPU Microarchitectures," Applied Sciences, vol. 8, no. 10, July 2018. 
[Paper](https://www.mdpi.com/2076-3417/8/10/1985)

## Overview
The code is fast implementation of bilateral filtering, non-local filtering, gaussian range filtering, bilateral non-local means filtering.

The implementation is based on their naive algorithms; thus, their acceleration algorithms, such as separable approaches and constant time algorithms, are not employed.


The code is written in C++ with SIMD intrinsic (SSE/AVX/AVX2/AVX512) and "Concurrency" parallel optimization.

## Requirements
* Visual Studio 2017
* Intel Compiler 18
* OpenCV 3.x

## Usage
cpu_env.h
```cpp
#define CV_AVX512 1 //if AVX512 is available, set to 1
#define CV_AVX2 1 //if AVX2 is available, set to 1
#define CV_AVX 1  //if AVX is available, set to 1
#define CV_SSE4_2 1 //if SSE4.2 is available, set to 1
#define CV_SSE4_1 1 //if SSE4.1 is available, set to 1

//AVX512
#define __USE_SCATTER_INSTRUCTION__ 1 //if scatter instruction is available, set to 1
#define __USE_FMA_INSTRUCTION__ 1 //if FMA instruction is available, set to 1

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
```

If `__XXX_PREVENTION__`=1, the implementation is fastest.
