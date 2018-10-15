#include "color.h"

#pragma warning(disable:4309)

using namespace std;
using namespace cv;

void mergeLineInterleaveBGRAVX_8u(const Mat& src, Mat& dest)
{
	const int sstep = src.cols * 3;
	const int dstep = dest.cols * 3;

	const uchar* bptr = src.ptr<uchar>(0);
	const uchar* gptr = src.ptr<uchar>(1);
	const uchar* rptr = src.ptr<uchar>(2);
	uchar* dptr = dest.ptr<uchar>(0);

	static const __m256i mask1 = _mm256_set_epi8(5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0);
	static const __m256i mask2 = _mm256_set_epi8(10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5);
	static const __m256i mask3 = _mm256_set_epi8(15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10);

	static const __m256i bmask1 = _mm256_set_epi8
	(255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
	static const __m256i bmask2 = _mm256_set_epi8
	(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255);

	__m256i a, b, c;

	for (int j = 0; j < dest.rows; j++)
	{
		for (int i = 0; i < dest.cols; i += 32)
		{
			a = _mm256_load_si256((__m256i*)(bptr + i));
			b = _mm256_load_si256((__m256i*)(gptr + i));
			c = _mm256_load_si256((__m256i*)(rptr + i));

			a = _mm256_shuffle_epi8(a, mask1);
			b = _mm256_shuffle_epi8(b, mask2);
			c = _mm256_shuffle_epi8(c, mask3);
			__m256i aa = _mm256_permute2x128_si256(a, a, 0x00);
			__m256i bb = _mm256_permute2x128_si256(b, b, 0x00);
			__m256i cc = _mm256_permute2x128_si256(c, c, 0x00);

			uchar* dptri = dptr + 3 * i;
			_mm256_stream_si256((__m256i*)(dptri), _mm256_blendv_epi8(cc, _mm256_blendv_epi8(aa, bb, bmask1), bmask2));
			_mm256_stream_si256((__m256i*)(dptri + 32), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask2), bmask1));
			aa = _mm256_permute2x128_si256(a, a, 0x11);
			bb = _mm256_permute2x128_si256(b, b, 0x11);
			cc = _mm256_permute2x128_si256(c, c, 0x11);
			_mm256_stream_si256((__m256i*)(dptri + 64), _mm256_blendv_epi8(aa, _mm256_blendv_epi8(bb, cc, bmask1), bmask2));
		}
		bptr += sstep;
		gptr += sstep;
		rptr += sstep;
		dptr += dstep;
	}
}

void mergeLineInterleaveBGRAVX_32f(const Mat& src, Mat& dest)
{
	const int sstep = src.cols * 3;
	const int dstep = dest.cols * 3;

	const float* bptr = src.ptr<float>(0);
	const float* gptr = src.ptr<float>(1);
	const float* rptr = src.ptr<float>(2);
	float* dptr = dest.ptr<float>(0);

	__m256 a, b, c;

	for (int j = 0; j < dest.rows; j++)
	{
		for (int i = 0; i < dest.cols; i += 8)
		{
			a = _mm256_load_ps(bptr + i);
			b = _mm256_load_ps(gptr + i);
			c = _mm256_load_ps(rptr + i);

			const __m256 aa = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
			const __m256 bb = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
			const __m256 cc = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
			a = _mm256_blend_ps(_mm256_blend_ps(aa, cc, 0x44), bb, 0x22);
			b = _mm256_blend_ps(_mm256_blend_ps(cc, bb, 0x44), aa, 0x22);
			c = _mm256_blend_ps(_mm256_blend_ps(bb, aa, 0x44), cc, 0x22);

			float* dptri = dptr + 3 * i;
			_mm256_stream_ps(dptri, _mm256_permute2f128_ps(a, c, 0x20));
			_mm256_stream_ps(dptri + 8, _mm256_permute2f128_ps(b, a, 0x30));
			_mm256_stream_ps(dptri + 16, _mm256_permute2f128_ps(c, b, 0x31));
		}
		bptr += sstep;
		gptr += sstep;
		rptr += sstep;
		dptr += dstep;
	}
}

void mergeLineInterleaveBGRAVX_32fcast(const Mat& src, Mat& dest)
{
	Mat a, b;
	src.convertTo(a, CV_32F);
	mergeLineInterleaveBGRAVX_32f(a, b);
	b.convertTo(dest, src.type());
}

void mergeLineInterleaveBGRAVX(cv::InputArray src_, cv::OutputArray dest_)
{
	int pad;
	if (src_.depth() == CV_8U)
		pad = (32 - src_.size().width % 32) % 32;
	else
		pad = (8 - src_.size().width % 8) % 8;

	dest_.create(Size(src_.size().width + pad, src_.size().height / 3), CV_MAKETYPE(src_.depth(), 3));
	Mat src = src_.getMat();
	Mat dest = dest_.getMat();
	copyMakeBorder(src, src, 0, 0, 0, pad, BORDER_REPLICATE);

	if (src.depth() == CV_8U)
	{
		mergeLineInterleaveBGRAVX_8u(src, dest);
	}
	else if (src.depth() == CV_32F)
	{
		mergeLineInterleaveBGRAVX_32f(src, dest);
	}
	else
	{
		mergeLineInterleaveBGRAVX_32fcast(src, dest);
	}
}


void splitBGRLineInterleaveAVX_8u(const Mat& src, Mat& dest)
{
	const int sstep = src.cols * 3;
	const int dstep = dest.cols * 3;

	const uchar* s = src.ptr<uchar>(0);
	uchar* B = dest.ptr<uchar>(0);//line by line interleave
	uchar* G = dest.ptr<uchar>(1);
	uchar* R = dest.ptr<uchar>(2);

	//BGR BGR BGR BGR BGR B x2
	//GR BGR BGR BGR BGR BG x2
	//R BGR BGR BGR BGR BGR x2
	//BBBBBBGGGGGRRRRR shuffle
	static const __m256i mask1 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14,
		0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
	//GGGGGBBBBBBRRRRR shuffle
	static const __m256i smask1 = _mm256_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15,
		6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
	static const __m256i ssmask1 = _mm256_setr_epi8(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

	//GGGGGGBBBBBRRRRR shuffle
	static const __m256i mask2 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
		0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	static const __m256i ssmask2 = _mm256_setr_epi8(0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10,
		0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10);

	static const __m256i bmask1 = _mm256_setr_epi8
	(255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	static const __m256i bmask2 = _mm256_setr_epi8
	(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0);

	static const __m256i bmask3 = _mm256_setr_epi8
	(255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	static const __m256i bmask4 = _mm256_setr_epi8
	(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0);

	__m256i a, b, c;

	for (int j = 0; j < src.rows; j++)
	{
		int i = 0;
		for (; i < src.cols; i += 32)
		{
			const __m256i _a = _mm256_load_si256((__m256i*)(s + 3 * i + 0));
			const __m256i _b = _mm256_load_si256((__m256i*)(s + 3 * i + 32));
			const __m256i _c = _mm256_load_si256((__m256i*)(s + 3 * i + 64));
			a = _mm256_permute2x128_si256(_a, _b, 0x30);
			b = _mm256_permute2x128_si256(_a, _c, 0x21);
			c = _mm256_permute2x128_si256(_b, _c, 0x30);
			a = _mm256_shuffle_epi8(a, mask1);
			b = _mm256_shuffle_epi8(b, mask2);
			c = _mm256_shuffle_epi8(c, mask2);
			_mm256_stream_si256((__m256i*)(B + i), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask1), bmask2));

			a = _mm256_shuffle_epi8(a, smask1);
			b = _mm256_shuffle_epi8(b, smask1);
			c = _mm256_shuffle_epi8(c, ssmask1);
			_mm256_stream_si256((__m256i*)(G + i), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask3), bmask2));

			a = _mm256_shuffle_epi8(a, ssmask1);
			b = _mm256_shuffle_epi8(b, ssmask2);
			c = _mm256_shuffle_epi8(c, ssmask1);
			_mm256_stream_si256((__m256i*)(R + i), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask3), bmask4));
		}
		R += dstep;
		G += dstep;
		B += dstep;
		s += sstep;
	}
}

void splitBGRLineInterleaveAVX_32f(const Mat& src, Mat& dest)
{
	const int size = src.size().area();
	const int dstep = src.cols * 3;
	const int sstep = src.cols * 3;

	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(1);
	float* R = dest.ptr<float>(2);

	for (int j = 0; j < src.rows; j++)
	{
		int i = 0;
		for (; i < src.cols; i += 8)
		{
			__m256 aa = _mm256_load_ps((s + 3 * i));
			__m256 bb = _mm256_load_ps((s + 3 * i + 8));
			__m256 cc = _mm256_load_ps((s + 3 * i + 16));
			const __m256 a = _mm256_permute2f128_ps(aa, bb, 0x30);
			const __m256 b = _mm256_permute2f128_ps(aa, cc, 0x21);
			const __m256 c = _mm256_permute2f128_ps(bb, cc, 0x30);

			aa = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
			aa = _mm256_blend_ps(aa, b, 0x44);
			cc = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
			aa = _mm256_blend_ps(aa, cc, 0x88);
			_mm256_stream_ps((B + i), aa);

			aa = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
			bb = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
			bb = _mm256_blend_ps(bb, aa, 0x11);
			cc = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
			bb = _mm256_blend_ps(bb, cc, 0x88);
			_mm256_stream_ps((G + i), bb);

			aa = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
			bb = _mm256_blend_ps(aa, b, 0x22);
			cc = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
			cc = _mm256_blend_ps(bb, cc, 0xcc);
			_mm256_stream_ps((R + i), cc);

		}
		R += dstep;
		G += dstep;
		B += dstep;
		s += sstep;
	}
}

void splitBGRLineInterleaveAVX_64f(const Mat& src, Mat& dest)
{
	const int size = src.size().area();
	const int dstep = src.cols * 3;
	const int sstep = src.cols * 3;

	const double* s = src.ptr<double>(0);
	double* B = dest.ptr<double>(0);//line by line interleave
	double* G = dest.ptr<double>(1);
	double* R = dest.ptr<double>(2);

	for (int j = 0; j < src.rows; j++)
	{
		int i = 0;
		for (; i < src.cols; i += 4)
		{
			const __m256d aa = _mm256_load_pd((s + 3 * i));
			const __m256d bb = _mm256_load_pd((s + 3 * i + 4));
			const __m256d cc = _mm256_load_pd((s + 3 * i + 8));

#if CV_AVX2
			__m256d a = _mm256_blend_pd(aa, bb, 0b0110);
			__m256d b = _mm256_blend_pd(a, cc, 0b0010);
			__m256d c = _mm256_permute4x64_pd(b, 0b01101100);
			_mm256_stream_pd((B + i), c);

			a = _mm256_blend_pd(aa, bb, 0b1001);
			b = _mm256_blend_pd(a, cc, 0b0100);
			c = _mm256_permute4x64_pd(b, 0b10110001);
			_mm256_stream_pd((G + i), c);

			a = _mm256_blend_pd(aa, bb, 0b1011);
			b = _mm256_blend_pd(a, cc, 0b1001);
			c = _mm256_permute4x64_pd(b, 0b11000110);
			_mm256_stream_pd((R + i), c);
#else
			__m256d a = _mm256_blend_pd(_mm256_permute2f128_pd(aa, aa, 0b00000001), aa, 0b0001);
			__m256d b = _mm256_blend_pd(_mm256_permute2f128_pd(cc, cc, 0b00000000), bb, 0b0100);
			__m256d c = _mm256_blend_pd(a, b, 0b1100);
			_mm256_stream_pd((B + i), c);

			a = _mm256_blend_pd(aa, bb, 0b1001);
			b = _mm256_blend_pd(a, cc, 0b0100);
			c = _mm256_permute_pd(b, 0b0101);
			_mm256_stream_pd((G + i), c);

			a = _mm256_blend_pd(_mm256_permute2f128_pd(aa, aa, 0b0001), bb, 0b0010);
			b = _mm256_blend_pd(_mm256_permute2f128_pd(cc, cc, 0b0001), cc, 0b1000);
			c = _mm256_blend_pd(a, b, 0b1100);
			_mm256_stream_pd((R + i), c);
#endif
		}
		R += dstep;
		G += dstep;
		B += dstep;
		s += sstep;
	}
}

void splitBGRLineInterleaveAVX_32fcast(const Mat& src, Mat& dest)
{
	Mat a, b;
	src.convertTo(a, CV_32F);
	dest.convertTo(b, CV_32F);
	splitBGRLineInterleaveAVX_32f(a, b);
	b.convertTo(dest, src.type());
}

void splitBGRLineInterleaveAVX(cv::InputArray src_, cv::OutputArray dest_)
{
	dest_.create(Size(src_.size().width, src_.size().height * 3), src_.depth());
	Mat src = src_.getMat();
	Mat dest = dest_.getMat();
	if (src.type() == CV_MAKE_TYPE(CV_8U, 3))
	{
		CV_Assert(src.cols % 32 == 0);
		splitBGRLineInterleaveAVX_8u(src, dest);
	}
	else if (src.type() == CV_MAKE_TYPE(CV_32F, 3))
	{
		CV_Assert(src.cols % 8 == 0);
		splitBGRLineInterleaveAVX_32f(src, dest);
	}
	else if (src.type() == CV_MAKE_TYPE(CV_64F, 3))
	{
		CV_Assert(src.cols % 4 == 0);
		splitBGRLineInterleaveAVX_64f(src, dest);
	}
	else
	{
		CV_Assert(src.cols % 8 == 0);
		splitBGRLineInterleaveAVX_32fcast(src, dest);
	}
}


void splitBGRUnitInterleaveAVX_8u(const Mat& src, Mat& dest)
{
	// for 32bit unit interleave

	const int sstep = src.cols * 3;
	const int dstep = dest.cols;

	const uchar* sp = src.ptr<uchar>(0);
	uchar* dp = dest.ptr<uchar>(0);//32bit unit interleave

	//BGR BGR BGR BGR BGR B x2
	//GR BGR BGR BGR BGR BG x2
	//R BGR BGR BGR BGR BGR x2
	//BBBBBBGGGGGRRRRR shuffle
	static const __m256i mask1 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
	//GGGGGBBBBBBRRRRR shuffle
	static const __m256i smask1 = _mm256_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
	static const __m256i ssmask1 = _mm256_setr_epi8(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

	//GGGGGGBBBBBRRRRR shuffle
	static const __m256i mask2 = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	static const __m256i ssmask2 = _mm256_setr_epi8(0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10);

	//const __m256i mask3 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);

	static const __m256i bmask1 = _mm256_setr_epi8
	(255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	static const __m256i bmask2 = _mm256_setr_epi8
	(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0);

	static const __m256i bmask3 = _mm256_setr_epi8
	(255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	static const __m256i bmask4 = _mm256_setr_epi8
	(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0);

	__m256i a, b, c;

	for (int j = 0; j < src.rows; j++)
	{
		int i = 0;
		for (; i < src.cols; i += 32)
		{
			a = _mm256_shuffle_epi8(_mm256_loadu2_m128i((__m128i*)(sp + 3 * i + 48), (__m128i*)(sp + 3 * i + 0)), mask1);
			b = _mm256_shuffle_epi8(_mm256_loadu2_m128i((__m128i*)(sp + 3 * i + 64), (__m128i*)(sp + 3 * i + 16)), mask2);
			c = _mm256_shuffle_epi8(_mm256_loadu2_m128i((__m128i*)(sp + 3 * i + 80), (__m128i*)(sp + 3 * i + 32)), mask2);
			_mm256_stream_si256((__m256i*)(dp + 3 * i + 0), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask1), bmask2));

			a = _mm256_shuffle_epi8(a, smask1);
			b = _mm256_shuffle_epi8(b, smask1);
			c = _mm256_shuffle_epi8(c, ssmask1);
			_mm256_stream_si256((__m256i*)(dp + 3 * i + 32), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask3), bmask2));

			a = _mm256_shuffle_epi8(a, ssmask1);
			b = _mm256_shuffle_epi8(b, ssmask2);
			c = _mm256_shuffle_epi8(c, ssmask1);
			_mm256_stream_si256((__m256i*)(dp + 3 * i + 64), _mm256_blendv_epi8(c, _mm256_blendv_epi8(b, a, bmask3), bmask4));
		}
		sp += sstep;
		dp += dstep;
	}
}

void splitBGRUnitInterleaveAVX_32f(const Mat& src, Mat& dest)
{
	/*
	const int size = src.size().area();
	const int dstep = src.cols * 3;
	const int sstep = src.cols * 3;

	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(1);
	float* R = dest.ptr<float>(2);

	for (int j = 0; j < src.rows; j++)
	{
	int i = 0;
	for (; i < src.cols; i += 8)
	{
	__m256 aa = _mm256_load_ps((s + 3 * i));
	__m256 bb = _mm256_load_ps((s + 3 * i + 8));
	__m256 cc = _mm256_load_ps((s + 3 * i + 16));
	__m256 a = _mm256_permute2f128_ps(aa, bb, 0x30);
	__m256 b = _mm256_permute2f128_ps(aa, cc, 0x21);
	__m256 c = _mm256_permute2f128_ps(bb, cc, 0x30);

	aa = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
	aa = _mm256_blend_ps(aa, b, 0x44);
	cc = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
	aa = _mm256_blend_ps(aa, cc, 0x88);
	_mm256_stream_ps((B + i), aa);

	aa = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
	bb = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
	bb = _mm256_blend_ps(bb, aa, 0x11);
	cc = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
	bb = _mm256_blend_ps(bb, cc, 0x88);
	_mm256_stream_ps((G + i), bb);

	aa = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
	bb = _mm256_blend_ps(aa, b, 0x22);
	cc = _mm256_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
	cc = _mm256_blend_ps(bb, cc, 0xcc);
	_mm256_stream_ps((R + i), cc);

	}
	R += dstep;
	G += dstep;
	B += dstep;
	s += sstep;
	}
	*/
}

void splitBGRUnitInterleaveAVX_32fcast(const Mat& src, Mat& dest)
{
	Mat a, b;
	src.convertTo(a, CV_32F);
	splitBGRLineInterleaveAVX_32f(a, b);
	b.convertTo(dest, src.type());
}

void splitBGRUnitInterleaveAVX(cv::InputArray src_, cv::OutputArray dest_)
{
	dest_.create(Size(src_.size().width * 3, src_.size().height), src_.depth());
	Mat src = src_.getMat();
	Mat dest = dest_.getMat();
	if (src.type() == CV_MAKE_TYPE(CV_8U, 3))
	{
		CV_Assert(src.cols % 32 == 0);
		splitBGRUnitInterleaveAVX_8u(src, dest);
	}
	else if (src.type() == CV_MAKE_TYPE(CV_32F, 3))
	{
		//CV_Assert(src.cols % 8 == 0);
		//splitBGRUnitInterleaveAVX_32f(src, dest);
	}
	else
	{
		//CV_Assert(src.cols % 8 == 0);
		//splitBGRUnitInterleaveAVX_32fcast(src, dest);
	}
}


void splitBGRLineInterleaveSSE_8u(const Mat& src, Mat& dest)
{
	const int size = src.size().area();
	const int dstep = src.cols * 3;
	const int sstep = src.cols * 3;

	const uchar* s = src.ptr<uchar>(0);
	uchar* B = dest.ptr<uchar>(0);//line by line interleave
	uchar* G = dest.ptr<uchar>(1);
	uchar* R = dest.ptr<uchar>(2);

	//BGR BGR BGR BGR BGR B
	//GR BGR BGR BGR BGR BG
	//R BGR BGR BGR BGR BGR
	//BBBBBBGGGGGRRRRR shuffle
	static const __m128i mask1 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
	//GGGGGBBBBBBRRRRR shuffle
	static const __m128i smask1 = _mm_setr_epi8(6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15);
	static const __m128i ssmask1 = _mm_setr_epi8(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

	//GGGGGGBBBBBRRRRR shuffle
	static const __m128i mask2 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
	static const __m128i ssmask2 = _mm_setr_epi8(0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9, 10);

	static const __m128i bmask1 = _mm_setr_epi8
	(255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	static const __m128i bmask2 = _mm_setr_epi8
	(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0);
	static const __m128i bmask3 = _mm_setr_epi8
	(255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	static const __m128i bmask4 = _mm_setr_epi8
	(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0);

	__m128i a, b, c;
	for (int j = 0; j < src.rows; j++)
	{
		int i = 0;
		for (; i < src.cols; i += 16)
		{
			a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i)), mask1);
			b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i + 16)), mask2);
			c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 3 * i + 32)), mask2);
			_mm_stream_si128((__m128i*)(B + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask1), bmask2));

			a = _mm_shuffle_epi8(a, smask1);
			b = _mm_shuffle_epi8(b, smask1);
			c = _mm_shuffle_epi8(c, ssmask1);
			_mm_stream_si128((__m128i*)(G + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask3), bmask2));

			a = _mm_shuffle_epi8(a, ssmask1);
			c = _mm_shuffle_epi8(c, ssmask1);
			b = _mm_shuffle_epi8(b, ssmask2);

			_mm_stream_si128((__m128i*)(R + i), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask3), bmask4));
		}
		R += dstep;
		G += dstep;
		B += dstep;
		s += sstep;
	}
}

void splitBGRLineInterleaveSSE_32f(const Mat& src, Mat& dest)
{
	const int size = src.size().area();
	const int dstep = src.cols * 3;
	const int sstep = src.cols * 3;

	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(1);
	float* R = dest.ptr<float>(2);

	for (int j = 0; j < src.rows; j++)
	{
		int i = 0;
		for (; i < src.cols; i += 4)
		{
			const __m128 a = _mm_load_ps((s + 3 * i));
			const __m128 b = _mm_load_ps((s + 3 * i + 4));
			const __m128 c = _mm_load_ps((s + 3 * i + 8));

			__m128 aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
			aa = _mm_blend_ps(aa, b, 4);
			__m128 cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
			aa = _mm_blend_ps(aa, cc, 8);
			_mm_stream_ps((B + i), aa);

			aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
			__m128 bb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
			bb = _mm_blend_ps(bb, aa, 1);
			cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
			bb = _mm_blend_ps(bb, cc, 8);
			_mm_stream_ps((G + i), bb);

			aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
			bb = _mm_blend_ps(aa, b, 2);
			cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
			cc = _mm_blend_ps(bb, cc, 12);
			_mm_stream_ps((R + i), cc);

		}
		R += dstep;
		G += dstep;
		B += dstep;
		s += sstep;
	}
}

void splitBGRLineInterleaveSSE_64f(const Mat& src, Mat& dest)
{
	const int size = src.size().area();
	const int dstep = src.cols * 3;
	const int sstep = src.cols * 3;

	const double* s = src.ptr<double>(0);
	double* B = dest.ptr<double>(0);//line by line interleave
	double* G = dest.ptr<double>(1);
	double* R = dest.ptr<double>(2);

	for (int j = 0; j < src.rows; j++)
	{
		int i = 0;
		for (; i < src.cols; i += 2)
		{
			const __m128d a = _mm_load_pd((s + 3 * i));
			const __m128d b = _mm_load_pd((s + 3 * i + 2));
			const __m128d c = _mm_load_pd((s + 3 * i + 4));

			__m128d aa = _mm_blend_pd(a, b, 0b10);
			_mm_stream_pd((B + i), aa);

			aa = _mm_shuffle_pd(a, a, 0b01);
			const __m128d bb = _mm_shuffle_pd(c, c, 0b01);
			aa = _mm_blend_pd(aa, bb, 0b10);
			_mm_stream_pd((G + i), aa);

			aa = _mm_blend_pd(b, c, 0b10);
			_mm_stream_pd((R + i), aa);

		}
		R += dstep;
		G += dstep;
		B += dstep;
		s += sstep;
	}
}

void splitBGRLineInterleaveSSE_32fcast(const Mat& src, Mat& dest)
{
	Mat a, b;
	src.convertTo(a, CV_32F);
	splitBGRLineInterleaveSSE_32f(a, b);
	b.convertTo(dest, src.type());
}

void splitBGRLineInterleaveSSE(cv::InputArray src_, cv::OutputArray dest_)
{
	dest_.create(Size(src_.size().width, src_.size().height * 3), src_.depth());
	Mat src = src_.getMat();
	Mat dest = dest_.getMat();
	if (src.type() == CV_MAKE_TYPE(CV_8U, 3))
	{
		CV_Assert(src.cols % 16 == 0);
		splitBGRLineInterleaveSSE_8u(src, dest);
	}
	else if (src.type() == CV_MAKE_TYPE(CV_32F, 3))
	{
		CV_Assert(src.cols % 4 == 0);
		splitBGRLineInterleaveSSE_32f(src, dest);
	}
	else if (src.type() == CV_MAKE_TYPE(CV_64F, 3))
	{
		CV_Assert(src.cols % 2 == 0);
		splitBGRLineInterleaveSSE_64f(src, dest);
	}
	else
	{
		CV_Assert(src.cols % 4 == 0);
		splitBGRLineInterleaveSSE_32fcast(src, dest);
	}
}

void cvtColorBGR2PLANE_8u(const Mat& src, Mat& dest)
{
	dest.create(Size(src.cols, src.rows * 3), CV_8U);

	const int size = src.size().area();
	const int ssesize = 3 * size - ((48 - (3 * size) % 48) % 48);
	const int ssecount = ssesize / 48;
	const uchar* s = src.ptr<uchar>(0);
	uchar* B = dest.ptr<uchar>(0);//line by line interleave
	uchar* G = dest.ptr<uchar>(src.rows);
	uchar* R = dest.ptr<uchar>(2 * src.rows);

	//BGR BGR BGR BGR BGR B	-> GGGGG RRRRR BBBBBB
	//GR BGR BGR BGR BGR BG -> GGGGGG RRRRR BBBBB
	//R BGR BGR BGR BGR BGR -> BBBBB GGGGG RRRRR

	static const __m128i mask0 = _mm_setr_epi8(1, 4, 7, 10, 13, 2, 5, 8, 11, 14, 0, 3, 6, 9, 12, 15);
	static const __m128i mask1 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);

	__m128i a, b, c, d, e;

	for (int i = 0; i < ssecount; i++)
	{
		a = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 0)), mask0);
		b = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 16)), mask1);
		c = _mm_shuffle_epi8(_mm_load_si128((__m128i*)(s + 32)), mask0);

		d = _mm_alignr_epi8(c, b, 11);
		e = _mm_alignr_epi8(d, a, 10);
		_mm_storeu_si128((__m128i*)(B), e);

		d = _mm_alignr_epi8(_mm_srli_si128(c, 5), _mm_slli_si128(b, 10), 10);
		e = _mm_alignr_epi8(d, _mm_slli_si128(a, 11), 11);
		_mm_storeu_si128((__m128i*)(G), e);

		d = _mm_alignr_epi8(_mm_srli_si128(c, 10), _mm_slli_si128(b, 5), 11);
		e = _mm_alignr_epi8(d, _mm_slli_si128(a, 6), 11);
		_mm_storeu_si128((__m128i*)(R), e);

		s += 48;
		R += 16;
		G += 16;
		B += 16;
	}
	for (int i = ssesize; i < 3 * size; i += 3)
	{
		B[0] = s[0];
		G[0] = s[1];
		R[0] = s[2];
		s += 3, R++, G++, B++;
	}
}

void cvtColorBGR2PLANE_32f(const Mat& src, Mat& dest)
{
	const int size = src.size().area();
	const int ssesize = 3 * size - ((12 - (3 * size) % 12) % 12);
	const int ssecount = ssesize / 12;
	const float* s = src.ptr<float>(0);
	float* B = dest.ptr<float>(0);//line by line interleave
	float* G = dest.ptr<float>(src.rows);
	float* R = dest.ptr<float>(2 * src.rows);

	for (int i = 0; i < ssecount; i++)
	{
		const __m128 a = _mm_load_ps(s);
		const __m128 b = _mm_load_ps(s + 4);
		const __m128 c = _mm_load_ps(s + 8);

		__m128 aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 2, 3, 0));
		aa = _mm_blend_ps(aa, b, 4);
		__m128 cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(1, 3, 2, 0));
		aa = _mm_blend_ps(aa, cc, 8);
		_mm_storeu_ps((B), aa);

		aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1));
		__m128 bb = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
		bb = _mm_blend_ps(bb, aa, 1);
		cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(2, 3, 1, 0));
		bb = _mm_blend_ps(bb, cc, 8);
		_mm_storeu_ps((G), bb);

		aa = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
		bb = _mm_blend_ps(aa, b, 2);
		cc = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
		cc = _mm_blend_ps(bb, cc, 12);
		_mm_storeu_ps((R), cc);

		s += 12;
		R += 4;
		G += 4;
		B += 4;
	}
	for (int i = ssesize; i < 3 * size; i += 3)
	{
		B[0] = s[0];
		G[0] = s[1];
		R[0] = s[2];
		s += 3, R++, G++, B++;
	}
}


template <class T>
void cvtColorBGR2PLANE_(const Mat& src, Mat& dest, int depth)
{
	vector<Mat> v(3);
	split(src, v);
	dest.create(Size(src.cols, src.rows * 3), depth);

	memcpy(dest.data, v[0].data, src.size().area() * sizeof(T));
	memcpy(dest.data + src.size().area() * sizeof(T), v[1].data, src.size().area() * sizeof(T));
	memcpy(dest.data + 2 * src.size().area() * sizeof(T), v[2].data, src.size().area() * sizeof(T));
}

void cvtColorBGR2PLANE(cv::InputArray src_, cv::OutputArray dest_)
{
	CV_Assert(src_.channels() == 3);

	Mat src = src_.getMat();
	dest_.create(Size(src.cols, src.rows * 3), src.depth());
	Mat dest = dest_.getMat();

	if (src.depth() == CV_8U)
	{
		//cvtColorBGR2PLANE_<uchar>(src, dest, CV_8U);
		cvtColorBGR2PLANE_8u(src, dest);
	}
	else if (src.depth() == CV_16U)
	{
		cvtColorBGR2PLANE_<ushort>(src, dest, CV_16U);
	}
	if (src.depth() == CV_16S)
	{
		cvtColorBGR2PLANE_<short>(src, dest, CV_16S);
	}
	if (src.depth() == CV_32S)
	{
		cvtColorBGR2PLANE_<int>(src, dest, CV_32S);
	}
	if (src.depth() == CV_32F)
	{
		//cvtColorBGR2PLANE_<float>(src, dest, CV_32F);
		cvtColorBGR2PLANE_32f(src, dest);
	}
	if (src.depth() == CV_64F)
	{
		cvtColorBGR2PLANE_<double>(src, dest, CV_64F);
	}
}


template <class T>
void cvtColorPLANE2BGR_(const Mat& src, Mat& dest, int depth)
{
	const int width = src.cols;
	const int height = src.rows / 3;
	T* b = (T*)src.ptr<T>(0);
	T* g = (T*)src.ptr<T>(height);
	T* r = (T*)src.ptr<T>(2 * height);

	Mat B(height, width, src.type(), b);
	Mat G(height, width, src.type(), g);
	Mat R(height, width, src.type(), r);
	vector<Mat> v(3);
	v[0] = B;
	v[1] = G;
	v[2] = R;
	merge(v, dest);
}

void cvtColorPLANE2BGR_8u_align(const Mat& src, Mat& dest)
{
	const int width = src.cols;
	const int height = src.rows / 3;

	if (dest.empty()) dest.create(Size(width, height), CV_8UC3);
	else if (width != dest.cols || height != dest.rows) dest.create(Size(width, height), CV_8UC3);
	else if (dest.type() != CV_8UC3) dest.create(Size(width, height), CV_8UC3);

	uchar* B = (uchar*)src.ptr<uchar>(0);
	uchar* G = (uchar*)src.ptr<uchar>(height);
	uchar* R = (uchar*)src.ptr<uchar>(2 * height);

	uchar* D = (uchar*)dest.ptr<uchar>(0);

	const int ssecount = width * height * 3 / 48;

	static const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	static const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	static const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	static const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
	static const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

	for (int i = ssecount; i--;)
	{
		__m128i a = _mm_load_si128((const __m128i*)B);
		__m128i b = _mm_load_si128((const __m128i*)G);
		__m128i c = _mm_load_si128((const __m128i*)R);

		a = _mm_shuffle_epi8(a, mask1);
		b = _mm_shuffle_epi8(b, mask2);
		c = _mm_shuffle_epi8(c, mask3);
		_mm_stream_si128((__m128i*)(D), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
		_mm_stream_si128((__m128i*)(D + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
		_mm_stream_si128((__m128i*)(D + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));

		D += 48;
		B += 16;
		G += 16;
		R += 16;
	}
}

void cvtColorPLANE2BGR_8u(const Mat& src, Mat& dest)
{
	const int width = src.cols;
	const int height = src.rows / 3;

	if (dest.empty()) dest.create(Size(width, height), CV_8UC3);
	else if (width != dest.cols || height != dest.rows) dest.create(Size(width, height), CV_8UC3);
	else if (dest.type() != CV_8UC3) dest.create(Size(width, height), CV_8UC3);

	uchar* B = (uchar*)src.ptr<uchar>(0);
	uchar* G = (uchar*)src.ptr<uchar>(height);
	uchar* R = (uchar*)src.ptr<uchar>(2 * height);

	uchar* D = (uchar*)dest.ptr<uchar>(0);

	const int ssecount = width * height * 3 / 48;
	const int rem = width * height * 3 - ssecount * 48;

	const __m128i mask1 = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
	const __m128i mask2 = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
	const __m128i mask3 = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
	const __m128i bmask1 = _mm_setr_epi8(0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0);
	const __m128i bmask2 = _mm_setr_epi8(255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255);

	for (int i = ssecount; i--;)
	{
		__m128i a = _mm_loadu_si128((const __m128i*)B);
		__m128i b = _mm_loadu_si128((const __m128i*)G);
		__m128i c = _mm_loadu_si128((const __m128i*)R);

		a = _mm_shuffle_epi8(a, mask1);
		b = _mm_shuffle_epi8(b, mask2);
		c = _mm_shuffle_epi8(c, mask3);

		_mm_storeu_si128((__m128i*)(D), _mm_blendv_epi8(c, _mm_blendv_epi8(a, b, bmask1), bmask2));
		_mm_storeu_si128((__m128i*)(D + 16), _mm_blendv_epi8(b, _mm_blendv_epi8(a, c, bmask2), bmask1));
		_mm_storeu_si128((__m128i*)(D + 32), _mm_blendv_epi8(c, _mm_blendv_epi8(b, a, bmask2), bmask1));

		D += 48;
		B += 16;
		G += 16;
		R += 16;
	}
	for (int i = rem; i--;)
	{
		D[0] = *B;
		D[1] = *G;
		D[2] = *R;
		D += 3;
		B++, G++, R++;
	}
}

void cvtColorPLANE2BGR(cv::InputArray src_, cv::OutputArray dest_)
{
	CV_Assert(src_.channels() == 1);
	Mat src = src_.getMat();
	if (dest_.empty())dest_.create(Size(src.cols, src.rows), CV_MAKETYPE(src.depth(), 3));
	Mat dest = dest_.getMat();

	if (src.depth() == CV_8U)
	{
		//cvtColorPLANE2BGR_<uchar>(src, dest, CV_8U);	
		if (src.cols % 16 == 0)
			cvtColorPLANE2BGR_8u_align(src, dest);
		else
			cvtColorPLANE2BGR_8u(src, dest);
	}
	else if (src.depth() == CV_16U)
	{
		cvtColorPLANE2BGR_<ushort>(src, dest, CV_16U);
	}
	if (src.depth() == CV_16S)
	{
		cvtColorPLANE2BGR_<short>(src, dest, CV_16S);
	}
	if (src.depth() == CV_32S)
	{
		cvtColorPLANE2BGR_<int>(src, dest, CV_32S);
	}
	if (src.depth() == CV_32F)
	{
		cvtColorPLANE2BGR_<float>(src, dest, CV_32F);
	}
	if (src.depth() == CV_64F)
	{
		cvtColorPLANE2BGR_<double>(src, dest, CV_64F);
	}
}

