#include "color.h"
#include "filter.h"

using namespace std;
using namespace cv;


class NonlocalMeansFilterInvorker_64f : public cv::ParallelLoopBody
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
	NonlocalMeansFilterInvorker_64f(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const double h, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
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

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				double* d = dest->ptr<double>(j);
				const double* sptr_ = src->ptr<double>(sr_y + tr_y + j) + sr_x + tr_x;
				for (int i = 0; i < dest->cols; i++)
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
				for (int i = 0; i < dest->cols; i++)
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
	}
};

class NonlocalMeansFilterInvorker_32f : public cv::ParallelLoopBody
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
	NonlocalMeansFilterInvorker_32f(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float h, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
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

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				float* d = dest->ptr<float>(j);
				const float* sptr_ = src->ptr<float>(sr_y + tr_y + j) + sr_x + tr_x;
				for (int i = 0; i < dest->cols; i++)
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
				for (int i = 0; i < dest->cols; i++)
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
	}
};

class NonlocalMeansFilterInvorker_8u : public cv::ParallelLoopBody
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
	NonlocalMeansFilterInvorker_8u(const Mat& src_, Mat& dest_, const int templateWindowSizeX_, const int templateWindowSizeY_, const int searchWindowSizeX_, const int searchWindowSizeY_, const float h, const int template_maxk, const int* template_ofs, const int search_maxk, const int* search_ofs)
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

		if (dest->channels() == 3)
		{
			const int colorstep = src->size().area() / 3;
			const int colorstep2 = colorstep * 2;
			for (int j = range.start; j < range.end; j++)
			{
				uchar* d = dest->ptr<uchar>(j);
				const uchar* sptr_ = src->ptr<uchar>(sr_y + tr_y + j) + sr_x + tr_x;
				for (int i = 0; i < dest->cols; i++)
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
				for (int i = 0; i < dest->cols; i++)
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
	}
};


namespace nlmf
{
	void nonLocalMeansFilter_Scalar(const Mat& src, Mat& dest, const Size templateWindowSize, const Size searchWindowSize, const double h, const int borderType, const bool isParallel)
	{
		dest = Mat::zeros(src.size(), src.type());

		const int cn = src.channels();
		const int templateH = templateWindowSize.width >> 1;
		const int templateV = templateWindowSize.height >> 1;
		const int searchH = searchWindowSize.width >> 1;
		const int searchV = searchWindowSize.height >> 1;
		const int bbx = templateH + searchH;
		const int bby = templateV + searchV;

		//create large size image for bounding box;
		Mat im;
		if (cn == 1)
		{
			copyMakeBorder(src, im, bby, bby, bbx, bbx, borderType);
		}
		else if (cn == 3)
		{
			Mat temp;
			copyMakeBorder(src, temp, bby, bby, bbx, bbx, borderType);
			cvtColorBGR2PLANE(temp, im);
		}

		vector<int> template_offset(templateWindowSize.area());
		int* template_ofs = &template_offset[0];
		vector<int> search_offset(searchWindowSize.area());
		int* search_ofs = &search_offset[0];
		int template_maxk = 0;
		setSpaceKernel(template_ofs, template_maxk, templateH, templateV, im.cols, true);
		int search_maxk = 0;
		setSpaceKernel(search_ofs, search_maxk, searchH, searchV, im.cols, true);

		if (src.depth() == CV_8U)
		{
			const NonlocalMeansFilterInvorker_8u body(im, dest, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, h, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dest.rows), body, isParallel ? -1 : 1);
		}
		else if (src.depth() == CV_32F)
		{
			const NonlocalMeansFilterInvorker_32f body(im, dest, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, h, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dest.rows), body, isParallel ? -1 : 1);
		}
		else if (src.depth() == CV_64F)
		{
			const NonlocalMeansFilterInvorker_64f body(im, dest, templateWindowSize.width, templateWindowSize.height, searchWindowSize.width, searchWindowSize.height, h, template_maxk, template_ofs, search_maxk, search_ofs);
			cv::parallel_for_(Range(0, dest.rows), body, isParallel ? -1 : 1);
		}
	}
}