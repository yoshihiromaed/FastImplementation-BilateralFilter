#include "filter.h"
#include "fmath.hpp"

using namespace cv;
using namespace std;


void setSpaceKernel(int* space_ofs, int& maxk, const int radiusH, const int radiusV, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV, radiusH);
	for (int i = -radiusV; i <= radiusV; i++)
	{
		for (int j = -radiusH; j <= radiusH; j++)
		{
			const double r = std::sqrt((float)i*i + (float)j*j);
			if (r > maxr && !isRectangle) continue;

			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}


/******************
	double
 ******************/
void setSpaceKernel(double* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle, const bool isSpaceWeight)
{
	const int maxr = std::max(radiusV, radiusH);
	for (int i = -radiusV; i <= radiusV; i++)
	{
		for (int j = -radiusH; j <= radiusH; j++)
		{
			const double r = std::sqrt((float)i*i + (float)j*j);
			if (r > maxr && !isRectangle) continue;

			double aw = r * r*gauss_space_coeff;
			space_weight[maxk] = isSpaceWeight ? (double)std::exp(r*r*gauss_space_coeff) : r * r;
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void setSpaceKernel_denormalSuppression(double* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV, radiusH);
	for (int i = -radiusV; i <= radiusV; i++)
	{
		for (int j = -radiusH; j <= radiusH; j++)
		{
			const double r = std::sqrt((float)i*i + (float)j*j);
			if (r > maxr && !isRectangle) continue;

			const double aw = r * r * gauss_space_coeff;
			space_weight[maxk] = std::exp(aw);
			space_weight[maxk] = max(space_weight[maxk], DBL_MIN);
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void setSpaceKernel_expArgClip(double* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const double gauss_space_coeff, const int imstep, const bool isRectangle, const double expClipVal, const double bias)
{
	const int maxr = std::max(radiusV, radiusH);
	for (int i = -radiusV; i <= radiusV; i++)
	{
		for (int j = -radiusH; j <= radiusH; j++)
		{
			const double r = std::sqrt((double)i*i + (double)j*j);
			if (r > maxr && !isRectangle) continue;

			const double aw = max(r * r * gauss_space_coeff, expClipVal);
			space_weight[maxk] = std::exp(aw)*bias;
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}


/******************
	float
 ******************/
void setSpaceKernel(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const float gauss_space_coeff, const int imstep, const bool isRectangle, const bool isSpaceWeight)
{
	const int maxr = std::max(radiusV, radiusH);
	for (int i = -radiusV; i <= radiusV; i++)
	{
		for (int j = -radiusH; j <= radiusH; j++)
		{
			const float r = std::sqrt((float)i*i + (float)j*j);
			if (r > maxr && !isRectangle) continue;

			float aw = r * r*gauss_space_coeff;
			space_weight[maxk] = isSpaceWeight ? (float)std::exp(r*r*gauss_space_coeff) : r * r;
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void setSpaceKernel_denormalSuppression(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const float gauss_space_coeff, const int imstep, const bool isRectangle)
{
	const int maxr = std::max(radiusV, radiusH);
	for (int i = -radiusV; i <= radiusV; i++)
	{
		for (int j = -radiusH; j <= radiusH; j++)
		{
			const float r = std::sqrt((float)i*i + (float)j*j);
			if (r > maxr && !isRectangle) continue;

			const float aw = r * r * gauss_space_coeff;
			space_weight[maxk] = std::exp(aw);
			space_weight[maxk] = max(space_weight[maxk], FLT_MIN);
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}

void setSpaceKernel_expArgClip(float* space_weight, int* space_ofs, int& maxk, const int radiusH, const int radiusV, const float gauss_space_coeff, const int imstep, const bool isRectangle, const float expClipVal, const float bias)
{
	const int maxr = std::max(radiusV, radiusH);
	for (int i = -radiusV; i <= radiusV; i++)
	{
		for (int j = -radiusH; j <= radiusH; j++)
		{
			const float r = std::sqrt((float)i*i + (float)j*j);
			if (r > maxr && !isRectangle) continue;

			const float aw = max(r * r * gauss_space_coeff, expClipVal);
			space_weight[maxk] = std::exp(aw)*bias;
			space_ofs[maxk++] = (int)(i*imstep + j);
		}
	}
}
