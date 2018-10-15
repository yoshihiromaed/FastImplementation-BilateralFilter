#include <immintrin.h>
#include <pmmintrin.h>
#include <iostream>

bool show_mxcsr(const bool showState, const bool showMask, const bool isClaer)
{
	bool ret = false;

	if (isClaer)
	{
		_MM_SET_EXCEPTION_STATE(0);
	}

	const unsigned int mxcsr = _mm_getcsr();
	//std::cout << mxcsr << std::endl;
	if (mxcsr & 0b10)
	{
		ret = true;
	}

	if (showState)
	{
		if (mxcsr & 0b1)
		{
			std::cout << "Invalid Operation happens" << std::endl;
		}
		if (mxcsr & 0b10)
		{
			std::cout << "Denormal happens" << std::endl;
		}
		if (mxcsr & 0b100)
		{
			std::cout << "Divede By Zero happens" << std::endl;
		}
		if (mxcsr & 0b1000)
		{
			std::cout << "Overflow happens" << std::endl;
		}
		if (mxcsr & 0b10000)
		{
			std::cout << "Underflow happens" << std::endl;
		}
		if (mxcsr & 0b100000)
		{
			std::cout << "Precision happens" << std::endl;
		}
	}

	if (showMask)
	{
		if (mxcsr & 0b10000000)
		{
			std::cout << "Invalid Operation Mask" << std::endl;
		}
		if (mxcsr & 0b100000000)
		{
			std::cout << "Denormal Mask" << std::endl;
		}
		if (mxcsr & 0b1000000000)
		{
			std::cout << "Divide By Zero Mask" << std::endl;
		}
		if (mxcsr & 0b10000000000)
		{
			std::cout << "Overflow Mask" << std::endl;
		}
		if (mxcsr & 0b100000000000)
		{
			std::cout << "Underflow Mask" << std::endl;
		}
		if (mxcsr & 0b1000000000000)
		{
			std::cout << "Precision Mask" << std::endl;
		}
		{
			const int round = (mxcsr & 0b110000000000000) >> 13;
			if (round == 0b00)
			{
				std::cout << "Round To Nearest" << std::endl;
			}
			else if (round == 0b01)
			{
				std::cout << "Round To Negative" << std::endl;
			}
			else if (round == 0b10)
			{
				std::cout << "Round To Positive" << std::endl;
			}
			else if (round == 0b11)
			{
				std::cout << "Round To Zero" << std::endl;
			}
		}
		if (mxcsr & 0b1000000000000000)
		{
			std::cout << "enable Flush To Zero" << std::endl;
		}
		if (mxcsr & 0b1000000)
		{
			std::cout << "enable Denormals Are Zero" << std::endl;
		}
	}

	return ret;
}