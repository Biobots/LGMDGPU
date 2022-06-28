#ifndef SCREEN
#define SCREEN

#include <kompute/Kompute.hpp>
#include <math.h>

class Screen
{
private:
	kp::Manager* handle;
	int width;
	int height;
	float hexRad;
public:
	int rowNum;
	int colNum;
	std::shared_ptr<kp::TensorT<uint32_t>> tensor;
	Screen() {}
	Screen(kp::Manager* mgr, int width, int height, float hexRad)
		:width(width),
		height(height),
		hexRad(hexRad)
	{
		handle = mgr;

		rowNum = (int)(floor(((float)height + hexRad) / (3 * hexRad)) * 2 + 1);
		colNum = (int)(floor((float)width / (sqrtf(3) * hexRad)) + 2);
		int rRangeMax = (rowNum - 1) / 2;
		int cRangeMax = colNum - 1;

		std::vector<uint32_t> tmpMap(width * height, 0);
		tensor = handle->tensorT<uint32_t>(tmpMap);
	}
	~Screen()
	{

	}
};

#endif // !SCREEN