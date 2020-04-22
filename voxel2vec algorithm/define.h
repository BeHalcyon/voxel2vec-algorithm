#pragma once

struct DoublePoint
{
	DoublePoint(double x = 0, double y = 0, double z = 0) :x(x), y(y), z(z) {}
	double x, y, z;
	double normalize(const DoublePoint& b) const
	{
		auto& a = *this;
		return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z));
	}
};
