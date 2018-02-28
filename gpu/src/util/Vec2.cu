/***********************************************************************
 * FileName: Vec2.cu
 * Author  : Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Vec2.cuh"

namespace cuthunder {

HD_CALLABLE Vec2::Vec2(double v0, double v1)
{
	_data[0] = v0;
	_data[1] = v1;
}

HD_CALLABLE Vec2::Vec2(const Vec2& that)
{
	for (int i = 0; i < 2; ++i)
		_data[i] = that._data[i];
}

HD_CALLABLE double Vec2::getElement(const int i) const
{
	return _data[i];
}

HD_CALLABLE void Vec2::setElement(const int i, const double value)
{
	_data[i] = value;
}

HD_CALLABLE void Vec2::reset(double v0, double v1)
{
	_data[0] = v0;
	_data[1] = v1;
}

HD_CALLABLE double& Vec2::operator()(int index)
{
	return _data[index];
}

HD_CALLABLE const double& Vec2::operator()(int index) const
{
	return _data[index];
}

HD_CALLABLE Vec2 Vec2::operator*(const double scale) const
{
	return Vec2(_data[0] * scale,
				_data[1] * scale);
}

HD_CALLABLE Vec2& Vec2::operator*=(const double scale)
{
	_data[0] *= scale;
	_data[1] *= scale;

	return *this;
}

} // end namespace cuthunder
