/***********************************************************************
 * FileName: Vec4.cu
 * Author  : Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Vec4.cuh"

namespace cuthunder {

HD_CALLABLE Vec4::Vec4(double v0, double v1, double v2, double v3)
{
	_data[0] = v0;
	_data[1] = v1;
	_data[2] = v2;
	_data[3] = v3;
}

HD_CALLABLE Vec4::Vec4(const Vec4& that)
{
	for (int i = 0; i < 4; ++i)
		_data[i] = that._data[i];
}

HD_CALLABLE double Vec4::getElement(const int i) const
{
	return _data[i];
}

HD_CALLABLE void Vec4::setElement(const int i, const double value)
{
	_data[i] = value;
}

HD_CALLABLE void Vec4::reset(double v0, double v1, double v2, double v3)
{
	_data[0] = v0;
	_data[1] = v1;
	_data[2] = v2;
	_data[3] = v3;
}

HD_CALLABLE double& Vec4::operator()(int index)
{
	return _data[index];
}

HD_CALLABLE const double& Vec4::operator()(int index) const
{
	return _data[index];
}

HD_CALLABLE Vec4 Vec4::operator*(const double scale) const
{
	return Vec4(_data[0] * scale,
				_data[1] * scale,
				_data[2] * scale,
				_data[3] * scale);
}

HD_CALLABLE Vec4& Vec4::operator*=(const double scale)
{
	_data[0] *= scale;
	_data[1] *= scale;
	_data[2] *= scale;
	_data[3] *= scale;

	return *this;
}

} // end namespace cuthunder
