/***********************************************************************
 * FileName: Vec3.cu
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Vec3.cuh"

namespace cuthunder {

HD_CALLABLE Vec3::Vec3(double v0, double v1, double v2)
{
	_data[0] = v0;
	_data[1] = v1;
	_data[2] = v2;
}

HD_CALLABLE Vec3::Vec3(const Vec3& that)
{
	for (int i = 0; i < 3; ++i)
		_data[i] = that._data[i];
}

HD_CALLABLE double Vec3::getElement(const int i) const
{
	return _data[i];
}

HD_CALLABLE void Vec3::setElement(const int i, const double value)
{
	_data[i] = value;
}

HD_CALLABLE void Vec3::reset(double v0, double v1, double v2)
{
	_data[0] = v0;
	_data[1] = v1;
	_data[2] = v2;
}

HD_CALLABLE double& Vec3::operator()(int index)
{
	return _data[index];
}

HD_CALLABLE const double& Vec3::operator()(int index) const
{
	return _data[index];
}

HD_CALLABLE Vec3 Vec3::operator*(const double scale) const
{
	return Vec3(_data[0] * scale,
				_data[1] * scale,
				_data[2] * scale);
}

HD_CALLABLE Vec3& Vec3::operator*=(const double scale)
{
	_data[0] *= scale;
	_data[1] *= scale;
	_data[2] *= scale;

	return *this;
}

HD_CALLABLE double Vec3::squaredNorm3()
{
    return (_data[0] * _data[0] +
    	    _data[1] * _data[1] +
    	    _data[2] * _data[2]);
}

} // end namespace cuthunder
