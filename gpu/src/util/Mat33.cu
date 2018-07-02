/***********************************************************************
 * FileName: Mat33.cu
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Mat33.cuh"

namespace cuthunder {

HD_CALLABLE Mat33::Mat33(double *dev_data, int index)
{
    init(dev_data, index);
}

HD_CALLABLE Mat33::Mat33(const Mat33& mat)
{
    for (int i = 0; i < 9; i++)
        _data[i] = mat._data[i];
}

HD_CALLABLE void Mat33::init(double *dev_data, int index)
{
    _data = dev_data + index * 9;
}

HD_CALLABLE double Mat33::getElement(const int row,
                                     const int col) const
{
    return _data[col + row * 3];
}

HD_CALLABLE double& Mat33::operator[](int index)
{
	return _data[index];
}

HD_CALLABLE const double& Mat33::operator[](int index) const
{
	return _data[index];
}
/*
HD_CALLABLE Mat33& Mat33::operator+=(const Mat33& mat)
{
    for (int i = 0; i < 9; i++)
        _data[i] += mat[i];

    return *this;
}

HD_CALLABLE Mat33& Mat33::operator*(const double scale) 
{
    for (int i = 0; i < 9; i++)
        _data[i] *= scale;

    return *this;
}

HD_CALLABLE Mat33& Mat33::operator*=(const double scale) 
{
    for (int i = 0; i < 9; i++)
        _data[i] *= scale;

    return *this;
}

HD_CALLABLE Mat33& Mat33::operator*=(const Mat33& mat)
{
    Mat33 res;
    double resData[9];

    res.init(resData, 0);

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                res[i + j * 3] += _data[i + k * 3] * mat[k + j * 3];
    
    for (int i = 0; i < 9; i++)
        _data[i] = res[i];

    return *this;
}

HD_CALLABLE Vec3 Mat33::operator*(const Vec3& vec) const
{
    double v0, v1, v2;

    v0 = _data[0] * vec(0)
       + _data[3] * vec(1)
       + _data[6] * vec(2);

    v1 = _data[1] * vec(0)
       + _data[4] * vec(1)
       + _data[7] * vec(2);

    v2 = _data[2] * vec(0)
       + _data[5] * vec(1)
       + _data[8] * vec(2);

    return Vec3(v0, v1, v2);
}
*/
} // end namespace cuthunder
