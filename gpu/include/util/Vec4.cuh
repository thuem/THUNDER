/***********************************************************************
 * FileName: Vec4.cuh
 * Author  : Zhao WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef VEC4_CUH
#define VEC4_CUH

#include "Device.cuh"

namespace cuthunder {

class Vec4
{
    public:

        HD_CALLABLE Vec4() {}

        HD_CALLABLE Vec4(double v0, double v1, double v2, double v3);

        HD_CALLABLE ~Vec4() {}

        HD_CALLABLE Vec4(const Vec4& that);

        HD_CALLABLE double getElement(const int i) const;

        HD_CALLABLE void setElement(const int i, const double value);

        HD_CALLABLE void reset(double v0, double v1, double v2, double v3);

        HD_CALLABLE double& operator()(int index);

        HD_CALLABLE const double& operator()(int index) const;

        HD_CALLABLE Vec4 operator*(const double scale) const;

        HD_CALLABLE Vec4& operator*=(const double scale);
        
        
    private:

        double _data[4] = {0};
    
};

} // end namespace cuthunder

#endif
