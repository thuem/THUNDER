/***********************************************************************
 * FileName: Vec2.cuh
 * Author  : Zhao WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef VEC2_CUH
#define VEC2_CUH

#include "Device.cuh"

namespace cuthunder {

class Vec2
{
    public:

        HD_CALLABLE Vec2() {}

        HD_CALLABLE Vec2(double v0, double v1);

        HD_CALLABLE ~Vec2() {}

        HD_CALLABLE Vec2(const Vec2& that);

        HD_CALLABLE double getElement(const int i) const;

        HD_CALLABLE void setElement(const int i, const double value);

        HD_CALLABLE void reset(double v0, double v1);

        HD_CALLABLE double& operator()(int index);

        HD_CALLABLE const double& operator()(int index) const;

        HD_CALLABLE Vec2 operator*(const double scale) const;

        HD_CALLABLE Vec2& operator*=(const double scale);
        
    private:

        double _data[2] = {0};
    
};

} // end namespace cuthunder

#endif
