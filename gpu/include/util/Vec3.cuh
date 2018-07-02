/***********************************************************************
 * FileName: Vec3.cuh
 * Author  : Kunpeng WANG, Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef VEC3_CUH
#define VEC3_CUH

#include "Device.cuh"

namespace cuthunder {

class Vec3
{
    public:

        HD_CALLABLE Vec3() {}

        HD_CALLABLE Vec3(double v0, double v1, double v2);

        HD_CALLABLE ~Vec3() {}

        HD_CALLABLE Vec3(const Vec3& that);

        HD_CALLABLE double getElement(const int i) const;

        HD_CALLABLE void setElement(const int i, const double value);

        HD_CALLABLE void reset(double v0, double v1, double v2);

        HD_CALLABLE double& operator()(int index);

        HD_CALLABLE const double& operator()(int index) const;

        HD_CALLABLE Vec3 operator*(const double scale) const;

        HD_CALLABLE Vec3& operator*=(const double scale);
        
        //new add
        HD_CALLABLE double squaredNorm3();
        
    private:

        double _data[3] = {0};
    
};

} // end namespace cuthunder

#endif
