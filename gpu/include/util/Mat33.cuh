/***********************************************************************
 * FileName: Mat33.cuh
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef MAT33_CUH
#define MAT33_CUH

#include "Device.cuh"
#include "Vec3.cuh"

namespace cuthunder {

class Mat33
{
    public:

        HD_CALLABLE Mat33() {}

        HD_CALLABLE Mat33(double *dev_data, int index);

        HD_CALLABLE Mat33(const Mat33& mat);
        
        HD_CALLABLE ~Mat33() {}

        HD_CALLABLE void init(double *dev_data, int index);

        HD_CALLABLE double getElement(const int row,
                                      const int col) const;

        HD_CALLABLE double& operator[](int index);
        
        HD_CALLABLE const double& operator[](int index) const;
        
        HD_CALLABLE __forceinline__ Mat33& operator+=(const Mat33& mat){
            for (int i = 0; i < 9; i++)
                _data[i] += mat[i];

            return *this;
        }
        
        HD_CALLABLE __forceinline__ Mat33& operator*(const double scale){
            for (int i = 0; i < 9; i++)
                _data[i] *= scale;

            return *this;
        }
        
        HD_CALLABLE __forceinline__ Mat33& operator*=(const double scale){
            for (int i = 0; i < 9; i++)
                _data[i] *= scale;

            return *this;
        }
        
        HD_CALLABLE __forceinline__ Mat33& operator*=(const Mat33& mat){
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
        
        HD_CALLABLE __forceinline__ Vec3 operator*(const Vec3& vec) const{
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

    private:

        double* _data;
    
};

} // end namespace cuthunder

#endif
