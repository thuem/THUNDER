/***********************************************************************
 * FileName: Volume.cuh
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef VOLUME_CUH
#define VOLUME_CUH

#include "Device.cuh"
#include "Config.cuh"
#include "Complex.cuh"
#include "TabFunction.cuh"
#include "Interpolation.cuh"

namespace cuthunder {

#define CU_VOLUME_INDEX_FT(i, j, k) \
    (k) * _nRow * (_nCol / 2 + 1) + (j) * (_nCol / 2 + 1) + (i)

#define CU_VOLUME_SUB_SPHERE_FT(a) \
    for (int k = CU_MAX(-_nSlc / 2, floor(iSlc - a)); \
             k <= CU_MIN(_nSlc / 2 - 1, ceil(iSlc + a)); \
             k++) \
        for (int j = CU_MAX(-_nRow / 2, floor(iRow - a)); \
                 j <= CU_MIN(_nRow / 2 - 1, ceil(iRow + a)); \
                 j++) \
            for (int i = CU_MAX(-_nCol / 2, floor(iCol - a)); \
                     i <= CU_MIN(_nCol / 2, ceil(iCol + a)); \
                     i++)

class Volume
{
    public:

        HD_CALLABLE Volume() {}

        HD_CALLABLE Volume(const Volume& vol);

        HD_CALLABLE ~Volume() {}

        HD_CALLABLE void init(Complex* data,   
                              const int nCol,
                              const int nRow,
                              const int nSlc);

        HD_CALLABLE void init(const int nCol,
                              const int nRow,
                              const int nSlc);

        HD_CALLABLE void clear();

        HD_CALLABLE int nSize() const;

        HD_CALLABLE int nColFT() const;
        HD_CALLABLE int nRowFT() const;
        HD_CALLABLE int nSlcFT() const;

        HD_CALLABLE void hostPtr(Complex *data) { _hostDataFT = data; }
        HD_CALLABLE Complex* hostPtr() const { return _hostDataFT; }

        HD_CALLABLE void devPtr(Complex *data) { _devDataFT = data; }
        HD_CALLABLE Complex* devPtr() const { return _devDataFT; }

        HD_CALLABLE bool coordinatesInBoundaryFT(const int iCol,
                                                 const int iRow,
                                                 const int iSlc) const;


        HD_CALLABLE Complex getFT(const int iCol,
                                  const int iRow,
                                  const int iSlc) const;

        HD_CALLABLE Complex getByInterpolationFT(const RFLOAT iCol,
                                                 const RFLOAT iRow,
                                                 const RFLOAT iSlc) const;

        HD_CALLABLE void setFT(Complex value,
                               const int iCol,
                               const int iRow,
                               const int iSlc);

        D_CALLABLE void addFT(Complex value,
                              const int iCol,
                              const int iRow,
                              const int iSlc);

        D_CALLABLE void addFT(const Complex value,
                              const double iCol,
                              const double iRow,
                              const double iSlc,
                              const TabFunction& kernel,
                              const double a,
                              bool debug = false);
        
        
        D_CALLABLE bool conjHalf(RFLOAT& iCol, 
                                 RFLOAT& iRow, 
                                 RFLOAT& iSlc) const;

        D_CALLABLE void getFromIndex(const int index, 
                                     int& i, 
                                     int& j, 
                                     int& k) const;

        D_CALLABLE int getIndexHalf(int i, int j, int k) const;

        D_CALLABLE Complex getFTHalf(const int iCol, 
                                     const int iRow, 
                                     const int iSlc) const;

        D_CALLABLE Complex getByInterpolationFT(RFLOAT iCol, 
                                                RFLOAT iRow, 
                                                RFLOAT iSlc, 
                                                const int interp) const;


    private:

        HD_CALLABLE int getIndex(int i, int j, int k, bool& flag) const;

    private:

        int _size = 0;

        int _nCol = 0;
        int _nRow = 0;
        int _nSlc = 0;

        /* Load from host */
        Complex *_devDataFT = NULL;

        /* Portable cuda memory */
        Complex *_hostDataFT = NULL; 
    
};

} // end namespace cuthunder

#endif
