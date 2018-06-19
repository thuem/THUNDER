/***********************************************************************
 * FileName: Weilume.cuh
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef WEILUME_CUH
#define WEILUME_CUH

#include "Config.h"
#include "huabin.h"

#include "Device.cuh"
#include "TabFunction.cuh"
#include "Interpolation.cuh"

namespace cuthunder {

#define CU_WEILUME_INDEX_FT(i, j, k) \
    (k) * _nRow * (_nCol / 2 + 1) + (j) * (_nCol / 2 + 1) + (i)

class Weilume
{
    public:

        HD_CALLABLE Weilume() {}

        HD_CALLABLE ~Weilume() {}

        HD_CALLABLE int nCol() const { return (_nCol / 2 + 1); }
        HD_CALLABLE int nRow() const { return _nRow; }
        HD_CALLABLE int nSlc() const { return _nSlc; }

        HD_CALLABLE RFLOAT get(const int iCol,
                               const int iRow,
                               const int iSlc) const;

        HD_CALLABLE RFLOAT getByInterpolation(const RFLOAT iCol,
                                              const RFLOAT iRow,
                                              const RFLOAT iSlc) const;

        HD_CALLABLE void set(const RFLOAT value,
                             const int iCol,
                             const int iRow,
                             const int iSlc);

        D_CALLABLE void add(const RFLOAT value,
                            const int iCol,
                            const int iRow,
                            const int iSlc);

        D_CALLABLE void add(const double value,
                            const double iCol,
                            const double iRow,
                            const double iSlc,
                            const TabFunction& kernel,
                            const double a,
                            bool debug = false);
        
        //new add
        HD_CALLABLE Weilume(const Weilume& vol);

        HD_CALLABLE void init(RFLOAT* data,   
                              const int nCol,
                              const int nRow,
                              const int nSlc);
        HD_CALLABLE void init(cudaArray* data,   
                              const int nCol,
                              const int nRow,
                              const int nSlc);

        HD_CALLABLE void init(const int nCol,
                              const int nRow,
                              const int nSlc);

        HD_CALLABLE void init(RFLOAT* data);

        HD_CALLABLE void hostPtr(RFLOAT *data) { _hostDataFT = data; }
        HD_CALLABLE RFLOAT* hostPtr() const { return _hostDataFT; }

        HD_CALLABLE void devPtr(RFLOAT *data) { _devDataFT = data; }
        HD_CALLABLE RFLOAT* devPtr() const { return _devDataFT; }
        
        HD_CALLABLE bool coordinatesInBoundaryFT(const int iCol,
                                                 const int iRow,
                                                 const int iSlc) const;

        D_CALLABLE bool conjHalf(RFLOAT& iCol, 
                                 RFLOAT& iRow, 
                                 RFLOAT& iSlc) const;
        
        D_CALLABLE void getFromIndex(const int index, 
                                     int& i, 
                                     int& j, 
                                     int& k) const;

        D_CALLABLE int getIndexHalf(int i, int j, int k) const;

        D_CALLABLE RFLOAT getFTHalf(const int iCol, 
                                    const int iRow, 
                                    const int iSlc) const;

        D_CALLABLE RFLOAT getByInterpolationFT(RFLOAT iCol, 
                                               RFLOAT iRow, 
                                               RFLOAT iSlc, 
                                               const int interp) const;


    private:

        HD_CALLABLE bool coordinatesInBoundary(const int iCol,
                                               const int iRow,
                                               const int iSlc) const;

        HD_CALLABLE int getIndex(int i, int j, int k) const;

    private:

        int _size = 0;

        int _nCol = 0;
        int _nRow = 0;
        int _nSlc = 0;

        RFLOAT *_devDataFT = NULL;
        RFLOAT *_hostDataFT = NULL;
        cudaArray *_cudaarray = NULL;
    
};

} // end namespace cuthunder

#endif
