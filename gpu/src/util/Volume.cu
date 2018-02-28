/***********************************************************************
 * FileName: Volume.cu
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Volume.cuh"

namespace cuthunder {

HD_CALLABLE Volume::Volume(const Volume& vol)
{
    // Copy-constructor will be called when
    // arguments of kernel are passed by object
    // value.
    //
    // This explicitly defines device side ctor.

    _nCol = vol._nCol;
    _nRow = vol._nRow;
    _nSlc = vol._nSlc;

    _size = vol._size;

    _devDataFT  = vol._devDataFT;
    _hostDataFT = vol._hostDataFT;
}

HD_CALLABLE void Volume::init(const int nCol,
                              const int nRow,
                              const int nSlc)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;

    _size = (nCol / 2 + 1) * nRow * nSlc;
}

HD_CALLABLE void Volume::init(Complex* data,
                              const int nCol,
                              const int nRow,
                              const int nSlc)
{
    devPtr(data);

    init(nCol, nRow, nSlc);
}

HD_CALLABLE void Volume::clear()
{
    init(0, 0, 0);

    _devDataFT = NULL;
    _hostDataFT = NULL;
}

HD_CALLABLE int Volume::nSize() const { return _size; }

HD_CALLABLE int Volume::nColFT() const { return (_nCol / 2 + 1); }

HD_CALLABLE int Volume::nRowFT() const { return _nRow; }

HD_CALLABLE int Volume::nSlcFT() const { return _nSlc; }

HD_CALLABLE int Volume::getIndex(int i, int j, int k, bool& flag) const
{
    if (i >= 0) flag = false;
    else { i *= -1; j *= -1; k *= -1; flag = true; }

    if (j < 0) j += _nRow;
    if (k < 0) k += _nSlc;

    return CU_VOLUME_INDEX_FT(i, j, k);
}

HD_CALLABLE Complex Volume::getFT(const int iCol,
                                  const int iRow,
                                  const int iSlc) const
{
    if(!coordinatesInBoundaryFT(iCol, iRow, iSlc))
        return Complex();
    
    bool flag;
    int index = getIndex(iCol, iRow, iSlc, flag);

    if (flag)
        return _devDataFT[index].conj();
    else
        return _devDataFT[index];
}

HD_CALLABLE Complex Volume::getByInterpolationFT(const double iCol,
                                                 const double iRow,
                                                 const double iSlc) const
{
    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    Complex result(0.0, 0.0);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                result += getFT(x0[0] + i, x0[1] + j, x0[2] + k)
                          * w[i][j][k];
    return result;
}

HD_CALLABLE void Volume::setFT(Complex value,
                               const int iCol,
                               const int iRow,
                               const int iSlc)
{
    if(!coordinatesInBoundaryFT(iCol, iRow, iSlc))
        return ;
    
    bool flag;
    int index = getIndex(iCol, iRow, iSlc, flag);
    flag ? value.conj() : value;

    _devDataFT[index] = value;
}

D_CALLABLE void Volume::addFT(Complex value,
                              const int iCol,
                              const int iRow,
                              const int iSlc)
{
    if(!coordinatesInBoundaryFT(iCol, iRow, iSlc))
        return ;
    
    bool flag;
    int index = getIndex(iCol, iRow, iSlc, flag);

    // _devDataFT[index] += (flag ? value.conj() : value);

    flag ? value.conj() : value;

    atomicAdd(_devDataFT[index].realAddr(), value.real());
    atomicAdd(_devDataFT[index].imagAddr(), value.imag());
}

D_CALLABLE void Volume::addFT(const Complex value,
                              const double iCol,
                              const double iRow,
                              const double iSlc,
                              const TabFunction& kernel,
                              const double a,
                              bool debug)
{
    for (int k = CU_MAX(-_nSlc / 2, floor(iSlc - a));
             k <= CU_MIN(_nSlc / 2 - 1, ceil(iSlc + a));
             k++) {
        for (int j = CU_MAX(-_nRow / 2, floor(iRow - a));
                 j <= CU_MIN(_nRow / 2 - 1, ceil(iRow + a));
                 j++) {
            for (int i = CU_MAX(-_nCol / 2, floor(iCol - a));
                     i <= CU_MIN(_nCol / 2, ceil(iCol + a));
                     i++)
            {
                double r = norm3d((double)(iCol - i),
                                  (double)(iRow - j),
                                  (double)(iSlc - k));

                if (r < a) {
                    addFT(value * kernel(r), i, j, k);
                }
            }
        }
    }
}

HD_CALLABLE bool Volume::coordinatesInBoundaryFT(const int iCol,
                                                 const int iRow,
                                                 const int iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
    {
        printf("*** Volume index out of boundary!\n");
        return false;
    }else
        return true;
}


D_CALLABLE bool Volume::conjHalf(double& iCol,
                                 double& iRow,
                                 double& iSlc) const
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;
    iSlc *= -1;

    return true;
}

D_CALLABLE void Volume::getFromIndex(const int index, 
                                     int& i, 
                                     int& j, 
                                     int& k) const
{
    i = index % (_nCol / 2 + 1);
    j = (index / (_nCol / 2 + 1)) % _nRow;
    k = (index / (_nCol / 2 + 1)) / _nRow;
    if(j > _nRow) j = _nRow - j;
    if(k > _nSlc) k = _nSlc - k;
}

D_CALLABLE int Volume::getIndexHalf(int i, int j, int k) const
{
    if (j < 0) j += _nRow;
    if (k < 0) k += _nSlc;

    return CU_VOLUME_INDEX_FT(i, j, k);
}

D_CALLABLE Complex Volume::getFTHalf(const int iCol,
                                     const int iRow,
                                     const int iSlc) const
{
    if(!coordinatesInBoundaryFT(iCol, iRow, iSlc))
        return Complex();
    
    int index = getIndexHalf(iCol, iRow, iSlc);
    return _devDataFT[index];
}

D_CALLABLE Complex Volume::getByInterpolationFT(double iCol,
                                                double iRow,
                                                double iSlc,
                                                const int interp) const
{
    bool conjug = conjHalf(iCol, iRow, iSlc);

    if(interp == 0)
    {
        Complex result = getFTHalf((int)rint(iCol), 
                                   (int)rint(iRow),
                                   (int)rint(iSlc));
        return conjug ? result.conj() : result;
    }

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    Complex result(0.0, 0.0);
    if((x0[1] != -1) && x0[2] != -1)
    {
        int index0 = getIndexHalf(x0[0], x0[1], x0[2]);
        for (int k = 0; k < 2; k++)
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++)
                {
                    int index = index0 + CU_VOLUME_INDEX_FT(i, j, k);
                    result += _devDataFT[index] * w[k][j][i];
                }
    }
    else
    {
        for (int k = 0; k < 2; k++)
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++)
                    result += getFTHalf(x0[0] + i, x0[1] + j, x0[2] + k)
                            * w[k][j][i];
    }
    return conjug ? result.conj() : result;
}

} // end namespace cuthunder
