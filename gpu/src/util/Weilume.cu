/***********************************************************************
 * FileName: Weilume.cu
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Weilume.cuh"

namespace cuthunder {

HD_CALLABLE int Weilume::getIndex(int i, int j, int k) const
{
    if (i < 0) { i *= -1; j *= -1; k *= -1; }
    if (j < 0) j += _nRow;
    if (k < 0) k += _nSlc;

    return (  k * _nRow * (_nCol / 2 + 1)
            + j * (_nCol / 2 + 1)
            + i );
}

HD_CALLABLE double Weilume::get(const int iCol,
                                const int iRow,
                                const int iSlc) const
{
    if(!coordinatesInBoundary(iCol, iRow, iSlc))
        return 0.0;

    int index = getIndex(iCol, iRow, iSlc);

    return _devDataFT[index];
}

HD_CALLABLE double Weilume::getByInterpolation(const double iCol,
                                               const double iRow,
                                               const double iSlc) const
{
    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    double result = 0.0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                result += get(x0[0] + i, x0[1] + j, x0[2] + k)
                          * w[i][j][k];
    return result;
}

HD_CALLABLE void Weilume::set(const double value,
                              const int iCol,
                              const int iRow,
                              const int iSlc)
{
    if(!coordinatesInBoundary(iCol, iRow, iSlc))
        return ;

    int index = getIndex(iCol, iRow, iSlc);

    _devDataFT[index] = value;
}

D_CALLABLE void Weilume::add(const double value,
                             const int iCol,
                             const int iRow,
                             const int iSlc)
{
    if(!coordinatesInBoundary(iCol, iRow, iSlc))
        return ;

    int index = getIndex(iCol, iRow, iSlc);

    atomicAdd(&_devDataFT[index], value);
}

D_CALLABLE void Weilume::add(const double value,
                             const double iCol,
                             const double iRow,
                             const double iSlc,
                             const TabFunction& kernel,
                             const double a,
                             bool debug)
{
    int k_L = CU_MAX(-_nSlc / 2, floor(iSlc - a));
    int k_H = CU_MIN(_nSlc / 2 - 1, ceil(iSlc + a));

    int j_L = CU_MAX(-_nRow / 2, floor(iRow - a));
    int j_H = CU_MIN(_nRow / 2 - 1, ceil(iRow + a));

    int i_L = CU_MAX(-_nCol / 2, floor(iCol - a));
    int i_H = CU_MIN(_nCol / 2 - 1, ceil(iCol + a));

    for (int k = k_L; k <= k_H; k++)
        for (int j = j_L; j <= j_H; j++)
            for (int i = i_L; i <= i_H; i++)
            {
                double r = norm3d((double)(iCol - i),
                                  (double)(iRow - j),
                                  (double)(iSlc - k));

                if (r < a)
                {
                    add(value * kernel(r), i, j, k);

                    /* Debug */
                    // if (debug)
                    // {
                    //     double inc = value * kernel(r);
                        
                    //     printf("r=%9.4f, kernel(r)=%9.7f, Volume(%3d,%3d,%3d) <- %9.4f\n",
                    //             r, kernel(r), i, j, k, inc);
                    // }
                }
            }
}

HD_CALLABLE bool Weilume::coordinatesInBoundary(const int iCol,
                                                const int iRow,
                                                const int iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol >  _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
    {
        printf("***Weilume out of bounary: (%4d,%4d,%4d)!\n",
               iCol, iRow, iSlc);
        return false;
    }else
        return true;
}


//new add
HD_CALLABLE Weilume::Weilume(const Weilume& vol)
{
    // Copy-constructor will be called when
    // arguments of kernel are passed by object
    // value.
    //
    // This explicitly defines device end ctor.

    _nCol = vol._nCol;
    _nRow = vol._nRow;
    _nSlc = vol._nSlc;

    _size = vol._size;

    _devDataFT  = vol._devDataFT;
    _hostDataFT = vol._hostDataFT;
}

HD_CALLABLE void Weilume::init(const int nCol,
                               const int nRow,
                               const int nSlc)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;

    _size = (nCol / 2 + 1) * nRow * nSlc;
}

HD_CALLABLE void Weilume::init(double* data,
                               const int nCol,
                               const int nRow,
                               const int nSlc)
{
    devPtr(data);

    init(nCol, nRow, nSlc);
}

HD_CALLABLE void Weilume::init(cudaArray* data,
                               const int nCol,
                               const int nRow,
                               const int nSlc)
{
    _cudaarray = data;

    init(nCol, nRow, nSlc);
}

HD_CALLABLE bool Weilume::coordinatesInBoundaryFT(const int iCol,
                                                  const int iRow,
                                                  const int iSlc) const
{
    if ((iCol < -_nCol / 2) || (iCol > _nCol / 2) ||
        (iRow < -_nRow / 2) || (iRow >= _nRow / 2) ||
        (iSlc < -_nSlc / 2) || (iSlc >= _nSlc / 2))
    {
        printf("*** Weilume index out of boundary!\n");
        return false;
    }else
        return true;
}

D_CALLABLE bool Weilume::conjHalf(double& iCol,
                                  double& iRow,
                                  double& iSlc) const
{
    if (iCol >= 0) return false;

    iCol *= -1;
    iRow *= -1;
    iSlc *= -1;

    return true;
}

D_CALLABLE void Weilume::getFromIndex(const int index, 
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

D_CALLABLE int Weilume::getIndexHalf(int i, int j, int k) const
{
    if (j < 0) j += _nRow;
    if (k < 0) k += _nSlc;

    return CU_WEILUME_INDEX_FT(i, j, k);
}

D_CALLABLE double Weilume::getFTHalf(const int iCol,
                                     const int iRow,
                                     const int iSlc) const
{
    if(!coordinatesInBoundaryFT(iCol, iRow, iSlc))
        return 0.0;
    
    int index = getIndexHalf(iCol, iRow, iSlc);
    return _devDataFT[index];
}

D_CALLABLE double Weilume::getByInterpolationFT(double iCol,
                                                double iRow,
                                                double iSlc,
                                                const int interp) const
{
    bool conjug = conjHalf(iCol, iRow, iSlc);

    if(interp == 0)
    {
        double result = getFTHalf((int)rint(iCol), 
                                  (int)rint(iRow),
                                  (int)rint(iSlc));
        return result;
    }

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);


    double result = 0.0;
    /*
    if((x0[1] != -1) && x0[2] != -1)
    {
        int index0 = getIndexHalf(x0[0], x0[1], x0[2]);
        for (int k = 0; k < 2; k++)
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++)
                {
                    int index = index0 + CU_WEILUME_INDEX_FT(i, j, k);
                    result += _devDataFT[index] * w[k][j][i];
                    //printf("w:%lf,box:%d\n", w[k][j][i], CU_WEILUME_INDEX_FT(i, j, k));
                }
    }
    else
    {
    */    
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++){
                result += getFTHalf(x0[0] + i, x0[1] + j, x0[2] + k)
                        * w[k][j][i];
            } 
    //} 
    return result;
}

} // end namespace cuthunder
