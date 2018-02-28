/***********************************************************************
 * FileName: Interpolation.cu
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#include "Interpolation.cuh"

namespace cuthunder {

HD_CALLABLE void WG_LINEAR_INTERP(double w[2], int& x0, const double x)
{
    x0 = floor(x);

    w[0] = 1 - (x - x0);
    w[1] = x - x0;
}

HD_CALLABLE void WG_BI_LINEAR_INTERP(double w[2][2], int x0[2], const double x[2])
{
    double xd[2];
    for (int i = 0; i < 2; i++)
    {
        x0[i] = floor(x[i]);
        xd[i] = x[i] - x0[i];
    }
    
    double v[2][2];
    for (int i = 0; i < 2; i++)
    {
        v[i][0] = 1 - xd[i];
        v[i][1] = xd[i];
    }

    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++)
            w[j][i] = v[0][i] * v[1][j];
}

HD_CALLABLE void WG_TRI_LINEAR_INTERPF(double w[2][2][2], int x0[3], const double x[3])
{
    double xd[3];
    for (int i = 0; i < 3; i++)
    {
        x0[i] = floor(x[i]);
        xd[i] = x[i] - x0[i];
    }

    double v[3][2];
    for (int i = 0; i < 3; i++)
    {
        v[i][0] = 1 - xd[i];
        v[i][1] = xd[i];
    }

    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
                w[k][j][i] = v[0][i] * v[1][j] * v[2][k];
}
} // end namespace cuthunder
