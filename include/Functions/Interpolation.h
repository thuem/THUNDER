/*******************************************************************************
 * Author: Mingxu Hu, Hongkun Yu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <cstdlib>
#include <cmath>

#include <gsl/gsl_sf_trig.h>

#include "Functions.h"

#define NEAREST_INTERP 0

/**
 * linear interpolation
 */
#define LINEAR_INTERP 1

/**
 * This macro loops over a 2D cell.
 */
#define FOR_CELL_DIM_2 \
    for (int j = 0; j < 2; j++) \
        for (int i = 0; i < 2; i++)
/***
    for (int i = 0; i < 2; i++) \
        for (int j = 0; j < 2; j++)
        ***/

/**
 * This macro loops over a 3D cell.
 */
#define FOR_CELL_DIM_3 \
    for (int k = 0; k < 2; k++) \
        for (int j = 0; j < 2; j++) \
            for (int i = 0; i < 2; i++)
    /***
    for (int i = 0; i < 2; i++) \
        for (int j = 0; j < 2; j++) \
            for (int k = 0; k < 2; k++)
            ***/

/**
 * This macro calculates the 1D linear interpolation result given values of two
 * sampling points and the distance between the interpolation point and the
 * first sampling point.
 *
 * @param v  a 2-array indicating the values of two sampling points
 * @param xd the distance between the interpolation point and the first sampling
 *           points
 */
#define LINEAR(v, xd) (v[0] * (1 - (xd)) + v[1] * (xd))

/**
 * This macro calculates the 2D linear interpolation result given values of
 * 2D cell of sampling points and the distances between the interpolation point
 * and the first sampling points along X-axis and Y-axis.
 *
 * @param v  a 2D cell indicating the values of four sampling points
 * @param xd a 2-array indicating the distances between the interpolation point
 *           and the first sampling points along X-axis and Y-axis
 */
#define BI_LINEAR(v, xd) (LINEAR(v[0], xd[0]) * (1 - (xd[1])) \
                        + LINEAR(v[1], xd[0]) * (xd[1]))

/**
 * This macro calculates the 2D linear interpolation result given values of 3D
 * cell of sampling points and the distances between the interpolation point and
 * the first sampling points along X-axis, Y-axis and Z-axis.
 *
 * @param v  a 3D cell indicating the values of eight sampling points
 * @param xd a 3-array indicating the distances between the interpolation point
 *           and the first sampling points along X-axis, Y-axis and Z-axis.
 */
#define TRI_LINEAR(v, xd) (BI_LINEAR(v[0], xd) * (1 - (xd[2])) \
                         + BI_LINEAR(v[1], xd) * (xd[2]))


/**
 * This function determines the weights of two sampling points during 1D linear
 * interpolation given the distance between the interpolation point and the
 * first sampling point.
 *
 * @param w  2-array indicating the weights
 * @param xd the distance between the interpolation point and the first sampling
 *           point
 */
inline void W_INTERP_LINEAR(double w[2], 
                            const double xd) 
{
    w[0] = 1 - xd;
    w[1] = xd;
}

inline void WG_INTERP_LINEAR(double w[2],
                             int& x0,
                             const double x)
{   
    x0 = floor(x);
    W_INTERP_LINEAR(w, x - x0);
}

inline void W_BI_INTERP_LINEAR(double w[2][2], 
                               const double xd[2])
{
    double v[2][2];
    for (int i = 0; i < 2; i++) 
        W_INTERP_LINEAR(v[i], xd[i]);

    FOR_CELL_DIM_2
    {
        w[j][i] = v[0][i] * v[1][j];
        // w[i][j] = v[0][i] * v[1][j];
    }
}

inline void WG_BI_INTERP_LINEAR(double w[2][2], 
                                int x0[2], 
                                const double x[2])
{
    double xd[2];
    for (int i = 0; i < 2; i++) 
    {
        x0[i] = floor(x[i]);
        xd[i] = x[i] - x0[i];
    }

    W_BI_INTERP_LINEAR(w, xd);
}

inline void W_TRI_INTERP_LINEAR(double w[2][2][2], 
                                const double xd[3])
{
    double v[3][2];
    for (int i = 0; i < 3; i++) 
        W_INTERP_LINEAR(v[i], xd[i]);

    FOR_CELL_DIM_3 
    {
        w[k][j][i] = v[0][i] * v[1][j] * v[2][k];
        //w[i][j][k] = v[0][i] * v[1][j] * v[2][k];
    }
}

inline void WG_TRI_INTERP_LINEAR(double w[2][2][2],
                                 int x0[3], 
                                 const double x[3])
{
    double xd[3];
    for (int i = 0; i < 3; i++)
    {
        x0[i] = floor(x[i]);
        xd[i] = x[i] - x0[i];
    }

    W_TRI_INTERP_LINEAR(w, xd);
}


#endif // INTERPOLATION_H
