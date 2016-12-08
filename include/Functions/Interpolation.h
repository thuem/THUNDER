/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <gsl/gsl_sf_trig.h>

#include <cstdlib>
#include <cmath>

/**
 * nearest point interpolation
 */
#define NEAREST_INTERP 0

/**
 * linear interpolation
 */
#define LINEAR_INTERP 1

/**
 * sinc interpolation
 */
#define SINC_INTERP 2

/**
 * This macro loops over a 2D cell.
 */
#define FOR_CELL_DIM_2 \
    for (int i = 0; i < 2; i++) \
        for (int j = 0; j < 2; j++)

/**
 * This macro loops over a 3D cell.
 */
#define FOR_CELL_DIM_3 \
    for (int i = 0; i < 2; i++) \
        for (int j = 0; j < 2; j++) \
            for (int k = 0; k < 2; k++)

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
 * These empty structs are used as tag classes for interpolation algorithm 
 * selection with Template classes
 */
struct InterpolaTypeNearest{};
struct InterpolaTypeLinear{};
struct InterpolaTypeSinc{};

/**
 * This function determines the weights of two sampling points during 1D nearest
 * point interpolation given the distance between the interpolation point and
 * the first sampling point.
 *
 * @param w  2-array indicating the weights
 * @param xd the distance between the interpolation point and the first sampling
 *           point
 */

inline void Weight_Interpolation(double w[2], 
                                const double xd, 
                                InterpolaTypeNearest)
{
    w[0] = 0;
    w[1] = 0;
    if (xd < 0.5) 
        w[0] = 1;
    else
        w[1] = 1;
}

/**
 * This function determines the weights of two sampling points during 1D linear
 * interpolation given the distance between the interpolation point and the
 * first sampling point.
 *
 * @param w  2-array indicating the weights
 * @param xd the distance between the interpolation point and the first sampling
 *           point
 */
inline void Weight_Interpolation(double w[2], 
                                const double xd, 
                                InterpolaTypeLinear)
{
    w[0] = 1 - xd;
    w[1] = xd;
}

/**
 * This function determines the weights of two sampling points during 1D sinc
 * interpolation given the distance between the interpolation point and the
 * first sampling point.
 *
 * @param w  2-array indicating the weights
 * @param xd the distance between the interpolation point and the first sampling
 *           point
 */
inline void Weight_Interpolation(double w[2], 
                                const double xd, 
                                InterpolaTypeSinc)
{
    w[0] = gsl_sf_sinc(xd);
    w[1] = gsl_sf_sinc(1 - xd);
}


/**
 * This template function determines the weights of two sampling points and
 * the coordinate of the first sampling point during 1D interpolation given
 * the coordinate of interpolation point using <Template class InterpolaType>.
 *
 * @param w  2-array indicating the weights
 * @param x0 the nearest grid point of the interpolation point
 * @param x  the interpolation point
 */
template <class InterpolaType> 
inline void Weight_Interpolation(double w[2], const double x)
{   
    Weight_Interpolation(w, x, InterpolaType());
}

template <class InterpolaType> 
inline void WeightGpoint_Interpolation(double w[2], int& x0, const double x)
{   
    x0 = floor(x);
    Weight_Interpolation<InterpolaType>(w, x - x0);
}


template <class InterpolaType>
inline void Weight_BI_Interpolation(double w[2][2], const double xd[2])
{
    double v[2][2];
    for (int i = 0; i < 2; i++) {
        Weight_Interpolation(v[i], xd[i], InterpolaType());
    }
    FOR_CELL_DIM_2 w[i][j] = v[0][i] * v[1][j];
}

template <class InterpolaType>
inline void WeightGpoint_BI_Interpolation(double w[2][2], 
                                      int x0[2], 
                                      const double x[2])
{
    double xd[2];
    for (int i = 0; i < 2; i++) {
        x0[i] = floor(x[i]);
        xd[i] = x[i] - x0[i];
    }
    Weight_BI_Interpolation<InterpolaType>(w, xd);
}

template <class InterpolaType>
inline void Weight_TRI_Interpolation(double w[2][2][2], const double xd[3])
{
    double v[3][2];
    for (int i = 0; i < 3; i++) {
        Weight_Interpolation(v[i], xd[i], InterpolaType());
    }
    FOR_CELL_DIM_3 w[i][j][k] = v[0][i] * v[1][j] * v[2][k];
}


template <class InterpolaType>
inline void WeightGpoint_TRI_Interpolation(double w[2][2][2], 
                                            int x0[3], 
                                            const double x[3])
{
    double xd[3];
    for (int i = 0; i < 3; i++) {
        x0[i] = floor(x[i]);
        xd[i] = x[i] - x0[i];
    }
    Weight_TRI_Interpolation<InterpolaType>(w, xd);
}


#endif // INTERPOLATION_H
