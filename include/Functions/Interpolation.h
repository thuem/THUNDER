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

#define NEAREST_INTERP 0

#define LINEAR_INTERP 1

#define SINC_INTERP 2

#define FOR_CELL_DIM_2 \
    for (int i = 0; i < 2; i++) \
        for (int j = 0; j < 2; j++)

#define FOR_CELL_DIM_3 \
    for (int i = 0; i < 2; i++) \
        for (int j = 0; j < 2; j++) \
            for (int k = 0; k < 2; k++)

#define LINEAR(v, xd) (v[0] * (1 - (xd)) + v[1] * (xd))
/* v[2], xd */

#define BI_LINEAR(v, xd) (LINEAR(v[0], xd[0]) * (1 - (xd[1])) \
                        + LINEAR(v[1], xd[0]) * (xd[1]))
/* v[2][2], xd[2] */

#define TRI_LINEAR(v, xd) (BI_LINEAR(v[0], xd) * (1 - (xd[2])) \
                         + BI_LINEAR(v[1], xd) * (xd[2]))
/* v[2][2][2], xd[2][2] */

/* W_ -> Weight, WG_ -> Weight & Grid */

#define W_NEAREST(w, xd) \
    [](double _w[2], const double _xd) \
    { \
        _w[0] = 0; \
        _w[1] = 0; \
        (_xd < 0.5) ? _w[0]++ : _w[1]++; \
    }(w, xd)

#define W_LINEAR(w, xd) \
    [](double _w[2], const double _xd) \
    { \
        _w[0] = 1 - _xd; \
        _w[1] = _xd; \
    }(w, xd)

#define W_SINC(w, xd) \
    [](double _w[2], const double _xd) \
    { \
        _w[0] = gsl_sf_sinc(_xd); \
        _w[1] = gsl_sf_sinc(1 - _xd); \
    }(w, xd)

#define WG_INTERP(INTERP, w, x0, x) \
    [](double _w[2], int& _x0, const double _x) \
    { \
        _x0 = floor(_x); \
        W_##INTERP(_w, _x - _x0); \
    }(w, x0, x)

#define WG_NEAREST(w, x0, x) WG_INTERP(NEAREST, w, x0, x)

#define WG_LINEAR(w, x0, x) WG_INTERP(LINEAR, w, x0, x)

#define WG_SINC(w, x0, x) WG_INTERP(SINC, w, x0, x)

#define W_BI_INTERP(INTERP, w, xd) \
    [](double _w[2][2], const double _xd[2]) \
    { \
        double v[2][2]; \
        for (int i = 0; i < 2; i++) W_##INTERP(v[i], _xd[i]); \
        FOR_CELL_DIM_2 _w[i][j] = v[0][i] * v[1][j]; \
    }(w, xd)

#define W_BI_NEAREST(w, xd) W_BI_INTERP(NEAREST, w, xd)

#define W_BI_LINEAR(w, xd) W_BI_INTERP(LINEAR, w, xd)

#define W_BI_SINC(w, xd) W_BI_INTERP(SINC, w, xd)

#define WG_BI_INTERP(INTERP, w, x0, x) \
    [](double _w[2][2], int _x0[2], const double _x[2]) \
    { \
        double xd[2]; \
        for (int i = 0; i < 2; i++) \
        { \
            _x0[i] = floor(_x[i]); \
            xd[i] = _x[i] - _x0[i]; \
        } \
        W_BI_##INTERP(_w, xd); \
    }(w, x0, x)

#define WG_BI_NEAREST(w, x0, x) WG_BI_INTERP(NEAREST, w, x0, x)

#define WG_BI_LINEAR(w, x0, x) WG_BI_INTERP(LINEAR, w, x0, x)

#define WG_BI_SINC(w, x0, x) WG_BI_INTERP(SINC, w, x0, x)

#define W_TRI_INTERP(INTERP, w, xd) \
    [](double _w[2][2][2], const double _xd[3]) \
    { \
        double v[3][2]; \
        for (int i = 0; i < 3; i++) W_##INTERP(v[i], _xd[i]); \
        FOR_CELL_DIM_3 _w[i][j][k] = v[0][i] * v[1][j] * v[2][k]; \
    }(w, xd)

#define W_TRI_NEAREST(w, xd) W_TRI_INTERP(NEAREST, w, xd)

#define W_TRI_LINEAR(w, xd) W_TRI_INTERP(LINEAR, w, xd)

#define W_TRI_SINC(w, xd) W_TRI_INTERP(SINC, w, xd)

#define WG_TRI_INTERP(INTERP, w, x0, x) \
    [](double _w[2][2][2], int _x0[3], const double _x[3]) \
    { \
        double xd[3]; \
        for (int i = 0; i < 3; i++) \
        { \
            _x0[i] = floor(_x[i]); \
            xd[i] = _x[i] - _x0[i]; \
        } \
        W_TRI_##INTERP(_w, xd); \
    }(w, x0, x)

#define WG_TRI_NEAREST(w, x0, x) WG_TRI_INTERP(NEAREST, w, x0, x)

#define WG_TRI_LINEAR(w, x0, x) WG_TRI_INTERP(LINEAR, w, x0, x)

#define WG_TRI_SINC(w, x0, x) WG_TRI_INTERP(SINC, w, x0, x)

#endif // INTERPOLATION_H
