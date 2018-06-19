/***********************************************************************
 * FileName: Interpolation.cuh
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef INTERPOLATION_CUH
#define INTERPOLATION_CUH

#include "Config.h"

#include "Device.cuh"
#include <cmath>

namespace cuthunder {

#define LINEAR(v, xd) (v[0] * (1 - (xd)) + v[1] * (xd))
/* v[2], xd */

#define BI_LINEAR(v, xd) (LINEAR(v[0], xd[0]) * (1 - (xd[1])) \
                        + LINEAR(v[1], xd[0]) * (xd[1]))
/* v[2][2], xd[2] */

#define TRI_LINEAR(v, xd) (BI_LINEAR(v[0], xd) * (1 - (xd[2])) \
                         + BI_LINEAR(v[1], xd) * (xd[2]))
/* v[2][2][2], xd[2][2] */

/* WG_ -> Weight & Grid */

HD_CALLABLE void WG_LINEAR_INTERP(RFLOAT w[2], int& x0, const RFLOAT x);

HD_CALLABLE void WG_BI_LINEAR_INTERPF(RFLOAT w[2][2], int x0[2], const RFLOAT x[2]);

HD_CALLABLE void WG_TRI_LINEAR_INTERPF(RFLOAT w[2][2][2], int x0[3], const RFLOAT x[3]);


} // end namespace cuthunder

#endif
