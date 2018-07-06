/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "CTF.h"

RFLOAT CTF(const RFLOAT f,
           const RFLOAT voltage,
           const RFLOAT defocus,
           const RFLOAT Cs,
           const RFLOAT amplitudeContrast,
           const RFLOAT phaseShift)
{
    RFLOAT lambda = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));

    RFLOAT w1 = TS_SQRT(1 - TSGSL_pow_2(amplitudeContrast));
    RFLOAT w2 = amplitudeContrast;

    RFLOAT K1 = M_PI * lambda;
    RFLOAT K2 = M_PI_2 * Cs * TSGSL_pow_3(lambda);

    RFLOAT ki = K1 * defocus * TSGSL_pow_2(f) + K2 * TSGSL_pow_4(f) - phaseShift;

    return -w1 * TS_SIN(ki) + w2 * TS_COS(ki);
}

void CTF(Image& dst,
         const RFLOAT pixelSize,
         const RFLOAT voltage,
         const RFLOAT defocusU,
         const RFLOAT defocusV,
         const RFLOAT theta,
         const RFLOAT Cs,
         const RFLOAT amplitudeContrast,
         const RFLOAT phaseShift)
{
    RFLOAT lambda = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));

    RFLOAT w1 = TS_SQRT(1 - TSGSL_pow_2(amplitudeContrast));
    RFLOAT w2 = amplitudeContrast;

    RFLOAT K1 = M_PI * lambda;
    RFLOAT K2 = M_PI_2 * Cs * TSGSL_pow_3(lambda);

    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        RFLOAT u = NORM(i / (pixelSize * dst.nColRL()),
                        j / (pixelSize * dst.nRowRL()));

        RFLOAT angle = atan2(j, i) - theta;
        RFLOAT defocus = -(defocusU + defocusV
                         + (defocusU - defocusV) * TS_COS(2 * angle)) / 2;

        RFLOAT ki = K1 * defocus * TSGSL_pow_2(u) + K2 * TSGSL_pow_4(u) - phaseShift;

        dst.setFTHalf(COMPLEX(-w1 * TS_SIN(ki) + w2 * TS_COS(ki), 0),
                      i,
                      j);
    }
}

void CTF(Image& dst,
         const RFLOAT pixelSize,
         const RFLOAT voltage,
         const RFLOAT defocusU,
         const RFLOAT defocusV,
         const RFLOAT theta,
         const RFLOAT Cs,
         const RFLOAT amplitudeContrast,
         const RFLOAT phaseShift,
         const RFLOAT r)
{
    RFLOAT lambda = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));

    RFLOAT w1 = TS_SQRT(1 - TSGSL_pow_2(amplitudeContrast));
    RFLOAT w2 = amplitudeContrast;

    RFLOAT K1 = M_PI * lambda;
    RFLOAT K2 = M_PI_2 * Cs * TSGSL_pow_3(lambda);

    RFLOAT r2 = TSGSL_pow_2(r);

    IMAGE_FOR_PIXEL_R_FT(r + 1)
    {
        RFLOAT v = QUAD(i, j);

        if (v < r2)
        {
            RFLOAT u = NORM(i / (pixelSize * dst.nColRL()),
                            j / (pixelSize * dst.nRowRL()));

            RFLOAT angle = atan2(j, i) - theta;
            RFLOAT defocus = -(defocusU + defocusV
                             + (defocusU - defocusV) * TS_COS(2 * angle)) / 2;

            RFLOAT ki = K1 * defocus * TSGSL_pow_2(u) + K2 * TSGSL_pow_4(u) - phaseShift;

            dst.setFTHalf(COMPLEX(-w1 * TS_SIN(ki) + w2 * TS_COS(ki), 0),
                          i,
                          j);
        }
    }
}

void CTF(RFLOAT* dst,
         const RFLOAT pixelSize,
         const RFLOAT voltage,
         const RFLOAT defocusU,
         const RFLOAT defocusV,
         const RFLOAT theta,
         const RFLOAT Cs,
         const RFLOAT amplitudeContrast,
         const RFLOAT phaseShift,
         const int nCol,
         const int nRow,
         const int* iCol,
         const int* iRow,
         const int _nPxl)
{
    RFLOAT lambda = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));

    RFLOAT w1 = TS_SQRT(1 - TSGSL_pow_2(amplitudeContrast));
    RFLOAT w2 = amplitudeContrast;

    RFLOAT K1 = M_PI * lambda;
    RFLOAT K2 = M_PI_2 * Cs * TSGSL_pow_3(lambda);

    for (int i = 0; i < _nPxl; i++)
    {
        RFLOAT u = NORM(iCol[i] / (pixelSize * nCol),
                        iRow[i] / (pixelSize * nRow));

        RFLOAT angle = atan2(iRow[i], iCol[i]) - theta;
        RFLOAT defocus = -(defocusU + defocusV
                         + (defocusU - defocusV) * TS_COS(2 * angle)) / 2;

        RFLOAT ki = K1 * defocus * TSGSL_pow_2(u) + K2 * TSGSL_pow_4(u) - phaseShift;

        dst[i] = -w1 * TS_SIN(ki) + w2 * TS_COS(ki);
    }
}
