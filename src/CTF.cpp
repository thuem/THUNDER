//This header file is add by huabin
#include "huabin.h"
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
           const RFLOAT Cs)
{
    RFLOAT lambda = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));

    RFLOAT K1 = M_PI * lambda;
    RFLOAT K2 = M_PI / 2 * Cs * TSGSL_pow_3(lambda);

    RFLOAT ki = -K1 * defocus * TSGSL_pow_2(f) + K2 * TSGSL_pow_4(f);

    return -w1 * sin(ki) + w2 * cos(ki);
}

void CTF(Image& dst,
         const RFLOAT pixelSize,
         const RFLOAT voltage,
         const RFLOAT defocusU,
         const RFLOAT defocusV,
         const RFLOAT theta,
         const RFLOAT Cs)
{
    RFLOAT lambda = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));

    //std::cout << "lambda = " << lambda << std::endl;

    RFLOAT K1 = M_PI * lambda;
    RFLOAT K2 = M_PI / 2 * Cs * TSGSL_pow_3(lambda);

    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        RFLOAT u = NORM(i / (pixelSize * dst.nColRL()),
                        j / (pixelSize * dst.nRowRL()));

        RFLOAT angle = atan2(j, i) - theta;
        RFLOAT defocus = -(defocusU + defocusV
                         + (defocusU - defocusV) * cos(2 * angle)) / 2;

        RFLOAT ki = K1 * defocus * TSGSL_pow_2(u) + K2 * TSGSL_pow_4(u);

        dst.setFT(COMPLEX(-w1 * sin(ki) + w2 * cos(ki), 0),
                  i,
                  j);
    }
}
