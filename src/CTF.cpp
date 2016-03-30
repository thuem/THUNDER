/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "CTF.h"

void CTF(Image& dst,
         const double pixelSize,
         const double voltage,
         const double defocusU,
         const double defocusV,
         const double theta,
         const double Cs)
{
    double lambda = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));

    double K1 = M_PI * lambda;
    double K2 = M_PI / 2 * Cs * gsl_pow_3(lambda);

    IMAGE_FOR_EACH_PIXEL_FT(dst)
    {
        double u = NORM(i / (pixelSize * dst.nColRL()),
                        j / (pixelSize * dst.nRowRL()));

        double angle = atan2(j, i) - theta;
        double defocus = -(defocusU + defocusV
                         + (defocusU - defocusV) * cos(2 * angle)) / 2;
        dst.setFT(COMPLEX(cos(K1 * defocus * gsl_pow_2(u)
                            + K2 * gsl_pow_4(u)),
                          0),
                  i,
                  j);
    }
}
