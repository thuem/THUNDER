/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "ImageFunctions.h"

void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow)
{
    double rCol = nTransCol / src.nColRL();
    double rRow = nTransRow / src.nRowRL();

    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        double phase= gsl_pow_2(M_PI) * (i * rCol + j * rRow);
        dst.setFT(src.getFT(i, j) * COMPLEX(cos(phase), -sin(phase)), i, j);
    }
}

void meshReverse(Image& img)
{
    IMAGE_FOR_EACH_PIXEL_FT(img)
        if ((i + j) % 2 == 1)
            img.setFT(-img.getFT(i, j), i, j);
}

void meshReverse(Volume& vol)
{
    VOLUME_FOR_EACH_PIXEL_FT(vol)
        if ((i + j + k) % 2 == 1)
            vol.setFT(-vol.getFT(i, j, k), i, j, k);
}
