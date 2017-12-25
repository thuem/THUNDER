//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Filter.h"

void bFactorFilter(Image& dst,
                   const Image& src,
                   const RFLOAT bFactor)
{
    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT f = TSGSL_pow_2(RFLOAT(i) / src.nColRL())
                 + TSGSL_pow_2(RFLOAT(j) / src.nRowRL());

        //dst.setFT(src.getFT(i, j) * exp(-0.25 * bFactor * f), i, j);
        dst.setFT(src.getFT(i, j) * exp(-0.5 * bFactor * f), i, j);
    }
}

void bFactorFilter(Volume& dst,
                   const Volume& src,
                   const RFLOAT bFactor)
{
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT f = TSGSL_pow_2(RFLOAT(i) / src.nColRL())
                 + TSGSL_pow_2(RFLOAT(j) / src.nRowRL())
                 + TSGSL_pow_2(RFLOAT(k) / src.nSlcRL());

        //dst.setFT(src.getFT(i, j, k) * exp(-0.25 * bFactor * f), i, j, k);
        dst.setFT(src.getFT(i, j, k) * exp(-0.5 * bFactor * f), i, j, k);
    }
}

void lowPassFilter(Image& dst,
                   const Image& src,
                   const RFLOAT thres,
                   const RFLOAT ew)
{
    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT f = NORM(RFLOAT(i) / src.nColRL(),
                        RFLOAT(j) / src.nRowRL());
        if (f < thres)
            dst.setFT(src.getFT(i, j), i, j);
        else if (f > thres + ew)
            dst.setFT(COMPLEX(0, 0), i, j);
        else
            dst.setFT(src.getFT(i, j) 
                    * (cos((f - thres) * M_PI / ew) / 2 + 0.5),
                      i,
                      j);
    }
}

void lowPassFilter(Volume& dst,
                   const Volume& src,
                   const RFLOAT thres,
                   const RFLOAT ew)
{
    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT f = NORM_3(RFLOAT(i) / src.nColRL(),
                          RFLOAT(j) / src.nRowRL(),
                          RFLOAT(k) / src.nSlcRL());
        if (f < thres)
            dst.setFT(src.getFT(i, j, k), i, j, k);
        else if (f > thres + ew)
            dst.setFT(COMPLEX(0, 0), i, j, k);
        else
            dst.setFT(src.getFT(i, j, k)
                    * (cos((f - thres) * M_PI / ew) / 2 + 0.5),
                      i,
                      j,
                      k);
    }
}

void highPassFilter(Image& dst,
                    const Image& src,
                    const RFLOAT thres,
                    const RFLOAT ew)
{
    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT f = NORM(RFLOAT(i) / src.nColRL(),
                        RFLOAT(j) / src.nRowRL());

        if (f > thres)
            dst.setFT(src.getFT(i, j), i, j);
        else if (f < thres - ew)
            dst.setFT(COMPLEX(0, 0), i, j);
        else
            dst.setFT(src.getFT(i, j) 
                    * (cos((thres - f) * M_PI / ew) / 2 + 0.5),
                      i,
                      j);
    }
}

void highPassFilter(Volume& dst,
                    const Volume& src,
                    const RFLOAT thres,
                    const RFLOAT ew)
{
    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT f = NORM_3(RFLOAT(i) / src.nColRL(),
                          RFLOAT(j) / src.nRowRL(),
                          RFLOAT(k) / src.nSlcRL());

        if (f > thres)
            dst.setFT(src.getFT(i, j, k), i, j, k);
        else if (f < thres - ew)
            dst.setFT(COMPLEX(0, 0), i, j, k);
        else
            dst.setFT(src.getFT(i, j, k)
                    * (cos((thres - f) * M_PI / ew) / 2 + 0.5),
                      i,
                      j,
                      k);
    }
}

void fscWeightingFilter(Volume& dst,
                        const Volume& src,
                        const vec& fsc)
{
    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(src)
    {
        RFLOAT f = NORM_3(RFLOAT(i) / src.nColRL(),
                          RFLOAT(j) / src.nRowRL(),
                          RFLOAT(k) / src.nSlcRL());

        int idx = AROUND(f * src.nColRL());

        if (idx < fsc.size())
            dst.setFT(src.getFT(i, j, k)
                    * sqrt(GSL_MAX_DBL(0, 2 * fsc(idx) / (1 + fsc(idx)))),
                      i,
                      j,
                      k);
        else
            dst.setFT(COMPLEX(0, 0), i, j, k);
    }
}
