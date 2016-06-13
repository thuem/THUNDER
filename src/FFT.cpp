/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "FFT.h"

FFT::FFT() {}

FFT::~FFT() {}

void FFT::fw(Image& img)
{
    FW_EXTRACT_P(img);

    //#pragma omp critical
    fwPlan = fftw_plan_dft_r2c_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    fftw_execute(fwPlan);

    FW_CLEAN_UP;
}

void FFT::bw(Image& img)
{
    BW_EXTRACT_P(img);

    //#pragma omp critical
    bwPlan = fftw_plan_dft_c2r_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    fftw_execute(bwPlan);

    SCALE_RL(img, 1.0 / img.sizeRL());

    BW_CLEAN_UP(img);
}

void FFT::fw(Volume& vol)
{
    FW_EXTRACT_P(vol);

    //#pragma omp critical
    fwPlan = fftw_plan_dft_r2c_3d(vol.nRowRL(),
                                  vol.nColRL(),
                                  vol.nSlcRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    fftw_execute(fwPlan);

    FW_CLEAN_UP;
}

void FFT::bw(Volume& vol)
{
    BW_EXTRACT_P(vol);

    //#pragma omp critical
    bwPlan = fftw_plan_dft_c2r_3d(vol.nRowRL(),
                                  vol.nColRL(),
                                  vol.nSlcRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    fftw_execute(bwPlan);

    SCALE_RL(vol, 1.0 / vol.sizeRL());

    BW_CLEAN_UP(vol);
}
