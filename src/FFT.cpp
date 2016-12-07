/*******************************************************************************
 * Author: Mingxu Hu, Hongkun Yu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "FFT.h"

#include <omp.h>

FFT::FFT() : _srcR(NULL), _srcC(NULL), _dstR(NULL), _dstC(NULL)  {}

FFT::~FFT() {}

void FFT::fw(Image& img)
{
    img.alloc(FT_SPACE);
    _dstC = (fftw_complex*)&img[0];
    _srcR = &img(0);
    CHECK_SPACE_VALID(_dstC, _srcR);

    #pragma omp critical
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
    img.alloc(RL_SPACE);
    _dstR = &img(0);
    _srcC = (fftw_complex*)&img[0];
    CHECK_SPACE_VALID(_dstR, _srcC);

    #pragma omp critical
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
    vol.alloc(FT_SPACE);
    _dstC = (fftw_complex*)&vol[0];
    _srcR = &vol(0);
    CHECK_SPACE_VALID(_dstC, _srcR);

    #pragma omp critical
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
    vol.alloc(RL_SPACE);
    _dstR = &vol(0);
    _srcC = (fftw_complex*)&vol[0];
    CHECK_SPACE_VALID(_dstR, _srcC);

    #pragma omp critical
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

void FFT::fwMT(Image& img)
{
    img.alloc(FT_SPACE);
    _dstC = (fftw_complex*)&img[0];
    _srcR = &img(0);
    CHECK_SPACE_VALID(_dstC, _srcR);

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    fwPlan = fftw_plan_dft_r2c_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    fftw_execute(fwPlan);

    FWMT_CLEAN_UP;

    fftw_cleanup_threads();
}

void FFT::bwMT(Image& img)
{
    img.alloc(RL_SPACE);
    _dstR = &img(0);
    _srcC = (fftw_complex*)&img[0];
    CHECK_SPACE_VALID(_dstR, _srcC);

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    bwPlan = fftw_plan_dft_c2r_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    fftw_execute(bwPlan);

    BWMT_CLEAN_UP(img);

    fftw_cleanup_threads();
    
    #pragma omp parallel for
    SCALE_RL(img, 1.0 / img.sizeRL());
}

void FFT::fwMT(Volume& vol)
{
    vol.alloc(FT_SPACE);
    _dstC = (fftw_complex*)&vol[0];
    _srcR = &vol(0);
    CHECK_SPACE_VALID(_dstC, _srcR);

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    fwPlan = fftw_plan_dft_r2c_3d(vol.nRowRL(),
                                  vol.nColRL(),
                                  vol.nSlcRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    fftw_execute(fwPlan);

    FWMT_CLEAN_UP;

    fftw_cleanup_threads();
}

void FFT::bwMT(Volume& vol)
{
    vol.alloc(RL_SPACE);
    _dstR = &vol(0);
    _srcC = (fftw_complex*)&vol[0];
    CHECK_SPACE_VALID(_dstR, _srcC);

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    bwPlan = fftw_plan_dft_c2r_3d(vol.nRowRL(),
                                  vol.nColRL(),
                                  vol.nSlcRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    fftw_execute(bwPlan);

    BWMT_CLEAN_UP(vol);

    fftw_cleanup_threads();
    
    #pragma omp parallel for
    SCALE_RL(vol, 1.0 / vol.sizeRL());
}
