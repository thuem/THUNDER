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

#include <omp_compat.h>

FFT::FFT() : _srcR(NULL),
             _srcC(NULL),
             _dstR(NULL),
             _dstC(NULL)  {}

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

    fftw_plan_with_nthreads(omp_get_max_threads());

    fwPlan = fftw_plan_dft_r2c_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    fftw_plan_with_nthreads(1);

    fftw_execute(fwPlan);

    FW_CLEAN_UP;
}

void FFT::bwMT(Image& img)
{
    img.alloc(RL_SPACE);
    _dstR = &img(0);
    _srcC = (fftw_complex*)&img[0];
    CHECK_SPACE_VALID(_dstR, _srcC);

    fftw_plan_with_nthreads(omp_get_max_threads());

    bwPlan = fftw_plan_dft_c2r_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    fftw_plan_with_nthreads(1);

    fftw_execute(bwPlan);

    #pragma omp parallel for
    SCALE_RL(img, 1.0 / img.sizeRL());

    BW_CLEAN_UP(img);
}

void FFT::fwMT(Volume& vol)
{
    vol.alloc(FT_SPACE);
    _dstC = (fftw_complex*)&vol[0];
    _srcR = &vol(0);
    CHECK_SPACE_VALID(_dstC, _srcR);

    fftw_plan_with_nthreads(omp_get_max_threads());

    fwPlan = fftw_plan_dft_r2c_3d(vol.nRowRL(),
                                  vol.nColRL(),
                                  vol.nSlcRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    fftw_plan_with_nthreads(1);

    fftw_execute(fwPlan);

    FW_CLEAN_UP;
}

void FFT::bwMT(Volume& vol)
{
    vol.alloc(RL_SPACE);
    _dstR = &vol(0);
    _srcC = (fftw_complex*)&vol[0];
    CHECK_SPACE_VALID(_dstR, _srcC);

    fftw_plan_with_nthreads(omp_get_max_threads());

    bwPlan = fftw_plan_dft_c2r_3d(vol.nRowRL(),
                                  vol.nColRL(),
                                  vol.nSlcRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    fftw_plan_with_nthreads(1);

    fftw_execute(bwPlan);

    #pragma omp parallel for
    SCALE_RL(vol, 1.0 / vol.sizeRL());

    BW_CLEAN_UP(vol);
}

void FFT::fwCreatePlan(const int nCol,
                       const int nRow,
                       const int nSlc)
{
    /***
    _srcR = fftw_alloc_real(nCol * nRow * nSlc);
    _dstC = fftw_alloc_complex((nCol / 2 + 1) * nRow * nSlc);
    ***/

    _srcR = (double*)fftw_malloc(nCol * nRow * nSlc * sizeof(double));
    _dstC = (fftw_complex*)fftw_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));

    fwPlan = fftw_plan_dft_r2c_3d(nRow,
                                  nCol,
                                  nSlc,
                                  _srcR,
                                  _dstC,
                                  FFTW_MEASURE);

    fftw_free(_srcR);
    fftw_free(_dstC);
}

void FFT::bwCreatePlan(const int nCol,
                       const int nRow,
                       const int nSlc)
{
    /***
    _srcC = fftw_alloc_complex((nCol / 2 + 1) * nRow * nSlc);
    _dstR = fftw_alloc_real(nCol * nRow * nSlc);
    ***/

    _srcC = (fftw_complex*)fftw_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));
    _dstR = (double*)fftw_malloc(nCol * nRow * nSlc * sizeof(double));

    #pragma omp critical
    bwPlan = fftw_plan_dft_c2r_3d(nRow,
                                  nCol,
                                  nSlc,
                                  _srcC,
                                  _dstR,
                                  FFTW_MEASURE);

    fftw_free(_srcC);
    fftw_free(_dstR);
}

void FFT::fwCreatePlanMT(const int nCol,
                         const int nRow,
                         const int nSlc)
{
    _srcR = (double*)fftw_malloc(nCol * nRow * nSlc * sizeof(double));
    _dstC = (fftw_complex*)fftw_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));

    fftw_plan_with_nthreads(omp_get_max_threads());

    fwPlan = fftw_plan_dft_r2c_3d(nRow,
                                  nCol,
                                  nSlc,
                                  _srcR,
                                  _dstC,
                                  FFTW_MEASURE);

    fftw_plan_with_nthreads(1);

    fftw_free(_srcR);
    fftw_free(_dstC);
}

void FFT::bwCreatePlanMT(const int nCol,
                         const int nRow,
                         const int nSlc)
{
    _srcC = (fftw_complex*)fftw_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));
    _dstR = (double*)fftw_malloc(nCol * nRow * nSlc * sizeof(double));

    fftw_plan_with_nthreads(omp_get_max_threads());

    bwPlan = fftw_plan_dft_c2r_3d(nRow,
                                  nCol,
                                  nSlc,
                                  _srcC,
                                  _dstR,
                                  FFTW_MEASURE);

    fftw_plan_with_nthreads(1);

    fftw_free(_srcC);
    fftw_free(_dstR);
}

void FFT::fwExecutePlan(Volume& vol)
{
    FW_EXTRACT_P(vol);

    fftw_execute_dft_r2c(fwPlan, _srcR, _dstC);

    _srcR = NULL;
    _dstC = NULL;
}

void FFT::bwExecutePlan(Volume& vol)
{
    BW_EXTRACT_P(vol);

    fftw_execute_dft_c2r(bwPlan, _srcC, _dstR);

    SCALE_RL(vol, 1.0 / vol.sizeRL());

    _srcC = NULL;
    _dstR = NULL;

    vol.clearFT();
}

void FFT::fwExecutePlanMT(Volume& vol)
{
    FW_EXTRACT_P(vol);

    fftw_execute_dft_r2c(fwPlan, _srcR, _dstC);

    _srcR = NULL;
    _dstC = NULL;
}

void FFT::bwExecutePlanMT(Volume& vol)
{
    BW_EXTRACT_P(vol);

    fftw_execute_dft_c2r(bwPlan, _srcC, _dstR);

    #pragma omp parallel for
    SCALE_RL(vol, 1.0 / vol.sizeRL());

    _srcC = NULL;
    _dstR = NULL;

    vol.clearFT();
}

void FFT::fwDestroyPlan()
{
    fftw_destroy_plan(fwPlan);
}

void FFT::bwDestroyPlan()
{
    fftw_destroy_plan(bwPlan);
}

/***
void FFT::fwDestroyPlanMT()
{
    fftw_destroy_plan(fwPlan);
}

void FFT::bwDestroyPlanMT()
{
    fftw_destroy_plan(bwPlan);
    fftw_cleanup_threads();
    
    #pragma omp parallel for
    SCALE_RL(vol, 1.0 / vol.sizeRL());
}
***/
