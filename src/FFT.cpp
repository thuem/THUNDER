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

FFT::FFT() {}

FFT::~FFT() {}

void FFT::fw(Image& img)
{
    FW_EXTRACT_P(img);

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
    BW_EXTRACT_P(img);

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
    FW_EXTRACT_P(vol);

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
    BW_EXTRACT_P(vol);

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
    FW_EXTRACT_P(img);

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    fwPlan = fftw_plan_dft_r2c_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    fftw_execute(fwPlan);

    FW_CLEAN_UP;

    fftw_cleanup_threads();
}

void FFT::bwMT(Image& img)
{
    BW_EXTRACT_P(img);

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    bwPlan = fftw_plan_dft_c2r_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    fftw_execute(bwPlan);

    fftw_cleanup_threads();

    #pragma omp parallel for
    SCALE_RL(img, 1.0 / img.sizeRL());

    BW_CLEAN_UP(img);
}

void FFT::fwMT(Volume& vol)
{
    FW_EXTRACT_P(vol);

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    fwPlan = fftw_plan_dft_r2c_3d(vol.nRowRL(),
                                  vol.nColRL(),
                                  vol.nSlcRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    fftw_execute(fwPlan);

    FW_CLEAN_UP;

    fftw_cleanup_threads();
}

void FFT::bwMT(Volume& vol)
{
    BW_EXTRACT_P(vol);

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    bwPlan = fftw_plan_dft_c2r_3d(vol.nRowRL(),
                                  vol.nColRL(),
                                  vol.nSlcRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    fftw_execute(bwPlan);

    fftw_cleanup_threads();

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

    _srcR = (double*)fftw_malloc(nCol * nRow * nSlc * sizeof(double);
    _dstC = (Complex*)fftw_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));

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

    _srcC = (Complex*)fftw_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));
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
    /***
    _srcR = fftw_alloc_real(nCol * nRow * nSlc);
    _dstC = fftw_alloc_complex((nCol / 2 + 1) * nRow * nSlc);
    ***/

    _srcR = (double*)fftw_malloc(nCol * nRow * nSlc * sizeof(double);
    _dstC = (Complex*)fftw_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    fwPlan = fftw_plan_dft_r2c_3d(nRow,
                                  nCol,
                                  nSlc,
                                  _srcR,
                                  _dstC,
                                  FFTW_MEASURE);

    fftw_free(_srcR);
    fftw_free(_dstC);
}

void FFT::bwCreatePlanMT(const int nCol,
                         const int nRow,
                         const int nSlc)
{
    /***
    _srcC = fftw_alloc_complex((nCol / 2 + 1) * nRow * nSlc);
    _dstR = fftw_alloc_real(nCol * nRow * nSlc);
    ***/

    _srcC = (Complex*)fftw_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));
    _dstR = (double*)fftw_malloc(nCol * nRow * nSlc * sizeof(double));

    fftw_init_threads();

    fftw_plan_with_nthreads(omp_get_max_threads());

    bwPlan = fftw_plan_dft_c2r_3d(nRow,
                                  nCol,
                                  nSlc,
                                  _srcC,
                                  _dstR,
                                  FFTW_MEASURE);

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
    #pragma omp critical
    fftw_destroy_plan(fwPlan);
}

void FFT::bwDestroyPlan()
{
    #pragma omp critical
    fftw_destroy_plan(bwPlan);
}

void FFT::fwDestroyPlanMT()
{
    fftw_destroy_plan(fwPlan);

    fftw_cleanup_threads();
}

void FFT::bwDestroyPlanMT()
{
    fftw_destroy_plan(bwPlan);

    fftw_cleanup_threads();
}
