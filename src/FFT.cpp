/** @file
 *  @author Mingxu Hu
 *  @version 1.4.13.190622
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2019/06/22 | 1.4.13.190622 | change int to long to fix overflow issue
 */

#include "FFT.h"

#include <omp_compat.h>

FFT::FFT() : _srcR(NULL),
             _srcC(NULL),
             _dstR(NULL),
             _dstC(NULL),
             fwPlan(NULL),
             bwPlan(NULL){}

FFT::~FFT() {}

/*void FFT::fw(Image& img)
{
    FW_EXTRACT_P(img);
    
    #pragma omp critical  (line28)
    fwPlan = TSFFTW_plan_dft_r2c_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    TSFFTW_execute(fwPlan);

    FW_CLEAN_UP;
}

void FFT::bw(Image& img)
{
    BW_EXTRACT_P(img);
   
    #pragma omp critical  (line44)
    bwPlan = TSFFTW_plan_dft_c2r_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    TSFFTW_execute(bwPlan);

    SCALE_RL(img, 1.0 / img.sizeRL());

    BW_CLEAN_UP(img);
}
*/

/**
void FFT::fw(Volume& vol)
{
    FW_EXTRACT_P(vol);

    if (vol.nSlcRL() == 1)
    {
        #pragma omp critical  (line64)
        fwPlan = TSFFTW_plan_dft_r2c_2d(vol.nRowRL(),
                                      vol.nColRL(),
                                      _srcR,
                                      _dstC,
                                      FFTW_ESTIMATE);
    }
    else
    {
        #pragma omp critical  (line73)
        fwPlan = TSFFTW_plan_dft_r2c_3d(vol.nRowRL(),
                                      vol.nColRL(),
                                      vol.nSlcRL(),
                                      _srcR,
                                      _dstC,
                                      FFTW_ESTIMATE);
    }

    TSFFTW_execute(fwPlan);

    FW_CLEAN_UP;
}

void FFT::bw(Volume& vol)
{
    BW_EXTRACT_P(vol);

    if (vol.nSlcRL() == 1)
    {
        #pragma omp critical  (line92)
        fwPlan = TSFFTW_plan_dft_r2c_2d(vol.nRowRL(),
                                      vol.nColRL(),
                                      _srcR,
                                      _dstC,
                                      FFTW_ESTIMATE);
    }
    else
    {
        #pragma omp critical  (line102)
        bwPlan = TSFFTW_plan_dft_c2r_3d(vol.nRowRL(),
                                      vol.nColRL(),
                                      vol.nSlcRL(),
                                      _srcC,
                                      _dstR,
                                      FFTW_ESTIMATE);
    }

    TSFFTW_execute(bwPlan);

    SCALE_RL(vol, 1.0 / vol.sizeRL());

    BW_CLEAN_UP(vol);
}*/

void FFT::fw(Image& img,
             const unsigned int nThread)
{
    /***
    img.alloc(FT_SPACE);
    _dstC = (TSFFTW_COMPLEX*)&img[0];
    _srcR = &img(0);
    CHECK_SPACE_VALID(_dstC, _srcR);
    ***/
    FW_EXTRACT_P(img);

    TSFFTW_plan_with_nthreads(nThread);

    fwPlan = TSFFTW_plan_dft_r2c_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcR,
                                  _dstC,
                                  FFTW_ESTIMATE);

    TSFFTW_plan_with_nthreads(1);

    TSFFTW_execute(fwPlan);

    FW_CLEAN_UP_MT;
}

void FFT::bw(Image& img,
             const unsigned int nThread)
{
    /***
    img.alloc(RL_SPACE);
    _dstR = &img(0);
    _srcC = (TSFFTW_COMPLEX*)&img[0];
    CHECK_SPACE_VALID(_dstR, _srcC);
    ***/
    BW_EXTRACT_P(img);

    TSFFTW_plan_with_nthreads(nThread);

    bwPlan = TSFFTW_plan_dft_c2r_2d(img.nRowRL(),
                                  img.nColRL(),
                                  _srcC,
                                  _dstR,
                                  FFTW_ESTIMATE);

    TSFFTW_plan_with_nthreads(1);

    TSFFTW_execute(bwPlan);

    #pragma omp parallel for num_threads(nThread) 
    SCALE_RL(img, 1.0 / img.sizeRL());

    BW_CLEAN_UP_MT(img);
}

void FFT::fw(Volume& vol,
             const unsigned int nThread)
{
    FW_EXTRACT_P(vol);

    TSFFTW_plan_with_nthreads(nThread);

    if (vol.nSlcRL() == 1)
        fwPlan = TSFFTW_plan_dft_r2c_2d(vol.nRowRL(),
                                        vol.nColRL(),
                                        _srcR,
                                        _dstC,
                                        FFTW_ESTIMATE);
    else
        fwPlan = TSFFTW_plan_dft_r2c_3d(vol.nRowRL(),
                                        vol.nColRL(),
                                        vol.nSlcRL(),
                                        _srcR,
                                        _dstC,
                                        FFTW_ESTIMATE);

    TSFFTW_plan_with_nthreads(1);

    TSFFTW_execute(fwPlan);

    FW_CLEAN_UP_MT;
}

void FFT::bw(Volume& vol,
             const unsigned int nThread)
{
    BW_EXTRACT_P(vol);

    TSFFTW_plan_with_nthreads(nThread);

    if (vol.nSlcRL() == 1)
        bwPlan = TSFFTW_plan_dft_c2r_2d(vol.nRowRL(),
                                      vol.nColRL(),
                                      _srcC,
                                      _dstR,
                                      FFTW_ESTIMATE);
    else
        bwPlan = TSFFTW_plan_dft_c2r_3d(vol.nRowRL(),
                                      vol.nColRL(),
                                      vol.nSlcRL(),
                                      _srcC,
                                      _dstR,
                                      FFTW_ESTIMATE);
    TSFFTW_plan_with_nthreads(1);

    TSFFTW_execute(bwPlan);

    #pragma omp parallel for num_threads(nThread) 
    SCALE_RL(vol, 1.0 / vol.sizeRL());

    BW_CLEAN_UP_MT(vol);
}

void FFT::fwCreatePlan(const long nCol,
                       const long nRow,
                       const unsigned int nThread)
{
    _srcR = (RFLOAT*)TSFFTW_malloc(nCol * nRow * sizeof(RFLOAT));
    _dstC = (TSFFTW_COMPLEX*)TSFFTW_malloc((nCol / 2 + 1) * nRow * sizeof(Complex));

    TSFFTW_plan_with_nthreads(nThread);

    fwPlan = TSFFTW_plan_dft_r2c_2d(nRow,
                                  nCol,
                                  _srcR,
                                  _dstC,
                                  FFTW_MEASURE);

    TSFFTW_plan_with_nthreads(1);

    TSFFTW_free(_srcR);
    TSFFTW_free(_dstC);
}

void FFT::fwCreatePlan(const long nCol,
                       const long nRow,
                       const long nSlc,
                       const unsigned int nThread)
{
    _srcR = (RFLOAT*)TSFFTW_malloc(nCol * nRow * nSlc * sizeof(RFLOAT));
    _dstC = (TSFFTW_COMPLEX*)TSFFTW_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));

    TSFFTW_plan_with_nthreads(nThread);

    fwPlan = TSFFTW_plan_dft_r2c_3d(nRow,
                                  nCol,
                                  nSlc,
                                  _srcR,
                                  _dstC,
                                  FFTW_MEASURE);

    TSFFTW_plan_with_nthreads(1);

    TSFFTW_free(_srcR);
    TSFFTW_free(_dstC);
}

void FFT::bwCreatePlan(const long nCol,
                       const long nRow,
                       const unsigned int nThread)
{
    _srcC = (TSFFTW_COMPLEX*)TSFFTW_malloc((nCol / 2 + 1) * nRow * sizeof(Complex));
    _dstR = (RFLOAT*)TSFFTW_malloc(nCol * nRow * sizeof(RFLOAT));
 
    TSFFTW_plan_with_nthreads(nThread);

    bwPlan = TSFFTW_plan_dft_c2r_2d(nRow,
                                  nCol,
                                  _srcC,
                                  _dstR,
                                  FFTW_MEASURE);

    TSFFTW_plan_with_nthreads(1);

    TSFFTW_free(_srcC);
    TSFFTW_free(_dstR);
}

void FFT::bwCreatePlan(const long nCol,
                       const long nRow,
                       const long nSlc,
                       const unsigned int nThread)
{
    _srcC = (TSFFTW_COMPLEX*)TSFFTW_malloc((nCol / 2 + 1) * nRow * nSlc * sizeof(Complex));
    _dstR = (RFLOAT*)TSFFTW_malloc(nCol * nRow * nSlc * sizeof(RFLOAT));

    TSFFTW_plan_with_nthreads(nThread);

    bwPlan = TSFFTW_plan_dft_c2r_3d(nRow,
                                  nCol,
                                  nSlc,
                                  _srcC,
                                  _dstR,
                                  FFTW_MEASURE);

    TSFFTW_plan_with_nthreads(1);

    TSFFTW_free(_srcC);
    TSFFTW_free(_dstR);
}

void FFT::fwExecutePlan(Image& img)
{
    FW_EXTRACT_P(img);

    TSFFTW_execute_dft_r2c(fwPlan, _srcR, _dstC);

    _srcR = NULL;
    _dstC = NULL;
}

void FFT::fwExecutePlan(Volume& vol)
{
    FW_EXTRACT_P(vol);

    TSFFTW_execute_dft_r2c(fwPlan, _srcR, _dstC);

    _srcR = NULL;
    _dstC = NULL;
}

void FFT::bwExecutePlan(Image& img,
                        const unsigned int nThread)
{
    BW_EXTRACT_P(img);

    TSFFTW_execute_dft_c2r(bwPlan, _srcC, _dstR);

    #pragma omp parallel for num_threads(nThread)
    SCALE_RL(img, 1.0 / img.sizeRL());

    _srcC = NULL;
    _dstR = NULL;

    img.clearFT();
}

void FFT::bwExecutePlan(Volume& vol,
                        const unsigned int nThread)
{
    BW_EXTRACT_P(vol);

    TSFFTW_execute_dft_c2r(bwPlan, _srcC, _dstR);

    #pragma omp parallel for num_threads(nThread)
    SCALE_RL(vol, 1.0 / vol.sizeRL());

    _srcC = NULL;
    _dstR = NULL;

    vol.clearFT();
}

void FFT::fwDestroyPlan()
{
    if (fwPlan)
    {
        TSFFTW_destroy_plan(fwPlan);

        fwPlan = NULL;
    }
}

void FFT::bwDestroyPlan()
{
    if (bwPlan)
    {
        TSFFTW_destroy_plan(bwPlan);

        bwPlan = NULL;
    }
}
