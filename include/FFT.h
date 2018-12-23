/** @file
 *  @author Mingxu Hu
 *  @author Shouqing Li
 *  @version 1.4.11.080914
 *  @copyright THUNDER Non-Commercial Software License Agreement
 *
 *  ChangeLog
 *  AUTHOR      | TIME       | VERSION       | DESCRIPTION
 *  ------      | ----       | -------       | -----------
 *  Mingxu Hu   | 2015/03/23 | 0.0.1.050323  | new file
 *  Shouqing Li | 2018/09/14 | 1.4.11.080914 | add documentation 
 *  Mingxu Hu   | 2018/12/22 | 1.4.11.081222 | add some emendation in the documentation
 *  
 *  @brief FFT.h contains several functions to carry out the Fast Fourier Transformation calculations for various conditions.   
 *
 *  The functions can be divided into four parts. The **CreatePlan** part carries out the function to create plans for Fast Fourier Transformation. The **ExecutePlan** one helps to execute plans created by the first part. The **DestroyPlan** can destroy the plans. The remain is the part to realize function. The prefix "fw" and "bw" are the abbreviation of "forward" and "backward" respectively, which represent Fourier transform and inverse Fourier transform. The suffix "MT" is used to describe whether the multiple threads function are on.
 */

#ifndef FFT_H
#define FFT_H

#ifdef SINGLE_PRECISION
#include <fftw3.h>
#else
#include <fftw3.h>
#endif

#include "Logging.h"
#include "Precision.h"

#include <omp_compat.h>

#include "Config.h"
#include "Complex.h"

#include "Image.h"
#include "Volume.h"
#include "ImageFunctions.h"

/**
 * @brief This macro checks whether the source and destination of Fourier transform are both repaired.
 */
#define CHECK_SPACE_VALID(dst, /**< [in] the destination of Fourier transform */ \
                          src  /**< [in[ the source of Fourier transform */ \
                          ) \
{ \
    if (src == NULL) \
    { \
        REPORT_ERROR("FFT Needs Input Data."); \
        abort(); \
    } \
    if (dst == NULL) \
    { \
        REPORT_ERROR("FFT Needs Output Data."); \
        abort(); \
    } \
}

/**
 * @brief This macro extracts the pointers of the real and Fourier space memory array from image/volume for performing Fourier transform.
 */
#define FW_EXTRACT_P(obj /**< [in, out] the //TODO */ \
                     ) \
{ \
    obj.alloc(FT_SPACE); \
    _dstC = (TSFFTW_COMPLEX*)&obj[0]; \
    _srcR = &obj(0); \
    CHECK_SPACE_VALID(_dstC, _srcR); \
}

/**
 * @brief This macro destorys the plan for performing Fourier transform and assigns the pointers to NULL.
 */
#define FW_CLEAN_UP_MT \
{ \
    TSFFTW_destroy_plan(fwPlan); \
    fwPlan = NULL; \
    _dstC = NULL; \
    _srcR = NULL; \
}

/**
 * @brief This macro extracts the pointers of the real and Fourier space memory array from image/volume for performing inverse Fourier transform.
 */
#define BW_EXTRACT_P(obj /**< [in, out] the //TODO */ \
                     ) \
{ \
    obj.alloc(RL_SPACE); \
    _dstR = &obj(0); \
    _srcC = (TSFFTW_COMPLEX*)&obj[0]; \
    CHECK_SPACE_VALID(_dstR, _srcC); \
}

/**
 * @brief This macro destorys the plan for performing inverse Fourier transform and assigns the pointers to NULL.
 */
#define BW_CLEAN_UP_MT(obj /**< [in, out] the //TODO */) \
{ \
    TSFFTW_destroy_plan(bwPlan); \
    bwPlan = NULL; \
    _dstR = NULL; \
    _dstR = NULL; \
    _srcC = NULL; \
    obj.clearFT(); \
}

/**
 * This macro executes a function in real space and performs Fourier transform on the destination image/volume.
 */
#define R2C_RL(dst,     /**< [out] the destination image/volume */ \
               src,     /**< [in] the source image/volume */ \
               function /**< [in] the function to be executed */ \
              ) \
    do \
    { \
        function; \
        FFT fft; \
        fft.fw(dst); \
    } while (0)

/**
 * This macro performs inverse Fourier transform on the source image/volume and executes a function in real space.
 */
#define C2R_RL(dst,     /**< [out] the destination image/volume */ \
               src,     /**< [in] the source /image/volume */ \
               function /**< [in] the function to be executed */ \
              ) \
    do \
    { \
        FFT fft; \
        fft.bw(src); \
        function; \
    } while (0)

/**
 * This macro performs inverse Fourier transform on the source image/volume, executes a function in real space and performs Fourier transform on the destination image/volume.
 */
#define C2C_RL(dst,     /**< [out] the destination image/volume */ \
               src,     /**< [in] the source image/volume */ \
               function /**< [in] the function to be executed */ \
              ) \
    do \
    { \
        FFT fft; \
        fft.bw(src); \
        function; \
        fft.fw(dst); \
    } while (0)

/**
 * This macro performs Fourier transform on the source image (volume), executes a function in Fourier space and perform inverse Fourier transform on the destination image/volume.
 */
#define R2R_FT(dst,     /**< [out] the destination image/volume */ \
               src,     /**< [in] the source image/volume */ \
               function /**< [in] the function to be executed */ \
              ) \
    do \
    { \
        FFT fft; \
        fft.fw(src); \
        function; \
        fft.bw(dst); \
    } while (0)

/**
 * This macro performs Fourier transform on the source image/volume and excutes a function in Fourier space.
 */
#define R2C_FT(dst,     /**< [out] the destination image/volume */ \
               src,     /**< [in] the source image/volume */ \
               function /**< [in] the function to be executed */ \
              ) \
    do \
    { \
        FFT fft; \
        fft.fw(src); \
        function; \
    } while (0)

/**
 * This macro executes a function in Fourier space and performs an inverse Fourier transform on the destination image/volume.
 */
#define C2R_FT(dst,     /**< [out] the destination image/volume */ \
               src,     /**< [in] the source image/volume */ \
               function /**< [in] the function to be executed */ \
              ) \
    do \
    { \
        function; \
        FFT fft; \
        fft.bw(dst); \
    } while (0)

/**
 * @brief This class performs Fourier transform and inverse Fourier transform on image/volume. It can also generate a fast Fourier plan, sotre it and execute it.
 */
class FFT
{
    private:

        /**
         * @brief a pointer points to the source of the Fourier transform
         */
        RFLOAT* _srcR;

        /**
         * @brief a pointer points to the source of the inverse Fourier transform
         */
        TSFFTW_COMPLEX* _srcC;

        /**
         * @brief a pointer points to the destination of the inverse Fourier transform
         */
        RFLOAT* _dstR;

        /**
         * @brief a pointer points to the destination of the Fourier transform
         */
        TSFFTW_COMPLEX* _dstC;

        /**
         * @brief the plan of Fourier transform
         */
        TSFFTW_PLAN fwPlan;

        /**
         * @brief the plan of inverse Fourier transform
         */
        TSFFTW_PLAN bwPlan;

    public:

        /**
         * @brief default constructor.
         */
        FFT();

        /**
         * @brief default destructor.
         */
        ~FFT();


        /**
         * @brief This function performs Fourier transform on an image using multiple threads.
         */
        void fw(Image& img,                   /**< [in] the image to be transformed */
                const unsigned int nThread    /**< [in] the number of threads to be used */
                );

        /**
         * @brief This function performs inverse Fourier transform on an image using multiple threads.
         */
        void bw(Image& img,                   /**< [in] the image to be transformed */
                const unsigned int nThread    /**< [in] the number of threads to be used */
                );

        /**
         * @brief This function performs Fourier transform on a volume using multiple threads.
         */
        void fw(Volume& vol,                 /**< [in] the volume to be transformed */
                const unsigned int nThread   /**< [in] the number of threads to be used */
                );

        /**
         * @brief This function performs inverse Fourier transform on a volume using multiple threads.
         */
        void bw(Volume& vol,                 /**< [in] the volume to be transformed */
                const unsigned int nThread   /**< [in] the number of threads to be used */
                );

        /**
         * @brief This function creates a plan to perform Fourier transform on an image using multiple threads.
         */
        void fwCreatePlan(const int nCol,              /**< [in] number of columns of the image */
                          const int nRow,              /**< [in] number of rows of the image */
                          const unsigned int nThread   /**< [in] the number of threads to be used */
                          );
        /**
         * @brief This function creates a plan to perform Fourier transform on a volume using multiple threads.
         */
        void fwCreatePlan(const int nCol,              /**< [in] number of columns of the image */
                          const int nRow,              /**< [in] number of rows of the image */
                          const int nSlc,              /**< [in] number of slices of the image */
                          const unsigned int nThread   /**< [in] the number of threads to be used */
                          );
        /**
         * @brief This function creates a plan to perform inverse Fourier transform on an image using multiple threads.
         */
        void bwCreatePlan(const int nCol,              /**< [in] number of columns of the image */
                          const int nRow,              /**< [in] number of rows of the image */
                          const unsigned int nThread   /**< [in] the number of threads to be used */
                          );
        /**
         * @brief This function creates a plan to perform inverse Fourier transform on a volume using multiple threads.
         */
        void bwCreatePlan(const int nCol,              /**< [in] number of columns of the image */
                          const int nRow,              /**< [in] number of rows of the image */
                          const int nSlc,              /**< [in] number of slices of the image */
                          const unsigned int nThread   /**< [in] the number of threads to be used */
                          );
        /**
         * @brief This function executes the created plan that performs Fourier transform on an image using multiple threads.
         */
        void fwExecutePlan(Image& img                  /**< [in] the image to be Fourier transformed */);
        
        /**
         * @brief This function executes the created plan that performs Fourier transform on a volume using multiple threads.
         */        
        void fwExecutePlan(Volume& vol                 /**< [in] the volume to be Fourier transformed */); 
        
        /**
         * @brief This function executes the created plan that performs inverse Fourier transform on an image using multiple threads.
         */
        void bwExecutePlan(Image& img,                  /**< [in] the image to be inverse Fourier transformed */
                           const unsigned int nThread   /**< [in] the number of threads to be used */
                           );

        /**
         * @brief This function executes the created plan that performs inverse Fourier transform on a volume using multiple threads.
         */
        void bwExecutePlan(Volume& vol,                 /**< [in] the volume to be inverse Fourier transformed */
                           const unsigned int nThread   /**< [in] the number of threads to be used */
                           );

        /**
         * @brief This function destroys the created plan that performs Fourier transform on an image or volume.
         */
        void fwDestroyPlan();

        /**
         * @brief This function destroys the created plan that performs inverse Fourier transform on an image or volume.
         */
        void bwDestroyPlan();
};

#endif // FFT_H 
