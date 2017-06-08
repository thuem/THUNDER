/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef IMAGE_FUNCTIONS_H
#define IMAGE_FUNCTIONS_H

#include <cmath>

#include <iostream>

#include <omp_compat.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics.h>

#include "Random.h"
#include "FFT.h"
#include "Image.h"
#include "Volume.h"

/**
 * This function extracts the centre block out of a volume in real space.
 *
 * @param dst the destination volume
 * @param src the source volume
 * @param ef  the extraction factor (0 < ef <= 1)
 */
inline void VOL_EXTRACT_RL(Volume& dst,
                           const Volume& src,
                           const double ef) 
{ 
    dst.alloc(AROUND(ef * src.nColRL()), 
              AROUND(ef * src.nRowRL()), 
              AROUND(ef * src.nSlcRL()), 
              RL_SPACE); 

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(dst) 
        dst.setRL(src.getRL(i, j, k), i, j, k); 
}
 
/**
 * This function extracts the centre block out of a volume in Fourier space.
 *
 * @param dst the destination volume
 * @param src the source volume
 * @param ef  the extraction factor (0 < ef <= 1)
 */
inline void VOL_EXTRACT_FT(Volume& dst,
                           const Volume& src,
                           const double ef) 
{ 
    dst.alloc(AROUND(ef * src.nColRL()), 
              AROUND(ef * src.nRowRL()), 
              AROUND(ef * src.nSlcRL()), 
              FT_SPACE); 

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_FT(dst)
        dst.setFT(src.getFT(i, j, k), i, j, k); 
}

/**
 * This function replaces the centre block of a volume with another volume in
 * real space.
 *
 * @param dst the destination volume
 * @param src the source volume
 */
inline void VOL_REPLACE_RL(Volume& dst, const Volume& src) 
{ 
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(src) 
        dst.setRL(src.getRL(i, j, k), i, j, k); 
}

/**
 * This function replaces the centre block of a volume with another volume in
 * Fourier space.
 *
 * @param dst the destination volume
 * @param src the source volume
 */
inline void VOL_REPLACE_FT(Volume& dst, const Volume& src) 
{ 
    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_FT(src) 
        dst.setFT(src.getFT(i, j, k), i, j, k); 
}

inline void IMG_PAD_RL(Image& dst,
                       const Image& src,
                       const int pf)
{
    dst.alloc(pf * src.nColRL(),
              pf * src.nRowRL(),
              RL_SPACE);

    SET_0_RL(dst);

    IMAGE_FOR_EACH_PIXEL_RL(src)
        dst.setRL(src.getRL(i, j), i, j);
}

/**
 * This function pads a volume in real space.
 *
 * @param dst the destination volume
 * @param src the source volume
 * @param pf  the padding factor
 */
inline void VOL_PAD_RL(Volume& dst,
                       const Volume& src,
                       const int pf) 
{ 
    dst.alloc(pf * src.nColRL(), 
              pf * src.nRowRL(), 
              pf * src.nSlcRL(), 
              RL_SPACE); 

    #pragma omp parallel for
    SET_0_RL(dst); 

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(src) 
        dst.setRL(src.getRL(i, j, k), i, j, k); 
}

inline void IMG_PAD_FT(Image& dst,
                       const Image& src,
                       const int pf)
{
    dst.alloc(pf * src.nColRL(),
              pf * src.nRowRL(),
              FT_SPACE);

    SET_0_FT(dst);

    IMAGE_FOR_EACH_PIXEL_FT(src)
        dst.setFT(src.getFT(i, j), i, j);
}

/**
 * This function pads a volume in Fourier space.
 *
 * @param dst the destination volume
 * @param src the source volume
 * @param pf  the padding factor
 */
inline void VOL_PAD_FT(Volume& dst,
                       const Volume& src,
                       const int pf) 
{ 
    dst.alloc(pf * src.nColRL(), 
              pf * src.nRowRL(), 
              pf * src.nSlcRL(), 
              FT_SPACE); 

    #pragma omp parallel for
    SET_0_FT(dst); 

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_FT(src) 
        dst.setFT(src.getFT(i, j, k), i, j, k); 
}

/**
 * This function replaces a slice of a volume with an image given in real space.
 *
 * @param dst the destination volume
 * @param src the source image
 * @param k   the index of the slice
 */
inline void SLC_REPLACE_RL(Volume& dst, const Image& src, const int k) 
{ 
    IMAGE_FOR_EACH_PIXEL_RL(src) 
        dst.setRL(src.getRL(i, j), i, j, k); 
}

/**
 * This function replaces a slice of a volume with an image given in Fourier
 * space.
 *
 * @param dst the destination volume
 * @param src the source image
 * @param k   the index of the slice
 */
inline void SLC_REPLACE_FT(Volume& dst, const Image& src, const int k) 
{ 
    IMAGE_FOR_EACH_PIXEL_FT(src) 
        dst.setFT(src.getFT(i, j), i, j, k); 
}

/**
 * This macro extracts a slice out of a volume and stores it in an image in
 * real space.
 *
 * @param dst the destination image
 * @param src the source volume
 * @param k   the index of the slice
 */
inline void SLC_EXTRACT_RL(Image& dst, const Volume& src, const int k) 
{ 
    IMAGE_FOR_EACH_PIXEL_RL(dst) 
        dst.setRL(src.getRL(i, j, k), i, j); 
}

/**
 * This macro extracts a slice out of a volume and stores it in an image in
 * Fourier space.
 *
 * @param dst the destination image
 * @param src the source volume
 * @param k   the index of the slice
 */
inline void SLC_EXTRACT_FT(Image& dst, const Volume& src, const int k) 
{ 
    IMAGE_FOR_EACH_PIXEL_FT(dst) 
        dst.setFT(src.getFT(i, j, k), i, j); 
}

void mul(Image& dst,
         const Image& a,
         const Image& b,
         const int r);

void mul(Image& dst,
         const Image& a,
         const Image& b,
         const int* iPxl,
         const int nPxl);

/**
 * This function generates a "translation image" with a given vector indicating
 * the number of columns and the number of rows. An image can be tranlated by
 * this vector, just by multiplying this "translation image" in Fourier space.
 *
 * @param dst       the translation image
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const double nTransCol,
               const double nTransRow);

/**
 * This function generates a "translation image" with a given vector indicating
 * the number of columns and the number of rows using mutliple threads. An 
 * image can be tranlated by this vector, just by multiplying this "translation 
 * image" in Fourier space.
 *
 * @param dst       the translation image
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translateMT(Image& dst,
                 const double nTransCol,
                 const double nTransRow);

/**
 * This function generates a "translation image" in a certain frequency
 * threshold with a given vector indicating the number of columns and the number
 * of rows. An image can be tranlated by this vector, just by multiplying this
 * "translation image" in Fourier space.
 *
 * @param dst       the translation image
 * @param r         the resolution threshold
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const double r,
               const double nTransCol,
               const double nTransRow);

/**
 * This function generates a "translation image" in a certain frequency
 * threshold with a given vector indicating the number of columns and the number
 * of rows using multiple threads. An image can be tranlated by this vector, just
 * by multiplying this "translation image" in Fourier space.
 *
 * @param dst       the translation image
 * @param r         the resolution threshold
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translateMT(Image& dst,
                 const double r,
                 const double nTransCol,
                 const double nTransRow);

void translate(Image& dst,
               const double nTransCol,
               const double nTransRow,
               const int* iCol,
               const int* iRow,
               const int* iPxl,
               const int nPxl);

void translateMT(Image& dst,
                 const double nTransCol,
                 const double nTransRow,
                 const int* iCol,
                 const int* iRow,
                 const int* iPxl,
                 const int nPxl);

void translate(Complex* dst,
               const double nTransCol,
               const double nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl);

void translateMT(Complex* dst,
                 const double nTransCol,
                 const double nTransRow,
                 const int nCol,
                 const int nRow,
                 const int* iCol,
                 const int* iRow,
                 const int nPxl);

/**
 * This function translations an image with a given vector indicating by the number
 * of columns and the number of rows.
 *
 * @param dst       the destination image (Fourier space)
 * @param src       the source image (Fourier space)
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow);

/**
 * This function translations an image with a given vector indicating by the number
 * of columns and the number of rows using multiple threads.
 *
 * @param dst       the destination image (Fourier space)
 * @param src       the source image (Fourier space)
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translateMT(Image& dst,
                 const Image& src,
                 const double nTransCol,
                 const double nTransRow);

/**
 * This function translations an image in a certain frequency threshold with a 
 * given vector indicating by the number of columns and the number of rows.
 *
 * @param dst       the destination image (Fourier space)
 * @param src       the source image (Fourier space)
 * @param r         the resolution threshold
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const Image& src,
               const double r,
               const double nTransCol,
               const double nTransRow);

/**
 * This function translations an image in a certain frequency threshold with a 
 * given vector indicating by the number of columns and the number of rows using
 * multiple threads.
 *
 * @param dst       the destination image (Fourier space)
 * @param src       the source image (Fourier space)
 * @param r         the resolution threshold
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translateMT(Image& dst,
                 const Image& src,
                 const double r,
                 const double nTransCol,
                 const double nTransRow);

void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow,
               const int* iCol,
               const int* iRow,
               const int* iPxl,
               const int nPxl);

void translateMT(Image& dst,
                 const Image& src,
                 const double nTransCol,
                 const double nTransRow,
                 const int* iCol,
                 const int* iRow,
                 const int* iPxl,
                 const int nPxl);

void translate(Complex* dst,
               const Complex* src,
               const double nTransCol,
               const double nTransRow,
               const int nCol,
               const int nRow,
               const int* iCol,
               const int* iRow,
               const int nPxl);


void translateMT(Complex* dst,
                 const Complex* src,
                 const double nTransCol,
                 const double nTransRow,
                 const int nCol,
                 const int nRow,
                 const int* iCol,
                 const int* iRow,
                 const int nPxl);

/**
 * This function calculates the cross-correlation image of two images in a
 * certain region.
 * @param dst the destination image
 * @param a the image A
 * @param b the image B
 * @param r the radius of the frequency
 */
void crossCorrelation(Image& dst,
                      const Image& a,
                      const Image& b,
                      const double r);

/**
 * This function calculates the most likely translation between two images using
 * max cross correlation method.
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 * @param a Image A
 * @param b Image B
 * @param r the radius of the frequency
 * @param maxX the maximum column translation
 * @param maxY the maximum row translation
 */
void translate(int& nTranCol,
               int& nTranRow,
               const Image& a,
               const Image& b,
               const double r,
               const int maxX,
               const int maxY);

/**
 * This function calculates the standard deviation of an image given when the
 * mean value is given.
 *
 * @param mean the mean value
 * @param the image to be calculated
 */
double stddev(const double mean,
              const Image& src);

/**
 * This function calculates the mean and standard deviation of an image.
 *
 * @param mean the mean value
 * @param stddev the standard devation
 * @param src the image to be calculated
 */
void meanStddev(double& mean,
                double& stddev,
                const Image& src);

double centreStddev(const double mean,
                    const Image& src,
                    const double r);

void centreMeanStddev(double& mean,
                      double& stddev,
                      const Image& src,
                      const double r);

/**
 * This function calculates the standard deviation of the background when the
 * mean value is given.
 *
 * @param mean the mean value
 * @param src the image to be calculated
 * @param r the radius
 */
double bgStddev(const double mean,
                const Image& src,
                const double r);

double bgSttdev(const double mean,
                const Volume& src,
                const double r);

/**
 * This function calculates the mean and standard deviation of the background.
 * The background stands for the outer region beyond a certain radius.
 *
 * @param mean the mean value
 * @param stddev the standard devation
 * @param src the image to be calculated
 * @param r the radius
 */
void bgMeanStddev(double& mean,
                  double& stddev,
                  const Image& src,
                  const double r);

void bgMeanStddev(double& mean,
                  double& stddev,
                  const Volume& src,
                  const double r);

void bgMeanStddev(double& mean,
                  double& stddev,
                  const Volume& src,
                  const double rU,
                  const double rL);


/**
 * This function removes white and black dust from the image. Any value out of
 * the range (mean - bDust * stddev, mean + wDust * stddev) will be replaced by
 * a draw from N(mean, stddev).
 * @param img the image to be processed
 * @param wDust the factor of white dust
 * @param bDust the factor of black dust
 * @param mean the mean value
 * @param stddev the standard deviation
 */
void removeDust(Image& img,
                const double wDust,
                const double bDust,
                const double mean,
                const double stddev);

/**
 * This fucntion normalises the image according to the mean and stddev of the
 * background dust points are removed according to wDust and bDust.
 *
 * @param img   the image to be processed
 * @param wDust the factor of white dust
 * @param bDust the factor of black dust
 * @param r     the radius
 */
void normalise(Image& img,
               const double wDust,
               const double bDust,
               const double r);

/**
 * This function extracts a sub-image from an image.
 *
 * @param dst  the destination image
 * @param src  the source image
 * @param xOff the column shift
 * @param yOff the row shift
 */
void extract(Image& dst,
             const Image& src,
             const int xOff,
             const int yOff);

/**
 * This function performs binning on an image.
 *
 * @param dst the destination image
 * @param src the source image
 * @param bf  the binning factor
 */
void binning(Image& dst,
             const Image& src,
             const int bf);

#endif // IMAGE_FUNCTIONS_H
