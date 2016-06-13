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
#include <vector>
#include <iostream>

#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics.h>

#include "Random.h"
#include "FFT.h"
#include "Image.h"
#include "Volume.h"

using namespace std;

/**
 * This macro extracts the centre block out of a volume in real space.
 * @param dst the destination volume
 * @param src the source volume
 * @param ef the extraction factor (0 < ef <= 1)
 */
#define VOL_EXTRACT_RL(dst, src, ef) \
    VOL_EXTRACT(RL, dst, src, ef) 

/**
 * This macro extracts the centre block out of a volume in Fourier space.
 * @param dst the destination volume
 * @param src the source volume
 * @param ef the extraction factor (0 < ef <= 1)
 */
#define VOL_EXTRACT_FT(dst, src, ef) \
    VOL_EXTRACT(FT, dst, src, ef)

/**
 * This macro extracts the centre block out of a volume.
 * @param SP the space in which the extraction perfroms (RL: real space, FT:
 * Fourier space)
 * @param dst the destination volume
 * @param src the source volume
 * @param ef the extraction factor (0 < ef <= 1)
 */
#define VOL_EXTRACT(SP, dst, src, ef) \
    [](Volume& _dst, const Volume& _src, const double _ef) \
    { \
        _dst.alloc(AROUND(_ef * _src.nColRL()), \
                   AROUND(_ef * _src.nRowRL()), \
                   AROUND(_ef * _src.nSlcRL()), \
                   SP##_SPACE); \
        VOLUME_FOR_EACH_PIXEL_##SP(_dst) \
            _dst.set##SP(_src.get##SP(i, j, k), i, j, k); \
    }(dst, src, ef)

/**
 * This macro replaces the centre block of a volume with another volume in real
 * space.
 * @param dst the destination volume
 * @param src the source volume
 */
#define VOL_REPLACE_RL(dst, src) \
    VOL_REPLACE(RL, dst, src)

/**
 * This macro replaces the centre block of a volume with another volume in
 * Fourier space.
 * @param dst the destination volume
 * @param src the source volume
 */
#define VOL_REPLACE_FT(dst, src) \
    VOL_REPLACE(FT, dst, src)

/**
 * This macro replaces the centre block of a volume with another volume.
 * @param SP the space in which the extraction perfroms (RL: real space, FT:
 * Fourier space)
 * @param dst the destination volume
 * @param src the source volume
 */
#define VOL_REPLACE(SP, dst, src) \
    [](Volume& _dst, const Volume& _src) \
    { \
        VOLUME_FOR_EACH_PIXEL_##SP(_src) \
            _dst.set##SP(_src.get##SP(i, j, k), i, j, k); \
    }(dst, src)

/**
 * This macro pads a volumen in real space.
 * @param dst the destination volume
 * @param src the source volume
 * @param pf the padding factor
 */
#define VOL_PAD_RL(dst, src, pf) \
    VOL_PAD(RL, dst, src, pf)

/**
 * This macro pads a volumen in Fourier space.
 * @param dst the destination volume
 * @param src the source volume
 * @param pf the padding factor
 */
#define VOL_PAD_FT(dst, src, pf) \
    VOL_PAD(FT, dst, src, pf)

/**
 * This macro pads a volume.
 * @param SP the space in which the extraction perfroms (RL: real space, FT:
 * Fourier space)
 * @param dst the destination volume
 * @param src the source volume
 * @param pf the padding factor
 */
#define VOL_PAD(SP, dst, src, pf) \
    [](Volume& _dst, const Volume& _src, const int _pf) \
    { \
        _dst.alloc(_pf * _src.nColRL(), \
                   _pf * _src.nRowRL(), \
                   _pf * _src.nSlcRL(), \
                    SP##_SPACE); \
        SET_0_##SP(_dst); \
        VOLUME_FOR_EACH_PIXEL_##SP(_src) \
            _dst.set##SP(_src.get##SP(i, j, k), i, j, k); \
    }(dst, src, pf)

/**
 * This macro replaces a slice of a volume with an image given in real space.
 * @param dst the destination volume
 * @param src the source image
 * @param k the index of the slice
 */
#define SLC_REPLACE_RL(dst, src, k) \
    SLC_REPLACE(RL, dst, src, k)

/**
 * This macro replaces a slice of a volume with an image given in Fourier
 * space.
 * @param dst the destination volume
 * @param src the source image
 * @param k the index of the slice
 */
#define SLC_REPLACE_FT(dst, src, k) \
    SLC_REPLACE(FT, dst, src, k)

/**
 * This macro replaces a slice of a volume with an image given.
 * @param SP the space in which the extraction perfroms (RL: real space, FT:
 * Fourier space)
 * @param dst the destination volume
 * @param src the source image
 * @param k the index of the slice
 */
#define SLC_REPLACE(SP, dst, src, k) \
    [](Volume& _dst, const Image& _src, const int _k) \
    { \
        IMAGE_FOR_EACH_PIXEL_##SP(_src) \
            _dst.set##SP(_src.get##SP(i, j), i, j, _k); \
    }(dst, src, k)

/**
 * This macro extracts a slice out of a volume and stores it in an image in real
 * space.
 * @param dst the destination image
 * @param src the source volume
 * @param k the index of the slice
 */
#define SLC_EXTRACT_RL(dst, src, k) \
    SLC_EXTRACT(RL, dst, src, k)

/**
 * This macro extracts a slice out of a volume and stores it in an image in
 * Fourier space.
 * @param dst the destination image
 * @param src the source volume
 * @param k the index of the slice
 */
#define SLC_EXTRACT_FT(dst, src, k) \
    SLC_EXTRACT(FT, dst, src, k)

/**
 * This macro extracts a slice out of a volume and stores it in an image.
 * @param SP the space in which the extraction perfroms (RL: real space, FT:
 * Fourier space)
 * @param dst the destination image
 * @param src the source volume
 * @param k the index of the slice
 */
#define SLC_EXTRACT(SP, dst, src, k) \
    [](Image& _dst, const Volume& _src, const int _k) \
    { \
        IMAGE_FOR_EACH_PIXEL_##SP(_dst) \
            _dst.set##SP(_src.get##SP(i, j, _k), i, j); \
    }(dst, src, k)

/**
 * This macro translations an image with a given vector indicating by the number
 * of columns and the number of rows.
 * @param dst the destination image (Fourier space)
 * @param src the source image (Fourier space)
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const Image& src,
               const double nTransCol,
               const double nTransRow);

/**
 * This macro translations an image in a certain frequency threshold with a 
 * given vector indicating by the number of columns and the number of rows.
 * @param dst the destination image (Fourier space)
 * @param src the source image (Fourier space)
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 */
void translate(Image& dst,
               const Image& src,
               const double r,
               const double nTransCol,
               const double nTransRow);

/**
 * This function calculates the cross-correlation image of two images in a
 * certain region.
 * @param dst the destination image
 * @param a the image A
 * @param b the image B
 * @param maxX the maximum column translation
 * @param maxY the maximum row translation
 */
void crossCorrelation(Image& dst,
                      const Image& a,
                      const Image& b);

/**
 * This function calculates the most likely translation between two images using
 * max cross correlation method.
 * @param nTransCol number of columns for translation
 * @param nTransRow number of rows for translation
 * @param a Image A
 * @param b Image B
 * @param maxX the maximum column translation
 * @param maxY the maximum row translation
 */
void translate(int& nTranCol,
               int& nTranRow,
               const Image& a,
               const Image& b,
               const int maxX,
               const int maxY);

/**
 * This function calculates the mean and standard deviation of the background.
 * The background stands for the outer region beyond a certain radius.
 * @param mean the mean value
 * @param stddev the standard devation
 * @param src the image to be calculated
 * @param r the radius
 */
void bgMeanStddev(double& mean,
                  double& stddev,
                  const Image& src,
                  const double r);

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
 * @param img the image to be processed
 * @param wDust the factor of white dust
 * @param bDust the factor of black dust
 * @param r the radius
 */
void normalise(Image& img,
               const double wDust,
               const double bDust,
               const double r);

/**
 * This function extracts a sub-image from an image.
 * @param dst the destination image
 * @param src the source image
 * @param xOff the column shift
 * @param yOff the row shift
 */
void extract(Image& dst,
             const Image& src,
             const int xOff,
             const int yOff);

#endif // IMAGE_FUNCTIONS_H
