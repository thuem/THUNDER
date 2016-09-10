/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef MASK_H
#define MASK_H

#include <cmath>
#include <functional>

#include "Random.h"
#include "Volume.h"
#include "Interpolation.h"

#define VOLUME_FOR_EACH_PIXEL_IN_GRID(a) \
    for (int z = -a; z < a; z++) \
        for (int y = -a; y < a; y++) \
            for (int x = -a; x < a; x++)

using namespace std;

/**
 * This function calculates the average value of pixels outside the circle of
 * a certain radius.
 *
 * @param img the image to be calculated
 * @param r   radius of the circle
 * @param ew  edge width of the circle
 */
double background(const Image& img,
                  const double r,
                  const double ew);

/**
 * This function calculates the average value of pixels not belonging to a
 * certain layer.
 *
 * @param img   the image to be calcualted
 * @param alpha the layer
 */
double background(const Image& img,
                  const Image& alpha); 

/**
 * This function calculates the average value of pixels outside the sphere of
 * a certain radius.
 *
 * @param vol the volume to be calculated
 * @param r   radius of the sphere
 * @param ew  edge width of the sphere
 */
double background(const Volume& vol,
                  const double r,
                  const double ew);

/**
 * This function calculates the average value of pixels not belonging to a
 * certain layer.
 *
 * @param vol   the volume to be calculated
 * @param alpha the layer
 */
double background(const Volume& vol,
                  const Volume& alpha);

/**
 * This function applys a soft mask on an image. The soft mask is calculated
 * from the source image with a certain radius and edge width.
 *
 * @param dst destination image
 * @param src source image
 * @param r   radius of the circle
 * @param ew  edge width of the cirlce
 */
void softMask(Image& dst,
              const Image& src,
              const double r,
              const double ew);

/**
 * This function applys a soft mask on an image. The soft mask is calculated
 * from the source image with a certain layer.
 *
 * @param dst   destination image
 * @param src   source image
 * @param alpha the layer
 */
void softMask(Image& dst,
              const Image& src,
              const Image& alpha);

/**
 * This function applys a soft mask on an image. The background will be
 * generated with given mean value and standard deviation.
 *
 * @param dst    destination image
 * @param src    source image
 * @param r      radius of the circle
 * @param ew     edge width of the cirlce
 * @param bgMean the mean value of the background
 * @param bgStd  the standard devation of the background
 */
void softMask(Image& dst,
              const Image& src,
              const double r,
              const double ew,
              const double bgMean,
              const double bgStd);

/**
 * This function applys a soft mask on a volume. The soft mask is calculated
 * from the source volume with a certain radius and edge width.
 *
 * @param dst destination volume
 * @param src source volume
 * @param r   radius of the sphere
 * @param ew  edge width of the sphere
 */
void softMask(Volume& dst,
              const Volume& src,
              const double r,
              const double ew);

/**
 * This function applys a soft mask on a volume. The soft mask is calculated
 * from the source image with a certain layer.
 *
 * @param dst   destination volume
 * @param src   source volume
 * @param alpha the layer
 */
void softMask(Volume& dst,
              const Volume& src,
              const Volume& alpha);

void softMask(Volume& dst,
              const Volume& src,
              const Volume& alpha,
              const double bg);

/**
 * This function generates a mask on a volume. The standard for generate mask is
 * that if the density of a voxel is larger than a threshold, the voxel of the
 * layer will be set to 1, otherwise it will be set to 0.
 *
 * @param dst destination volume
 * @param src source volume
 * @param dt  the density threshold factor (typyical value, 10)
 */
void genMask(Volume& dst,
             const Volume& src,
             const double dt,
             const double r);

/**
 *
 *
 * @param ext the length of extending in pixel (typical value, 3)
 */
void genMask(Volume& dst,
             const Volume& src,
             const double dt,
             const double ext,
             const double r);

/**
 *
 *
 * @param ew the edge width of masking (typical value, 6)
 */
void genMask(Volume& dst,
             const Volume& src,
             const double dt,
             const double ext,
             const double ew,
             const double r);

#endif // MASK_H
