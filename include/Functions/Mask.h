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

#include "Volume.h"
#include "Interpolation.h"

/**
 * This function calculates the average value of pixels outside the circle of
 * a certain radius.
 * @param img the image to be calculated
 * @param r radius of the circle
 * @param ew edge width of the circle
 */
double background(const Image& img,
                  const double r,
                  const double ew);

/**
 * This function calculates the average value of pixels not belonging to a
 * certain layer.
 * @param img the image to be calcualted
 * @param alpha the layer
 */
double background(const Image& img,
                  const Image& alpha); 

/**
 * This function calculates the average value of pixels outside the sphere of
 * a certain radius.
 * @param vol the volume to be calculated
 * @param r radius of the sphere
 * @param ew edge width of the sphere
 */
double background(const Volume& vol,
                  const double r,
                  const double ew);

/**
 * This function calculates the average value of pixels not belonging to a
 * certain layer.
 * @param vol the volume to be calculated
 * @param alpha the layer
 */
double background(const Volume& vol,
                  const Volume& alpha);

/**
 * This function applys a soft mask on an image. The soft mask is calculated
 * from the source image with a certain radius and edge width.
 * @param dst destination image
 * @param src source image
 * @param radius of the circle
 * @param ew edge width of the cirlce
 */
void softMask(Image& dst,
              const Image& src,
              const double r,
              const double ew);

/**
 * This function applys a soft mask on an image. The soft mask is calculated
 * from the source image with a certain layer.
 * @param dst destination image
 * @param src source image
 * @param alpha the layer
 */
void softMask(Image& dst,
              const Image& src,
              const Image& alpha);

/**
 * This function applys a soft mask on a volume. The soft mask is calculated
 * from the source volume with a certain radius and edge width.
 * @param dst destination volume
 * @param src source volume
 * @param radius of the sphere
 * @param ew edge width of the sphere
 */
void softMask(Volume& dst,
              const Volume& src,
              const double r,
              const double ew);

/**
 * This function applys a soft mask on a volume. The soft mask is calculated
 * from the source image with a certain layer.
 * @param dst destination volume
 * @param src source volume
 * @param alpha the layer
 */
void softMask(Volume& dst,
              const Volume& src,
              const Volume& alpha);

/***
void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold);

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold,
                  const double extend);

void generateMask(Volume& dst,
                  const Volume& src,
                  const double densityThreshold,
                  const double extend,
                  const double ew);
                  ***/

#endif // MASK_H
