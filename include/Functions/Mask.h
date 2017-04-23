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
#include <numeric>
#include <functional>
#include <algorithm>

#include "omp_compat.h"

#include "Random.h"
#include "Volume.h"
#include "Macro.h"

/**
 * recommended extension
 */
#define GEN_MASK_EXT 4

#define GEN_MASK_INIT_STEP 0.2

#define GEN_MASK_GAP 0.05

/**
 * This function calculates the number of pixels inside the circle of a certain radius.
 *
 * @param r  radius of the circle
 * @param ew edge width of the circle 
 */
double nPixel(const double r,
              const double ew);

/**
 * This function calculates the number of voxels inside the cirlce of a certain radius.
 *
 * @param r  radius of the circle
 * @param ew edge width of the circle
 */
double nVoxel(const double r,
              const double ew);

double regionMean(const Image& img,
                  const int r);

double regionMean(const Volume& vol,
                  const int r);

double regionMean(const Image& img,
                  const double rU,
                  const double rL);

double regionMean(const Volume& vol,
                  const double rU,
                  const double rL);

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

double background(const Volume& vol,
                  const double rU,
                  const double rL,
                  const double ew);

void softMask(Image& mask,
              const double r,
              const double ew);

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
 * from the source image with a certain radius and edge width.
 *
 * @param dst destination image
 * @param src source image
 * @param r   radius of the circle
 * @param ew  edge width of the cirlce
 * @param bg  background
 */
void softMask(Image& dst,
              const Image& src,
              const double r,
              const double ew,
              const double bg);

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
 * This function applys a soft mask on an image.
 *
 * @param dst   destination image
 * @param src   source image
 * @param alpha the layer
 * @param bg    background
 */
void softMask(Image& dst,
              const Image& src,
              const Image& alpha,
              const double bg);

/**
 * This function applys a soft mask on an image. The background will be
 * generated with given mean value and standard deviation.
 *
 * @param dst    destination image
 * @param src    source image
 * @param alpha  the layer
 * @param bgMean mean value of background
 * @param bgStd  standard deviation of background
 */
void softMask(Image& dst,
              const Image& src,
              const Image& alpha,
              const double bgMean,
              const double bgStd);

void softMask(Volume& mask,
              const double r,
              const double ew);

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

void softMask(Volume& dst,
              const Volume& src,
              const double r,
              const double ew,
              const double bg);

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

/**
 * This function applys a softmask on a volume. The soft mask is calculated from
 * the source image with a certain layer. The background is assigned with a
 * parameter.
 *
 * @param dst   destination volume
 * @param src   source volume
 * @param alpha layer
 * @param bg    background
 */
void softMask(Volume& dst,
              const Volume& src,
              const Volume& alpha,
              const double bg);

void regionBgSoftMask(Image& dst,
                      const Image& src,
                      const double r,
                      const double ew,
                      const double rU,
                      const double rL);

void regionBgSoftMask(Volume& dst,
                      const Volume& src,
                      const double r,
                      const double ew,
                      const double rU,
                      const double rL);

void removeIsolatedPoint(Volume& vol);

void extMask(Volume& vol,
             const double ext);

void softEdge(Volume& vol,
              const double ew);

void genMask(Volume& dst,
             const Volume& src,
             const double thres);

void genMask(Volume& dst,
             const Volume& src,
             const double thres,
             const double ext);

void genMask(Volume& dst,
             const Volume& src,
             const double thres,
             const double ext,
             const double ew);

/**
 * This function generates a mask on a volume. The standard for generating mask is
 * that if the density of a voxel is larger than a threshold.
 *
 * @param dst destination volume
 * @param src source volume
 * @param dt  the density threshold factor (typyical value, 10)
 * @param r   the radius of the ball containing information
 */
void autoMask(Volume& dst,
              const Volume& src,
              const double r);

/**
 * This function generates a mask on a volume. The mask is generated in the
 * following steps:
 * 1. Generate the backbone of the mask which containing only 0 and 1. The
 * standard for generating the backbone is that if the density of a voxel is
 * larger than a threshold, the voxel of the layer will be set to 1, otherwise
 * it will be set to 0. The threshold is calculated by the mean and stadard
 * deviation of the background (3/4 radius to radius part).
 * 2. Extend the backbone of the mask. For a voxel of the extended backbone, if
 * there is a voxel in the original backone is 1, and the distance between them
 * is smaller than a certain parameter, it will be assigned to 1. Meanwhile,
 * when the extend paramter is smaller than 0, this step becomes shrinking the
 * backbone of the mask. For a voxel of the shrinked backbone, if there is a
 * voxel in the original backbone is 0, and the distance between them is smaller
 * than a certain paramter, it will be assigned to 0.
 *
 * @param dst destination volume
 * @param src source volume
 * @param dt  the density threshold factor (typical value, 10)
 * @param ext the length of extending in pixel (typical value, 3)
 * @param r   the radius of the ball containing information
 */
void autoMask(Volume& dst,
              const Volume& src,
              const double ext,
              const double r);

/**
 * This function generates a mask on a volume. The mask is generated in the
 * following steps:
 * 1. Generate the backbone of the mask which containing only 0 and 1. The
 * standard for generating the backbone is that if the density of a voxel is
 * larger than a threshold, the voxel of the layer will be set to 1, otherwise
 * it will be set to 0. The threshold is calculated by the mean and stadard
 * deviation of the background (0.85 radius to radius part).
 * 2. Extend the backbone of the mask. For a voxel of the extended backbone, if
 * there is a voxel in the original backone is 1, and the distance between them
 * is smaller than a certain parameter, it will be assigned to 1. Meanwhile,
 * when the extend paramter is smaller than 0, this step becomes shrinking the
 * backbone of the mask. For a voxel of the shrinked backbone, if there is a
 * voxel in the original backbone is 0, and the distance between them is smaller
 * than a certain paramter, it will be assigned to 0.
 * 3. Make the mask have a soft cosine edge.
 *
 * @param dst destination volume
 * @param src source volume
 * @param dt  the density threshold factor (typical value, 10)
 * @param ext the length of extending in pixel (typical value, 3)
 * @param ew  the edge width of masking (typical value, 6)
 * @param r   the radius of the ball containing information
 */
void autoMask(Volume& dst,
              const Volume& src,
              const double ext,
              const double ew,
              const double r);

#endif // MASK_H
