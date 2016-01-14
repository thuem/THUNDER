/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <cmath>

#include <gsl/gsl_sf_trig.h>

#include "Complex.h"
#include "Functions.h"
#include "Enum.h"

void floor(int& x0, int& y0,
           const double x, const double y);
// calculate the floor values

void floor(int& x0, int& y0, int& z0,
           const double x, const double y, const double z);
// calculate the floor values

void floorDiff(int& x0,
               double& xd,
               const double x);

void floorDiff(int& x0, int& y0,
               double& xd, double& yd,
               const double x, const double y);
// calculate the floor values and the differences between floors values and
// input values

void floorDiff(int& x0, int& y0, int& z0,
               double& xd, double& yd, double& zd,
               const double x, const double y, const double z);
// calculate the floor values and the differences between floors values and
// input values

Complex linear(const Complex v0,
               const Complex v1,
               const double xd);
// calculate the linear interpolation

double linear(const double v0,
              const double v1,
              const double xd);
// calculate the linear interpolation

Complex biLinear(const Complex v00, const Complex v01,
                 const Complex v10, const Complex v11,
                 const double xd,
                 const double zd);
// calculate the bilinear interpolation

double biLinear(const double v00, const double v01,
                const double v10, const double v11,
                const double xd,
                const double zd);
// calculate the bilinear interpolation

Complex triLinear(const Complex v000, const Complex v001,
                  const Complex v010, const Complex v011,
                  const Complex v100, const Complex v101,
                  const Complex v110, const Complex v111,
                  const double xd,
                  const double yd,
                  const double zd);
// calculate the trilinear interpolation

double triLinear(const double v000, const double v001,
                 const double v010, const double v011,
                 const double v100, const double v101,
                 const double v110, const double v111,
                 const double xd,
                 const double yd,
                 const double zd);
// calculate the trilinear interpolation

void nearestWeight(double& w0,
                   double& w1,
                   const double xd);

void nearestWeightGrid(double& w0,
                       double& w1,
                       int& x0,
                       const double x);

void biNearestWeight(double& w00, double& w01,
                     double& w10, double& w11,
                     const double xd,
                     const double yd);

void biNearestWeightGrid(double& w00, double& w01,
                         double& w10, double& w11,
                         int& x0,
                         int& y0,
                         const double x,
                         const double y);

void triNearestWeight(double& w000, double& w001,
                      double& w010, double& w011,
                      double& w100, double& w101,
                      double& w110, double& w111,
                      const double xd,
                      const double yd,
                      const double zd);

void triNearestWeightGrid(double& w000, double& w001,
                          double& w010, double& w011,
                          double& w100, double& w101,
                          double& w110, double& w111,
                          int& x0,
                          int& y0,
                          int& z0,
                          const double x,
                          const double y,
                          const double z);

void linearWeight(double& w0,
                  double& w1,
                  const double xd);
// calculate the weights of linear interpolation

void linearWeightGrid(double& w0,
                      double& w1,
                      const int& x0,
                      const double x);
// calculate the weights of linear interpolation

void biLinearWeight(double& w00, double& w01,
                    double& w10, double& w11,
                    const double xd,
                    const double yd);
// calculate the weights of bilinear interpolation

void biLinearWeightGrid(double& w00, double& w01,
                        double& w10, double& w11,
                        int& x0,
                        int& y0,
                        const double x,
                        const double y);
// calculate the weights of bilinear interpolation

void triLinearWeight(double& w000, double& w001,
                     double& w010, double& w011,
                     double& w100, double& w101,
                     double& w110, double& w111,
                     const double xd,
                     const double yd,
                     const double zd);
// calculate the weights of trilinear interpolation

void triLinearWeightGrid(double& w000, double& w001,
                         double& w010, double& w011,
                         double& w100, double& w101,
                         double& w110, double& w111,
                         int& x0,
                         int& y0,
                         int& z0,
                         const double x,
                         const double y,
                         const double z);
// calculate the weights of trilinear interpolation

void sincWeight(double& w0,
                double& w1,
                const double xd);
// calculate the weights of sinc interpolation

void sincWeightGrid(double& w0,
                    double& w1,
                    int& x0,
                    const double x);
// calculate the weights of sinc interpolation

void biSincWeight(double& w00, double& w01,
                  double& w10, double& w11,
                  const double xd,
                  const double yd);
// calculate the weights of 2D sinc interpolation

void biSincWeightGrid(double& w00, double& w01,
                      double& w10, double& w11,
                      int& x0,
                      int& y0,
                      const double x,
                      const double y);
// calculate the weights of 2D sinc interpolation

void triSincWeight(double& w000, double& w001,
                   double& w010, double& w011,
                   double& w100, double& w101,
                   double& w110, double& w111,
                   const double xd,
                   const double yd,
                   const double zd);
// calculate the weights of 3D sinc interpolation

void triSincWeightGrid(double& w000, double& w001,
                       double& w010, double& w011,
                       double& w100, double& w101,
                       double& w110, double& w111,
                       int& x0,
                       int& y0,
                       int& z0,
                       const double x,
                       const double y,
                       const double z);
// calculate the weights of 3D sinc interpolation

void interpolationWeight2D(double& w00, double& w01,
                           double& w10, double& w11,
                           const double xd,
                           const double yd,
                           const Interpolation2DStyle style = linear2D);

void interpolationWeight2DGrid(double& w00, double& w01,
                               double& w10, double& w11,
                               int& x0,
                               int& y0,
                               const double x,
                               const double y,
                               const Interpolation2DStyle style = linear2D);

void interpolationWeight3D(double& w000, double& w001,
                           double& w010, double& w011,
                           double& w100, double& w101,
                           double& w110, double& w111,
                           const double xd, 
                           const double yd,
                           const double zd,
                           const Interpolation3DStyle style = linear3D);

void interpolationWeight3DGrid(double& w000, double& w001,
                               double& w010, double& w011,
                               double& w100, double& w101,
                               double& w110, double& w111,
                               int& x0,
                               int& y0,
                               int& z0,
                               const double x,
                               const double y,
                               const double z,
                               const Interpolation3DStyle style = linear3D);
#endif // INTERPOLATION_H
