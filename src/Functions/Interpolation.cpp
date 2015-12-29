/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency: 
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Interpolation.h"

void floor(int& x0,
           int& y0,
           const double x,
           const double y)
{
    x0 = floor(x);
    y0 = floor(y);
}

void floor(int& x0,
           int& y0,
           int& z0,
           const double x,
           const double y,
           const double z)
{
    floor(x0, y0, x, y);
    z0 = floor(z);
}

void floorDiff(int& x0,
               double& xd,
               const double x)
{
    x0 = floor(x);

    xd = x - x0;
}

void floorDiff(int& x0,
               int& y0,
               double& xd,
               double& yd,
               const double x,
               const double y)
{
    floor(x0, y0, x, y);

    xd = x - x0;
    yd = y - y0;
}

void floorDiff(int& x0,
               int& y0,
               int& z0,
               double& xd,
               double& yd,
               double& zd,
               const double x,
               const double y,
               const double z)
{
    floor(x0, y0, z0, x, y, z);

    xd = x - x0;
    yd = y - y0;
    zd = z - z0;
}

Complex linear(const Complex v0,
               const Complex v1,
               const double xd)
{
    return v0 * (1 - xd) + v1 * xd;
}

double linear(const double v0,
              const double v1,
              const double xd)
{
    return v0 * (1 - xd) + v1 * xd;
}

Complex biLinear(const Complex v00, const Complex v01,
                 const Complex v10, const Complex v11, 
                 const double xd,
                 const double yd)
{
    // interpolate in y axis
    Complex v0 = linear(v00, v01, yd);
    Complex v1 = linear(v10, v11, yd);

    // interpolate in x axis
    return linear(v0, v1, xd);
}

double biLinear(const double v00, const double v01,
                const double v10, const double v11, 
                const double xd,
                const double yd)
{
    // interpolate in y axis
    double v0 = linear(v00, v01, yd);
    double v1 = linear(v10, v11, yd);

    // interpolate in x axis
    return linear(v0, v1, xd);
}

Complex triLinear(const Complex v000, const Complex v001,
                  const Complex v010, const Complex v011,
                  const Complex v100, const Complex v101,
                  const Complex v110, const Complex v111,
                  const double xd,
                  const double yd,
                  const double zd)
{
    // interpolate in z axis
    Complex v00 = linear(v000, v001, zd);
    Complex v01 = linear(v010, v011, zd);
    Complex v10 = linear(v100, v101, zd);
    Complex v11 = linear(v110, v111, zd);

    return biLinear(v00, v01, v10, v11, xd, yd);
}

double triLinear(const double v000, const double v001,
                 const double v010, const double v011,
                 const double v100, const double v101,
                 const double v110, const double v111,
                 const double xd,
                 const double yd,
                 const double zd)
{
    // interpolate in z axis
    double v00 = linear(v000, v001, zd);
    double v01 = linear(v010, v011, zd);
    double v10 = linear(v100, v101, zd);
    double v11 = linear(v110, v111, zd);

    return biLinear(v00, v01, v10, v11, xd, yd);
}

void nearestWeight(double& w0,
                   double& w1,
                   const double xd)
{
    w0 = 0;
    w1 = 0;

    (xd < 0.5) ? w0++ : w1++;
}

void nearestWeightGrid(double& w0,
                       double& w1,
                       int& x0,
                       const double x)
{
    double xd;
    floorDiff(x0, xd, x);

    nearestWeight(w0, w1, xd);
}

void biNearestWeight(double& w00, double& w01,
                     double& w10, double& w11,
                     const double xd,
                     const double yd)
{
    double vx0, vx1;
    nearestWeight(vx0, vx1, xd);
    double vy0, vy1;
    nearestWeight(vy0, vy1, yd);

    w00 = vx0 * vy0;
    w01 = vx0 * vy1;
    w10 = vx1 * vy0;
    w11 = vx1 * vy1;
}

void biNearestWeightGrid(double& w00, double& w01,
                         double& w10, double& w11,
                         int& x0,
                         int& y0,
                         const double x,
                         const double y)
{
    double xd, yd;
    floorDiff(x0, y0, xd, yd, x, y);

    biNearestWeight(w00, w01, w10, w11, xd, yd);
}

void triNearestWeight(double& w000, double& w001,
                      double& w010, double& w011,
                      double& w100, double& w101,
                      double& w110, double& w111,
                      const double xd,
                      const double yd,
                      const double zd)
{
    double vx0, vx1;
    nearestWeight(vx0, vx1, xd);
    double vy0, vy1;
    nearestWeight(vy0, vy1, yd);
    double vz0, vz1;
    nearestWeight(vz0, vz1, zd);

    w000 = vx0 * vy0 * vz0;
    w001 = vx0 * vy0 * vz1;
    w010 = vx0 * vy1 * vz0;
    w011 = vx0 * vy1 * vz1;
    w100 = vx1 * vy0 * vz0;
    w101 = vx1 * vy0 * vz1;
    w110 = vx1 * vy1 * vz0;
    w111 = vx1 * vy1 * vz1;
}

void triNearestWeightGrid(double& w000, double& w001,
                          double& w010, double& w011,
                          double& w100, double& w101,
                          double& w110, double& w111,
                          int& x0,
                          int& y0,
                          int& z0,
                          const double x,
                          const double y,
                          const double z)
{
    double xd, yd, zd;
    floorDiff(x0, y0, z0, xd, yd, zd, x, y, z);

    triNearestWeight(w000, w001, w010, w011,
                     w100, w101, w110, w111,
                     xd, yd, zd);
}

void linearWeight(double& w0,
                  double& w1,
                  const double xd)
{
    w0 = 1 - xd;
    w1 = xd;
}

void linearWeightGrid(double& w0,
                      double& w1,
                      int& x0, 
                      const double x)
{
    x0 = floor(x);
    double xd = x - x0;
    linearWeight(w0, w1, xd);
}

void biLinearWeight(double& w00, double& w01,
                    double& w10, double& w11,
                    const double xd,
                    const double yd)
{
    w00 = (1 - xd) * (1 - yd);
    w01 = (1 - xd) * yd;
    w10 = xd * (1 - yd);
    w11 = xd * yd;
}

void biLinearWeightGrid(double& w00, double& w01,
                        double& w10, double& w11,
                        int& x0,
                        int& y0,
                        const double x,
                        const double y)
{
    double xd, yd;
    floorDiff(x0, y0, xd, yd, x, y);

    biLinearWeight(w00, w01, w10, w11, xd, yd);
}

void triLinearWeight(double& w000, double& w001,
                     double& w010, double& w011,
                     double& w100, double& w101,
                     double& w110, double& w111,
                     const double xd,
                     const double yd,
                     const double zd)
{
    double w00, w01, w10, w11;

    // in y and z axis
    biLinearWeight(w00, w01, w10, w11, yd, zd);

    // in x axis
    w000 = (1 - xd) * w00;
    w001 = (1 - xd) * w01;
    w010 = (1 - xd) * w10;
    w011 = (1 - xd) * w11;
    w100 = xd * w00;
    w101 = xd * w01;
    w110 = xd * w10;
    w111 = xd * w11;
}

void triLinearWeightGrid(double& w000, double& w001,
                         double& w010, double& w011,
                         double& w100, double& w101,
                         double& w110, double& w111,
                         int& x0,
                         int& y0,
                         int& z0,
                         const double x,
                         const double y,
                         const double z)
{
    double xd, yd, zd;
    floorDiff(x0, y0, z0, xd, yd, zd, x, y, z);

    triLinearWeight(w000, w001, w010, w011,
                    w100, w101, w110, w111,
                    xd, yd, zd);
}

void sincWeight(double& w0,
                double& w1,
                const double xd)
{
    w0 = gsl_sf_sinc(xd);
    w1 = gsl_sf_sinc(1 - xd);
}

void sincWeightGrid(double& w0,
                    double& w1,
                    int& x0,
                    const double x)
{
    x0 = floor(x);
    double xd = x - x0;
    sincWeight(w0, w1, xd);
}

void biSincWeight(double& w00, double& w01,
                  double& w10, double& w11,
                  const double xd,
                  const double yd)
{
    // in x axis
    double xw0, xw1;
    sincWeight(xw0, xw1, xd);
    // in x axis
    double yw0, yw1;
    sincWeight(yw0, yw1, yd);

    w00 = xw0 * yw0;
    w01 = xw0 * yw1;
    w10 = xw1 * yw0;
    w11 = xw1 * yw1;
}

void biSincWeightGrid(double& w00, double& w01,
                      double& w10, double& w11,
                      int& x0,
                      int& y0,
                      const double x,
                      const double y)
{
    double xd, yd;
    floorDiff(x0, y0, xd, yd, x, y);

    biSincWeight(w00, w01, w10, w11, xd, yd);
}

void triSincWeight(double& w000, double& w001,
                   double& w010, double& w011,
                   double& w100, double& w101,
                   double& w110, double& w111,
                   const double xd,
                   const double yd,
                   const double zd)
{
    double xw0, xw1;
    double w00, w01, w10, w11;

    // in x aix
    sincWeight(xw0, xw1, xd);
    // in y and z axis
    biSincWeight(w00, w01, w10, w11, yd, zd);

    // in x axis
    w000 = xw0 * w00;
    w001 = xw0 * w01;
    w010 = xw0 * w10;
    w011 = xw0 * w11;
    w100 = xw1 * w00;
    w101 = xw1 * w01;
    w110 = xw1 * w10;
    w111 = xw1 * w11;
}

void triSincWeightGrid(double& w000, double& w001,
                       double& w010, double& w011,
                       double& w100, double& w101,
                       double& w110, double& w111,
                       int& x0,
                       int& y0,
                       int& z0,
                       const double x,
                       const double y,
                       const double z)
{
    double xd, yd, zd;
    floorDiff(x0, y0, z0, xd, yd, zd, x, y, z);

    triSincWeight(w000, w001, w010, w011,
                  w100, w101, w110, w111,
                  xd, yd, zd);
}

void interpolationWeight2D(double& w00, double& w01,
                           double& w10, double& w11,
                           const double xd,
                           const double yd,
                           const Interpolation2DStyle style)
{
    if (style == nearest2D)
        biNearestWeight(w00, w01, w10, w11, xd, yd);
    else if (style == linear2D)
        biLinearWeight(w00, w01, w10, w11, xd, yd);
    else if (style == sinc2D)
        biSincWeight(w00, w01, w10, w11, xd, yd);
}

void interpolationWeight2DGrid(double& w00, double& w01,
                               double& w10, double& w11,
                               int& x0,
                               int& y0,
                               const double x,
                               const double y,
                               const Interpolation2DStyle style)
{
    if (style == nearest2D)
        biNearestWeightGrid(w00, w01, w10, w11, x0, y0, x, y);
    else if (style == linear2D)
        biLinearWeightGrid(w00, w01, w10, w11, x0, y0, x, y);
    else if (style == sinc2D)
        biSincWeightGrid(w00, w01, w10, w11, x0, y0, x, y);
}

void interpolationWeight3D(double& w000, double& w001,
                           double& w010, double& w011,
                           double& w100, double& w101,
                           double& w110, double& w111,
                           const double xd,
                           const double yd,
                           const double zd,
                           const Interpolation3DStyle style)
{
    if (style == nearest3D)
        triNearestWeight(w000, w001, w010, w011,
                         w100, w101, w110, w111,
                         xd, yd, zd);
    else if (style == linear3D)
        triLinearWeight(w000, w001, w010, w011,
                        w100, w101, w110, w111,
                        xd, yd, zd);
    else if (style == sinc3D)
        triSincWeight(w000, w001, w010, w011,
                      w100, w101, w110, w111,
                      xd, yd, zd);
}

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
                               const Interpolation3DStyle style)
{
    if (style == nearest3D)
        triNearestWeightGrid(w000, w001, w010, w011,
                             w100, w101, w110, w111,
                             x0, y0, z0, x, y, z);
    else if (style == linear3D)
        triLinearWeightGrid(w000, w001, w010, w011,
                            w100, w101, w110, w111,
                            x0, y0, z0, x, y, z);
    else if (style == sinc3D)
        triSincWeightGrid(w000, w001, w010, w011,
                          w100, w101, w110, w111,
                          x0, y0, z0, x, y, z);
}
