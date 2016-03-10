/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Coordinate5D.h"

Coordinate5D::Coordinate5D() {}

Coordinate5D::Coordinate5D(const double phi,
                           const double theta,
                           const double psi,
                           const double x,
                           const double y)
{
    this->phi = phi;
    this->theta = theta;
    this->psi = psi;
    this->x = x;
    this->y = y;
}

void display(const Coordinate5D& coord)
{
    printf("%12f %12f %12f %12f %12f\n",
           coord.phi,
           coord.theta,
           coord.psi,
           coord.x,
           coord.y);
    /***
    printf("theta = %f, phi = %f, psi = %f, x = %f, y = %f\n",
           coord.theta,
           coord.phi,
           coord.psi,
           coord.x,
           coord.y);
           ***/
}
