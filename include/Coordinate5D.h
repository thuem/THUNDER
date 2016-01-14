/*******************************************************************************
 * Authors Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef COORDINATE_5D_H
#define COORDINATE_5D_H

#include <cstdio>

struct Coordinate5D
{
    double phi;
    double theta;
    double psi;
    double x;
    double y;

    Coordinate5D();

    Coordinate5D(const double phi,
                 const double theta,
                 const double psi,
                 const double x,
                 const double y);
};

void display(const Coordinate5D& coord);

#endif // COORDINATE_5D_H
