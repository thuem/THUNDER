/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef SYMMETRY_OPERATION_H
#define SYMMETRY_OPERATION_H

#include <vector>
#include <cstdio>

#include <armadillo>

#include "Macro.h"
#include "Typedef.h"
#include "Error.h"

using namespace std;
using namespace arma;

struct RotationSO
{
    int fold;

    vec3 axis;

    RotationSO(const int fold,
               const double x,
               const double y,
               const double z);
};

struct ReflexionSO
{
    vec3 plane;

    ReflexionSO(const double x,
                const double y,
                const double z);
};

struct InversionSO
{
    InversionSO();
};

struct SymmetryOperation
{
    int id;
    /* 0: rotation
     * 1: reflexion
     * 2: inversion */

    vec3 axisPlane;
    // This is 3-element vector for storing rotation axis or reflexion plane.

    int fold;

    SymmetryOperation(const RotationSO rts);
    SymmetryOperation(const ReflexionSO rfs);
    SymmetryOperation(const InversionSO ivs);
};

void display(const SymmetryOperation so);

void display(const vector<SymmetryOperation>& so);

#endif // SYMMETRY_OPERATION_H
