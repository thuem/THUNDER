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

#include <cstdio>

#include "Macro.h"
#include "Typedef.h"
#include "Utils.h"

/**
 * @ingroup SymmetryOperation
 * @brief Symmetry Operation of Rotation
 */
struct RotationSO
{
    /**
     * The number of fold in rotation.
     */
    int fold;

    /**
     * The rotation axis indicated by a vector.
     */
    vec3 axis;

    /**
     * Construct a specific RotationSO with a fold and a rotation axis.
     * @param fold the number of fold in rotation
     * @param x x component of rotation axis
     * @param y y component of rotation axis
     * @param z z component of rotation axis
     */
    RotationSO(const int fold,
               const double x,
               const double y,
               const double z);
};

/**
 * @ingroup SymmetryOperation
 * @brief Symmetry Operation of Reflexion
 */
struct ReflexionSO
{
    vec3 plane;

    ReflexionSO(const double x,
                const double y,
                const double z);
};

/**
 * @ingroup SymmetryOperation
 * @brief Symmetry Operation of Inversion
 */
struct InversionSO
{
    InversionSO();
};

/**
 * @ingroup SymmetryOperation
 * @brief Symmetry Operation Including Rotation, Reflexion and Inversion
 */
struct SymmetryOperation
{
    /**
     * The code indicating the type of symmetry operation.
     * 0: rotation
     * 1: reflexion
     * 2: inversion
     */
    int id;

    /**
     * If it is a rotation, this 3-vector stores the rotation axis.
     * If it is a reflexion, this 3-vector stores the reflexion plane.
     */
    vec3 axisPlane;

    /**
     * If it s a rotation, it stores the fold of the rotation.
     */
    int fold;

    /**
     * Construct a SymmetryOperation object from a rotation symmetry operation.
     * @param rts the rotation symmetry operation
     */
    SymmetryOperation(const RotationSO rts);

    /**
     * Construct a SymmetryOperation object from a reflexion symmetry operation.
     * @param rfs the reflexion symmetry operation
     */
    SymmetryOperation(const ReflexionSO rfs);

    /**
     * Construct a SymmetryOperation object from a inversion symmetry operation/
     * @param ivs the inversion symmetry operation
     */
    SymmetryOperation(const InversionSO ivs);
};

/**
 * This function displays the components in a SymmetryOperation object.
 * @param so the SymmetryOperation object
 */
void display(const SymmetryOperation so);

/**
 * This function displays the componenets in a vector of SymmetryOperation
 * objects.
 * @param so a vector of the SymmetryOperation objects.
 */
void display(const vector<SymmetryOperation>& so);

#endif // SYMMETRY_OPERATION_H
