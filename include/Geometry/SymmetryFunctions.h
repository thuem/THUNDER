/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef SYMMETRY_FUNCTIONS_H
#define SYMMETRY_FUNCTIONS_H

#include <cstdlib>

#include "Macro.h"
#include "Typedef.h"
#include "Logging.h"

#include "Euler.h"
#include "PointGroup.h"
#include "SymmetryOperation.h"
#include "Utils.h"

/**
 * This function translates a string indicating the symmetry group to the code
 * of symmetry group and the order in that group.
 * @param pgGroup the code of symmetry group
 * @param pgOrder the order in the symmetry group
 * @param sym the string indicating the symmetry group
 */
void symmetryGroup(int& pgGroup,
                   int& pgOrder,
                   const char sym[]);

/**
 * This function adds symmetry operations to a vector according to the symmetry
 * information.
 * @param entry a vector for storing the symmetry operations
 * @param pgGroup the code of symmetry group
 * @param pgOrder the order in the symmetry group
 */
void fillSymmetryEntry(vector<SymmetryOperation>& entry,
                       const int pgGroup,
                       const int pgOrder);

/**
 * This function counts the number of symmetry operations, the number of
 * rotation axises, the number of mirros planes and the number of inversions
 * according to the vector of symmetry operations.
 * @param nSymmetryOperation the number of symmetry operations
 * @param nRotation the number of rotation axises
 * @param nReflexion the number of mirror planes
 * @param nInversion the number of inversions
 * @param entry a vector for storing the symmetry operations
 */
void countSymmetryElements(int& nSymmetryOperation,
                           int& nRotation,
                           int& nReflexion,
                           int& nInversion,
                           const vector<SymmetryOperation>& entry);

#endif // SYMMETRY_FUNCTIONS_H
