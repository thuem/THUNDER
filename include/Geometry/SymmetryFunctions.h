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
#include <vector>

#include <glog/logging.h>

#include "Macro.h"
#include "Typedef.h"
#include "Error.h"

#include "Euler.h"
#include "PointGroup.h"
#include "SymmetryOperation.h"

using namespace std;

/* This function translates a string indicating the symmetry group to the code
 * of symmetry group and the order in that group.
 * @param pgGroup the code of symmetry group
 * @param pgOrder the order in the symmetry group
 * @param sym the string indicating the symmetry group
 */
void symmetryGroup(int& pgGroup,
                   int& pgOrder,
                   const char sym[]);

void fillSymmetryEntry(vector<SymmetryOperation>& entry,
                       const int pgGroup,
                       const int pgOrder);
/* add symmetry information to the entry */

void countSymmetryElements(int& nSymmetryOperation,
                           int& nRotation,
                           int& nReflexion,
                           int& nInversion,
                           const vector<SymmetryOperation>& entry);
/* count the number of symmetry operations
 * count the number of rotation axis, mirror plane and inversions
 * respectively */

#endif // SYMMETRY_FUNCTIONS_H
