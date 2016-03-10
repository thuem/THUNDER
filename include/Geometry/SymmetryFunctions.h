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

#include <boost/regex.hpp>
#include <cstdlib>
#include <vector>

#include "Macro.h"
#include "Typedef.h"
#include "Error.h"

#include "Euler.h"
#include "PointGroup.h"
#include "SymmetryOperation.h"

//using namespace std;
using namespace boost;

void symmetryGroup(int& pgGroup,
                   int& pgOrder,
                   const char sym[]);
/* translate sym to symmetry group
 * return false when translation is not possible */

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
