/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef SYMMETRY_H
#define SYMMETRY_H

#include <cstdlib>
#include <vector>

#include <armadillo>

#include "Macro.h"
#include "Typedef.h"
#include "Error.h"

#include "Euler.h"
#include "PointGroup.h"
#include "SymmetryOperation.h"
#include "SymmetryFunctions.h"

using namespace std;
using namespace arma;

#define SAME_MATRIX(A, B) (norm(A - B) < 1e12)
/* double has an accuracy of 15-16 decimal numbers 
 * As the mulplication here will not decrease the accuracy for 1e3 fold,
 * choosing 1e12 as the accuracy limit is sufficient. */

class Symmetry
{
    private:

        vector<mat33> _L;

        vector<mat33> _R;

    public:

        Symmetry();

        Symmetry(const char sym[]);

        Symmetry(const int pgGroup,
                 const int pgOrder);

        Symmetry(const vector<SymmetryOperation>& entry);

        Symmetry(const Symmetry& that);

        ~Symmetry();

        Symmetry& operator=(const Symmetry& that);

        void get(mat33& L,
                 mat33& R,
                 const int i) const;
        /* get the ith symmetry element */

        void init(const char sym[]);

        void init(const int pgGroup,
                  const int pgOrder);

        void init(const vector<SymmetryOperation>& entry);

        int nSymmetryElement() const;

        void clear();

    private:

        void append(const mat33& L,
                    const mat33& R);

        void set(const mat33& L,
                 const mat33& R,
                 const int i);
        /* set the ith symmetry element */

        void fillLR(const vector<SymmetryOperation>& entry);

        bool novo(const mat33& L,
                  const mat33& R) const;
        /* check whether (L, R) is novo or not */

        void completePointGroup();
};

void display(const Symmetry& sym);

#endif // SYMMETRY_H
