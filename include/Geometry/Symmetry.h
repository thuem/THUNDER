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

#define SYM_ID_LENGTH 4

#define SAME_MATRIX(A, B) (norm(A - B) < 1e4)
/* axis has an accuracy of 6-7 decimal numbers 
 * As the mulplication here will not decrease the accuracy for 1e2 fold,
 * choosing 1e4 as the accuracy limit is sufficient. */

#define ASY(PG, phi, theta) \
    [](const double _phi, const double _theta) \
    { \
        vec3 norm; \
        direction(norm, _phi, _theta); \
        return ((sum(norm % vec3(pg_##PG##_a1)) >= 0) && \
                (sum(norm % vec3(pg_##PG##_a2)) >= 0) && \
                (sum(norm % vec3(pg_##PG##_a3)) >= 0)); \
    }(phi, theta)

class Symmetry
{
    private:

        int _pgGroup;

        int _pgOrder;

        vector<mat33> _L;

        vector<mat33> _R;

    public:

        Symmetry();

        Symmetry(const char sym[]);

        Symmetry(const int pgGroup,
                 const int pgOrder);

        Symmetry(const Symmetry& that);

        ~Symmetry();

        Symmetry& operator=(const Symmetry& that);

        void init(const char sym[]);

        int pgGroup() const;

        int pgOrder() const;

        void get(mat33& L,
                 mat33& R,
                 const int i) const;
        /* get the ith symmetry element */

        int nSymmetryElement() const;

        void clear();

    private:

        void init();

        void init(const vector<SymmetryOperation>& entry);

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

bool asymmetryUnit(const vec3 dir,
                   const Symmetry& sym);

bool asymmetryUnit(const double phi,
                   const double theta,
                   const Symmetry& sym);

bool asymmetryUnit(const double phi,
                   const double theta,
                   const int pgGroup,
                   const int pgOrder);

void symmetryCounterpart(double& phi,
                         double& psi,
                         const Symmetry& sym);

void symmetryCounterpart(double& ex,
                         double& ey,
                         double& ez,
                         const Symmetry& sym);

#endif // SYMMETRY_H
