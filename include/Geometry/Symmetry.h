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

#include <glog/logging.h>
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

/**
 * Maximum ID Length of Symmetry
 * Example: C5, C4H, O, I, D2V
 */
#define SYM_ID_LENGTH 4

/**
 * This macros determines whether A and B matrix is equal with tolerence of 1e4.
 * @param A A matrix
 * @param B B matrix
 */
#define SAME_MATRIX(A, B) (norm(A - B) < 1e4)

/**
 * This marcos determines whether a direction given by phi and theta belongs to
 * a certain asymmetric unit.
 * @param PG code indicating the symmetry group
 * @param phi phi
 * @param theta theta
 */
#define ASY(PG, phi, theta) \
    [](const double _phi, const double _theta) \
    { \
        vec3 norm; \
        direction(norm, _phi, _theta); \
        return ((sum(norm % vec3(pg_##PG##_a1)) >= 0) && \
                (sum(norm % vec3(pg_##PG##_a2)) >= 0) && \
                (sum(norm % vec3(pg_##PG##_a3)) >= 0)); \
    }(phi, theta)

/**
 * @ingroup Symmetry
 * @brief Symmetry class can generate and store a vector of transformation
 * matrices according to symmetry group information.
 */
class Symmetry
{
    private:

        /**
         * the code of point group
         */
        int _pgGroup;

        /**
         * the order of point group
         */
        int _pgOrder;

        /**
         * a vector of left transformation matrices
         */
        vector<mat33> _L;

        /**
         * a vector of right transformation matices
         */
        vector<mat33> _R;

    public:

        /**
         * default constructor
         */
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
