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
#include <iostream>

#include <glog/logging.h>

#include "Macro.h"
#include "Typedef.h"
#include "Error.h"

#include "Euler.h"
#include "PointGroup.h"
#include "SymmetryOperation.h"
#include "SymmetryFunctions.h"

using namespace std;

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
#define SAME_MATRIX(A, B) (sqrt((A - B).colwise().squaredNorm().sum()) < 1e4)

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
        return ((norm.dot(vec3(pg_##PG##_a1)) >= 0) && \
                (norm.dot(vec3(pg_##PG##_a2)) >= 0) && \
                (norm.dot(vec3(pg_##PG##_a3)) >= 0)); \
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

        /**
         * construct from the ID of symmetry group
         * @param sym ID of symmetry group
         */
        Symmetry(const char sym[]);

        /**
         * construct from the code of point group and the order of point group
         * @param pgGroup the code of point group
         * @param pgOrder the order of point group
         */
        Symmetry(const int pgGroup,
                 const int pgOrder);

        Symmetry(const Symmetry& that);

        ~Symmetry();

        Symmetry& operator=(const Symmetry& that);

        void init(const char sym[]);

        /**
         * This function returns the code of point group.
         */
        int pgGroup() const;

        /**
         * This function returns the order of point group.
         */
        int pgOrder() const;

        /**
         * This function gets the left transformation matrix and the right
         * transformation matrix of the ith symmetry element.
         * @param L the left transformation matrix
         * @param R the right transformation matrix
         * @param i the rank of symmetry element
         */
        void get(mat33& L,
                 mat33& R,
                 const int i) const;

        /**
         * This function calculates how many symmetry elements there are in this
         * symmetry group.
         */
        int nSymmetryElement() const;

        /**
         * This function clears up the storage.
         */
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

/**
 * This function displays the content of a Symmetry object.
 * @param sym the Symmetry object to be displayed
 */
void display(const Symmetry& sym);

/**
 * This function determines whether the direction given belongs to a certain
 * asymmetric unit of a point group or not.
 * @param dir the direction vector
 * @param sym the symmetry group
 */
bool asymmetryUnit(const vec3 dir,
                   const Symmetry& sym);

/**
 * This function determines whether the direction given by phi and theta
 * belongs to a certain asymmetric unit of a point group or not.
 * @param phi phi
 * @param theta theta
 * @param sym the symmetry group
 */
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
