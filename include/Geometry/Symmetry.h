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

#include "Macro.h"
#include "Typedef.h"
#include "Error.h"
#include "Logging.h"

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
 *
 * @param PG    code indicating the symmetry group
 * @param phi   phi
 * @param theta theta
 */
#define ASY(PG, phi, theta) \
    ({ \
        double _phi = (phi), _theta = (theta); \
        vec3 norm; \
        direction(norm, _phi, _theta); \
        ((norm.dot(vec3(pg_##PG##_a1)) >= 0) && \
                (norm.dot(vec3(pg_##PG##_a2)) >= 0) && \
                (norm.dot(vec3(pg_##PG##_a3)) >= 0)); \
    })


/**
 * @ingroup Symmetry
 * @brief   Symmetry class can generate and store a vector of transformation
 *          matrices according to symmetry group information.
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
         *
         * @param sym ID of symmetry group
         */
        Symmetry(const char sym[]);

        /**
         * construct from the code of point group and the order of point group
         *
         * @param pgGroup the code of point group
         * @param pgOrder the order of point group
         */
        Symmetry(const int pgGroup,
                 const int pgOrder);

        /**
         * copy constructor
         */
        Symmetry(const Symmetry& that);

        /**
         * default deconstructor
         */
        ~Symmetry();

        /**
         * an overwrite of = operator for copying content of a Symmetry object
         */
        Symmetry& operator=(const Symmetry& that);

        /**
         * This function initialises by inputing the symmetry code. For example,
         * C15 stands for a 15-fold symmetry around Z-axis, D5 stands for a
         * 5-fold symmetry around Z-axis and a 2-fold symmetry around X-axis.
         *
         * @param sym the symmetry code
         */
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
         *
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

        /**
         * This function initialises the content.
         */
        void init();

        /**
         * This function initialises by input a series of symmetry operations.
         *
         * @param entry a series of symmetry operations
         */
        void init(const vector<SymmetryOperation>& entry);

        /**
         * This function adds a left matrix and a right matrix.
         *
         * @param L the left matrix
         * @param R the right matrix
         */
        void append(const mat33& L,
                    const mat33& R);

        /**
         * This function sets the i-th left matrix and right matrix.
         *
         * @param L the left matrix
         * @param R the right matrix
         */
        void set(const mat33& L,
                 const mat33& R,
                 const int i);

        /**
         * This function fills lefts matrices and right matrices by inputing
         * a series of symmetry operations.
         *
         * @param entry a series of symmetry operations
         */
        void fillLR(const vector<SymmetryOperation>& entry);

        /**
         * This function checks whether (L, R) is novo or not.
         *
         * @param L the left matrix
         * @param R the right matrix
         */
        bool novo(const mat33& L,
                  const mat33& R) const;

        /**
         * This function completes the transformation matrices of the certain
         * point group by searching and assigning the missing part.
         */
        void completePointGroup();
};

/**
 * This function displays the content of a Symmetry object.
 *
 * @param sym the Symmetry object to be displayed
 */
void display(const Symmetry& sym);

/**
 * This function determines whether the direction given belongs to a certain
 * asymmetric unit of a point group or not.
 *
 * @param dir the direction vector
 * @param sym the symmetry group
 */
bool asymmetryUnit(const vec3 dir,
                   const Symmetry& sym);

/**
 * This function determines whether the direction given by phi and theta
 * belongs to a certain asymmetric unit of a point group or not.
 *
 * @param phi   phi
 * @param theta theta
 * @param sym   the symmetry group
 */
bool asymmetryUnit(const double phi,
                   const double theta,
                   const Symmetry& sym);

/**
 * This function determines whether the direction given by phi and theta belongs
 * to a certain asymmetric unit of a point group or not.
 *
 * @param phi     phi
 * @param theta   theta
 * @param pgGroup the code of point group
 * @param pgOrder the order of point group
 */
bool asymmetryUnit(const double phi,
                   const double theta,
                   const int pgGroup,
                   const int pgOrder);

/**
 * This function changes the direction given by phi and theta to the
 * corresponding direction belonging to a certain asymetric unit.
 *
 * @param phi   phi
 * @param theta theta
 * @param sym   the symmetry group
 */
void symmetryCounterpart(double& phi,
                         double& theta,
                         const Symmetry& sym);

/**
 * This function changes the direction given by a 3-vector to the corresponding
 * direction belonging to a certain asymetric unit.
 *
 * @param ex  the 1st element of the direction vector
 * @param ey  the 2nd element of the direction vector
 * @param ez  the 3rd element of the direction vector
 * @param sym the symmetry group
 */
void symmetryCounterpart(double& ex,
                         double& ey,
                         double& ez,
                         const Symmetry& sym);

/**
 * This function generates all corresponding rotation matrices of a rotation
 * matrix for a given symmetry group.
 *
 * @param sr  the corresponding rotation matrices
 * @param rot the rotation matrix
 * @param sym the symmety group
 */
void symmetryRotation(vector<mat33>& sr,
                      const mat33 rot,
                      const Symmetry* sym = NULL);

/***
void angleSymmetryC(double& phi,
                    double& theta,
                    double& psi,
                    const vec4& src,
                    const int fold);

void angleSymmetryD(double& phi,
                    double& theta,
                    double& psi,
                    const vec4& src,
                    const int fold);

void rotate3DSymmetryC(mat33& dst,
                       const vec4& src,
                       const int fold);

void rotate3DSymmetryD(mat33& dst,
                       const vec4& src,
                       const int fold);
***/

#endif // SYMMETRY_H
