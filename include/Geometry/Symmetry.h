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

#include <iostream>

#include "Macro.h"
#include "Typedef.h"
#include "Logging.h"

#include "PointGroup.h"
#include "SymmetryOperation.h"
#include "SymmetryFunctions.h"

#include "Functions.h"

/**
 * Maximum ID Length of Symmetry
 * Example: C5, C4H, O, I, D2V
 */
#define SYM_ID_LENGTH 5

/**
 * This marcos determines whether a direction given by phi and theta belongs to
 * a certain asymmetric unit.
 *
 * @param PG    code indicating the symmetry group
 * @param phi   phi
 * @param theta theta
 */
#define ASY_3(PG, phi, theta) \
    ({ \
         double _phi = (phi), _theta = (theta); \
         vec3 norm; \
         direction(norm, _phi, _theta); \
         ((norm.dot(vec3(pg_##PG##_a1)) >= 0) && \
          (norm.dot(vec3(pg_##PG##_a2)) >= 0) && \
          (norm.dot(vec3(pg_##PG##_a3)) >= 0)); \
     })

#define ASY_4(PG, phi, theta) \
    ({ \
         double _phi = (phi), _theta = (theta); \
         vec3 norm; \
         direction(norm, _phi, _theta); \
         ((norm.dot(vec3(pg_##PG##_a1)) >= 0) && \
          (norm.dot(vec3(pg_##PG##_a2)) >= 0) && \
          (norm.dot(vec3(pg_##PG##_a3)) >= 0) && \
          (norm.dot(vec3(pg_##PG##_a4)) >= 0)); \
     })

inline bool SAME_MATRIX(const mat33& A,
                        const mat33& B)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (fabs(A(i, j) - B(i, j)) > EQUAL_ACCURACY)
                return false;

    return true;
};

static const vec4 ANCHOR_POINT_2(1, 0, 0, 0);

/***
static const vec4 ANCHOR_POINT_2(-0.320523,
                                 0.239029,
                                 0.0310742,
                                 -0.916059);
***/

static const vec4 ANCHOR_POINT_0(0, 0, 0, 1);

static const vec4 ANCHOR_POINT_1(0,
                                 0.395009,
                                 0.893974,
                                 0.211607);

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

        vector<vec4> _quat;

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

        vec4 quat(const int i) const;

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

        void append(const vec4& quat);

        /**
         * This function sets the i-th left matrix and right matrix.
         *
         * @param L the left matrix
         * @param R the right matrix
         */
        void set(const mat33& L,
                 const mat33& R,
                 const int i);

        void set(const vec4& quat,
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

bool asymmetry(const Symmetry& sym);

void symmetryCounterpart(vec4& quat,
                         const Symmetry& sym);

void symmetryRotation(vector<mat33>& sr,
                      const mat33 rot,
                      const Symmetry* sym = NULL);

#endif // SYMMETRY_H
