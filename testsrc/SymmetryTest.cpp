/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <cstdio>
#include <iostream>

#include "Symmetry.h"

//#define TEST_ASY

//#define TEST_FILL_RL

//#define TEST_SAME_MATRIX

//#define TEST_COUNTERPART

//#define ANCHOR_POINT

#define TEST

//#define TEST_QUAT

//#define TEST_ROT

#define SYM "C4"

#define N 1000

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char* argv[])
{
    loggerInit(argc, argv);
   
#ifdef TEST_ASY
    printf("ASY(O, 0, 0): %d\n", ASY(O, 0, 0));
    printf("ASY(0, M_PI, 0): %d\n", ASY(O, M_PI, 0));
    printf("ASY(0, M_PI, M_PI / 2): %d\n", ASY(O, M_PI, M_PI / 2));
#endif

#ifdef TEST_FILL_RL
    mat33 R;
    rotate3D(R, 0, vec3(0.5773502, 0.5773502, 0.5773502));

    std::cout << R << std::endl;
#endif

#ifdef TEST_SAME_MATRIX
    mat33 A = mat33::Zero();
    mat33 B = mat33::Ones();
    /***
    mat33 A << 1, 0, 0,
               0, 1, 0,
               0, 0, 1;
    mat33 B << 0, 1, 0,
               1, 0, 0,
               0, 0, 1;
    ***/

    //std::cout << (sqrt((A - B).colwise().squaredNorm().sum()) << std::endl;

    std::cout << SAME_MATRIX(A, B) << std::endl;
#endif

#ifdef TEST_COUNTERPART
    Symmetry sym(SYM);

    for (int i = 0; i < N; i++)
    {
        vec4 quat;
        randQuaternion(quat);

        //std::cout << quat << std::endl << std::endl;

        symmetryCounterpart(quat, sym);

        //std::cout << quat << std::endl << std::endl;

        // vec4 r(0, 1, 0, 0);
        
        vec4 r;

        quaternion_mul(r, quat, ANCHOR_POINT_0);
        quaternion_mul(r, r, quaternion_conj(quat));

        //std::cout << r.dot(ANCHOR_POINT) << std::endl;

        printf("%15.6lf %15.6lf %15.6lf\n",
               r(1),
               r(2),
               r(3));
    }
#endif

#ifdef ANCHOR_POINT
    gsl_rng* engine = get_random_engine();

    vec3 v(gsl_ran_gaussian(engine, 1),
           gsl_ran_gaussian(engine, 1),
          gsl_ran_gaussian(engine, 1));

    v /= v.norm();

    std::cout << v << std::endl;
#endif

#ifdef TEST
        Symmetry sym(argv[1]);

        std::cout << "nSymmetryElement = " << sym.nSymmetryElement() << std::endl;

        display(sym);

        /***
        vector<mat33> sr;

        mat33 rot;
        rot << 1, 0, 0,
               0, 1, 0,
               0, 0, 1;

        symmetryRotation(sr, rot, &sym);

        std::cout << "pgOrder = " << sym.pgOrder() << std::endl;

        for (int i = 0; i < (int)sr.size(); i++)
            std::cout << sr[i] << std::endl;
        ***/
#endif

#ifdef TEST_QUAT
        Symmetry sym(argv[1]);

        vec4 q0 = ANCHOR_POINT_1;

        printf("%12.6lf %12.6lf %12.6lf\n", q0(1), q0(2), q0(3));

        for (int i = 0; i < sym.nSymmetryElement(); i++)
        {
            vec4 q = sym.quat(i);

            quaternion_mul(q0, q, ANCHOR_POINT_1);
            quaternion_mul(q0, q0, quaternion_conj(q));
            
            printf("%12.6lf %12.6lf %12.6lf\n", q0(1), q0(2), q0(3));
        }
#endif

#ifdef TEST_ROT
        Symmetry sym(argv[1]);

        vec3 r;
        
        r(0) = ANCHOR_POINT_1(1);
        r(1) = ANCHOR_POINT_1(2);
        r(2) = ANCHOR_POINT_1(3);

        printf("%12.6lf %12.6lf %12.6lf\n", r(0), r(1), r(2));

        for (int i = 0; i < sym.nSymmetryElement(); i++)
        {
            mat33 L, R;
            sym.get(L, R, i);

            vec3 v = R * r;
            
            vec4 q;

            printf("%12.6lf %12.6lf %12.6lf\n", v(0), v(1), v(2));

            quaternion(q, R);

            vec4 t;
            quaternion_mul(t, q, ANCHOR_POINT_1);
            quaternion_mul(t, t, quaternion_conj(q));

            printf("%12.6lf %12.6lf %12.6lf\n", t(1), t(2), t(3));
        }
#endif
}
