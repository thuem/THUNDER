/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "DirectionalStat.h"

#define MU 0

#define KAPPA_0 0
#define KAPPA_1 0.5
#define KAPPA_2 1
#define KAPPA_3 2
#define KAPPA_4 4
#define KAPPA_5 8
#define KAPPA_6 16
#define KAPPA_7 100000

#define N 1000

#define TEST_SAMPLE_ACG

//#define TEST_INFER_ACG

//#define TEST_PDF_VMS

//#define TEST_SAMPLE_VMS

//#define TEST_INFER_VMS

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char* argv[])
{
#ifdef TEST_SAMPLE_ACG
    mat44 mat;

    /***
    mat << 1, 0, 0, 0,
           0, 5, 0, 0,
           0, 0, 5, 0,
           0, 0, 0, 5;
           ***/

    /***
    mat << 1, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 1;
    ***/

    /***
    mat << 1000, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 1;
    ***/

    mat << 1000, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 1;

    mat4 acg(N, 4);

    sampleACG(acg, mat, N);
    
    /***
    for (int i = 0; i < N; i++)
        printf("%15.6lf %15.6lf %15.6lf %15.6lf\n",
               acg(i, 0),
               acg(i, 1),
               acg(i, 2),
               acg(i, 3));
    ***/

    double k0, k1;

    inferACG(k0, k1, acg);

    printf("k0 = %15.6lf, k1 = %15.6lf\n", k0, k1);

    for (int i = 0; i < N; i++)
        if (acg(i, 0) < 0)
            acg.row(i) *= -1;

    inferACG(k0, k1, acg);

    printf("k0 = %15.6lf, k1 = %15.6lf\n", k0, k1);
#endif

#ifdef TEST_INFER_ACG
    mat4 acg = mat4::Zero(N, 4);

    acg.col(0) = vec::Ones(N);
    //acg.col(1) = vec::Ones(N);

    /***
    for (int i = 0; i < N; i++)
        printf("%15.6lf %15.6lf %15.6lf %15.6lf\n",
               acg(i, 0),
               acg(i, 1),
               acg(i, 2),
               acg(i, 3));
               ***/

    double k0, k1;

    inferACG(k0, k1, acg);

    printf("k0 = %15.6lf, k1 = %15.6lf\n", k0, k1);

#endif

#ifdef TEST_PDF_VMS
    for (double theta = -M_PI; theta < M_PI; theta += 0.01)
        printf("%6f   %6f   %6f   %6f   %6f   %6f\n",
               theta,
               pdfVMS(vec2(cos(theta), sin(theta)), vec2(cos(MU), sin(MU)), KAPPA_0),
               pdfVMS(vec2(cos(theta), sin(theta)), vec2(cos(MU), sin(MU)), KAPPA_1),
               pdfVMS(vec2(cos(theta), sin(theta)), vec2(cos(MU), sin(MU)), KAPPA_2),
               pdfVMS(vec2(cos(theta), sin(theta)), vec2(cos(MU), sin(MU)), KAPPA_3),
               pdfVMS(vec2(cos(theta), sin(theta)), vec2(cos(MU), sin(MU)), KAPPA_4));
#endif

    mat2 VMS_0 = mat2::Zero(N, 2);
    mat2 VMS_1 = mat2::Zero(N, 2);
    mat2 VMS_2 = mat2::Zero(N, 2);
    mat2 VMS_3 = mat2::Zero(N, 2);
    mat2 VMS_4 = mat2::Zero(N, 2);
    mat2 VMS_5 = mat2::Zero(N, 2);
    mat2 VMS_6 = mat2::Zero(N, 2);
    mat2 VMS_7 = mat2::Zero(N, 2);

    sampleVMS(VMS_0, vec2(cos(MU), sin(MU)), KAPPA_0, N);
    sampleVMS(VMS_1, vec2(cos(MU), sin(MU)), KAPPA_1, N);
    sampleVMS(VMS_2, vec2(cos(MU), sin(MU)), KAPPA_2, N);
    sampleVMS(VMS_3, vec2(cos(MU), sin(MU)), KAPPA_3, N);
    sampleVMS(VMS_4, vec2(cos(MU), sin(MU)), KAPPA_4, N);
    sampleVMS(VMS_5, vec2(cos(MU), sin(MU)), KAPPA_5, N);
    sampleVMS(VMS_6, vec2(cos(MU), sin(MU)), KAPPA_6, N);
    sampleVMS(VMS_7, vec2(cos(MU), sin(MU)), KAPPA_7, N);

#ifdef TEST_SAMPLE_VMS
    for (int i = 0; i < N; i++)
        printf("%15.6lf %15.6lf   %15.6lf %15.6lf   %15.6lf %15.6f   %15.6lf %15.6lf   %15.6lf %15.6lf   %15.6lf %15.6lf   %15.6lf %15.6lf   %15.6lf %15.6lf\n",
               VMS_0(i, 0),
               VMS_0(i, 1),
               VMS_1(i, 0),
               VMS_1(i, 1),
               VMS_2(i, 0),
               VMS_2(i, 1),
               VMS_3(i, 0),
               VMS_3(i, 1),
               VMS_4(i, 0),
               VMS_4(i, 1),
               VMS_5(i, 0),
               VMS_5(i, 1),
               VMS_6(i, 0),
               VMS_6(i, 1),
               VMS_7(i, 0),
               VMS_7(i, 1));
#endif

#ifdef TEST_INFER_VMS
    vec2 mu;
    double kappa;

    inferVMS(mu, kappa, VMS_0);
    printf("mu = (%lf, %lf), kappa = %lf\n", mu(0), mu(1), kappa);
    inferVMS(mu, kappa, VMS_1);
    printf("mu = (%lf, %lf), kappa = %lf\n", mu(0), mu(1), kappa);
    inferVMS(mu, kappa, VMS_2);
    printf("mu = (%lf, %lf), kappa = %lf\n", mu(0), mu(1), kappa);
    inferVMS(mu, kappa, VMS_3);
    printf("mu = (%lf, %lf), kappa = %lf\n", mu(0), mu(1), kappa);
    inferVMS(mu, kappa, VMS_4);
    printf("mu = (%lf, %lf), kappa = %lf\n", mu(0), mu(1), kappa);
    inferVMS(mu, kappa, VMS_5);
    printf("mu = (%lf, %lf), kappa = %lf\n", mu(0), mu(1), kappa);
    inferVMS(mu, kappa, VMS_6);
    printf("mu = (%lf, %lf), kappa = %lf\n", mu(0), mu(1), kappa);
    inferVMS(mu, kappa, VMS_7);
    printf("mu = (%lf, %lf), kappa = %lf\n", mu(0), mu(1), kappa);
#endif
}
