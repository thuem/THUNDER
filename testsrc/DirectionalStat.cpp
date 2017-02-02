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
#define KAPPA_7 32

#define N 1000

//#define TEST_PDF_VMS

#define TEST_SAMPLE_VMS

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char* argv[])
{
#ifdef TEST_PDF_VMS
    for (double theta = -M_PI; theta < M_PI; theta += 0.01)
        printf("%6f   %6f   %6f   %6f   %6f   %6f\n",
               theta,
               pdfVMS(theta, MU, KAPPA_0),
               pdfVMS(theta, MU, KAPPA_1),
               pdfVMS(theta, MU, KAPPA_2),
               pdfVMS(theta, MU, KAPPA_3),
               pdfVMS(theta, MU, KAPPA_4));
#endif

#ifdef TEST_SAMPLE_VMS
    vec VMS_0 = vec::Zero(N);
    vec VMS_1 = vec::Zero(N);
    vec VMS_2 = vec::Zero(N);
    vec VMS_3 = vec::Zero(N);
    vec VMS_4 = vec::Zero(N);
    vec VMS_5 = vec::Zero(N);
    vec VMS_6 = vec::Zero(N);
    vec VMS_7 = vec::Zero(N);

    sampleVMS(VMS_0, MU, KAPPA_0, N);
    sampleVMS(VMS_1, MU, KAPPA_1, N);
    sampleVMS(VMS_2, MU, KAPPA_2, N);
    sampleVMS(VMS_3, MU, KAPPA_3, N);
    sampleVMS(VMS_4, MU, KAPPA_4, N);
    sampleVMS(VMS_5, MU, KAPPA_5, N);
    sampleVMS(VMS_6, MU, KAPPA_6, N);
    sampleVMS(VMS_7, MU, KAPPA_7, N);

    for (int i = 0; i < N; i++)
        printf("%15.6lf   %15.6lf   %15.6lf   %15.6lf   %15.6lf   %15.6lf   %15.6lf   %15.6lf\n",
               VMS_0(i),
               VMS_1(i),
               VMS_2(i),
               VMS_3(i),
               VMS_4(i),
               VMS_5(i),
               VMS_6(i),
               VMS_7(i));
#endif
}
