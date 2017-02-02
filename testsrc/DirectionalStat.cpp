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

#define TEST_PDF_VMS

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char* argv[])
{
#ifdef TEST_PDF_VMS
    for (double theta = -M_PI; theta < M_PI; theta += 0.01)
        printf("%6f   %6f   %6f   %6f   %6f   %06f\n",
               theta,
               pdfVMS(theta, MU, KAPPA_0),
               pdfVMS(theta, MU, KAPPA_1),
               pdfVMS(theta, MU, KAPPA_2),
               pdfVMS(theta, MU, KAPPA_3),
               pdfVMS(theta, MU, KAPPA_4));
#endif
}
