/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "MLFunctions.h"

double norm(const Image& src1,
            const Image& src2,
            const Image& invSigma)
{
    gsl_vector_complex vecSrc1, vecSrc2, vecInvSigma;

    vecSrc1.size = src1.sizeFT();
    vecSrc1.data = &src1[0];

    vecSrc2.size = src2.sizeFT();
    vecSrc2.data = &src2[0];

    vecInvSigma.size = invSigma.sizeFT();
    vecInvSimga.data = &invSigma[0];

    gsl_vector_complex vec;
    gsl_vector_complex_memcpy(vec, vecSrc1);
}

double norm(const Projector& proj,
            const Coordinate5D coord,
            const Image& ctf,
            const Image& invSigma)
{
    Image 
