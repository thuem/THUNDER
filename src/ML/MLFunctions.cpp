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

double norm(const Image& src,
            const Projector& proj,
            const Coordinate5D coord,
            const Image& ctf,
            const Image& invSigma)
{
    Image img(ctf.nColRL(), ctf.nRowRL(), fourierSpace);
    proj.project(img, coord);

    MUL_FT(img, ctf);
    SUB_FT(img, src);
    MUL_FT(img, invSigma);
    return gsl_pow_2(norm(img)) / 2;
}
