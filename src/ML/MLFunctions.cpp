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
    Image img = src1;
    SUB_FT(img, src2);
    MUL_FT(img, invSigma); 
}

double norm(const Projector& proj,
            const Coordinate5D coord,
            const Image& ctf,
            const Image& invSigma)
{
    Image 
