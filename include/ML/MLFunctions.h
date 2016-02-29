/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef TYPEDEF_H
#define TYPEDEF_H

double norm(const Image& src1,
            const Image& src2,
            const Image& invSigma);

double norm(const Projector& proj,
            const Coordinate5D coord,
            const Image& ctf,
            const Image& invSigma);

#endif // TYPEDEF_H
