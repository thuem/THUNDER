/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef ML_FUNCTIONS_H
#define ML_FUNCTIONS_H

double norm(const Projector& proj,
            const Coordinate5D coord,
            const Image& ctf,
            const Image& invSigma);

#endif // ML_FUNCTIONS_H
