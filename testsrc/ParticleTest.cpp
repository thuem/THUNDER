/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

#define N 10 
#define K 50

int main(int argc, const char* argv[])
{
    Symmetry sym("C5");

    Particle particle(N, 30, 30, &sym);

    for (int i = 0; i < K; i++)
        particle.perturb();

    display(particle);
}
