/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

#define N 6000

int main(int argc, const char* argv[])
{
    Symmetry sym("C5");

    Particle particle(6000, 30, 30, &sym);

    display(particle);
}
