/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Random.h"

void rand(char dst[], const int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = gsl_rng_get(RANDR) % 26 + 65;
}
