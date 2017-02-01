/*******************************************************************************
 * Author: Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include <iostream>

#include "Random.h"

#define N 20
#define M 10

#define TEST_DISCRETE_FLAT

INITIALIZE_EASYLOGGINGPP

int main(int argc, const char* argv[])
{
    gsl_rng* engine = get_random_engine();

#ifdef TEST_DISCRETE_FLAT
    for (int i = 0; i < M; i++)
    {
        //std::cout << gsl_rng_get(engine) % N << std::endl;
        std::cout << gsl_rng_uniform_int(engine, N) << std::endl;
    }
#endif
}
