/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef RANDOM_H
#define RANDOM_H

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

gsl_rng* get_random_engine();
#endif // RANDOM_H
