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

const static gsl_rng_type* RANDT = gsl_rng_default;
static gsl_rng* RANDR = gsl_rng_alloc(RANDT);

#endif // RANDOM_H
