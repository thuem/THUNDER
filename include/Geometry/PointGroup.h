/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#ifndef POINT_GROUP_H
#define POINT_GROUP_H

#define PG_CN   202
#define PG_DN   206
#define PG_T    209
#define PG_O    212
#define PG_I    214

/**
 * 3-fold Axis 1 <--> 3-fold Axis 2
 */
static const double pg_T_a1[] = {-0.942809,
                                 0,
                                 0};

/**
 * 3-fold Axis 2 <--> 3-fold Axis 3
 */
static const double pg_T_a2[] = {0.471405,
                                 0.272165,
                                 0.7698};

/**
 * 3-fold Axis 3 <--> 3-fold Axis 1
 */
static const double pg_T_a3[] = {0.471405,
                                 0.816497,
                                 0};

/**
 * 2-fold Axis 1 <--> 3-fold Axis 2
 */
static const double pg_TD_a1[] = {-0.942809,
                                  0,
                                  0};

/**
 * 3-fold Axis 2 <--> 3-fold Axis 5
 */
static const double pg_TD_a2[] = {0.471405,
                                  0.272165,
                                  0.7698};

/**
 * 3-fold Axis 5 <--> 2-fold Axis 1
 */
static const double pg_TD_a3[] = {0,
                                  0.471405,
                                  -0.666667};

/**
 * 3-fold Axis 1 <--> 2-fold Axis 1
 */
static const double pg_TH_a1[] = {-0.816496,
                                  0,
                                  0};

/**
 * 2-fold Axis 1 <--> 2-fold Axis 2
 */
static const double pg_TH_a2[] = {0.707107,
                                  0.408248,
                                  -0.57735};

/**
 * 2-fold Axis 2 <--> 3-fold Axis 1
 */
static const double pg_TH_a3[] = {-0.408248,
                                  -0.707107,
                                  0};

/**
 * 3-fold Axis 1 <--> 3-fold Axis 2
 */
static const double pg_O_a1[] = {0, -1, 1};

/**
 * 3-fold Axis 1 <--> 4-fold Axis
 */
static const double pg_O_a2[] = {1, 1, 0};

/**
 * 4-fold Axis <--> 3-fold Axis 1
 */
static const double pg_O_a3[] = {-1, 1, 0};

static const double pg_I1_a1[] = {0, 1, 0};

static const double pg_I1_a2[] = {-0.309017,
                                  -0.809017,
                                  -0.5};

static const double pg_I1_a3[] = {-0.309017,
                                  -0.809017,
                                  0.5};

/**
 * 5-fold Axis 1 <--> 5-fold Axis 2
 */
static const double pg_I2_a1[] = {0, 1, 0};

/**
 * 5-fold Axis 2 <--> 3-fold Axis
 */
static const double pg_I2_a2[] = {-0.5,
                                  -0.809017,
                                  0.309017};

/**
 * 3-fold Axis <--> 5-fold Axis 1
 */
static const double pg_I2_a3[] = {0.5,
                                  -0.809017,
                                  0.309017}; 

static const double pg_I3_a1[] = {0, 1, 0};

static const double pg_I3_a2[] = {-0.587785, -0.809017, 0};

static const double pg_I3_a3[] = {0.262866, -0.809017, 0.525731};

static const double pg_I4_a1[] = {0, -1, 0};

static const double pg_I4_a2[] = {0.262866, 0.809017, -0.525731};

static const double pg_I4_a3[] = {-0.587785, 0.809017, 0};

static const double pg_I1H_a1[] = {0, 1, 0};

static const double pg_I1H_a2[] = {-0.309017, -0.809017, -0.5};

static const double pg_I1H_a3[] = {-0.309017, -0.809017, -0.5};

static const double pg_I1H_a4[] = {0, 0, 1};

static const double pg_I2H_a1[] = {0, 1, 0};

static const double pg_I2H_a2[] = {-0.5, -0.809017, 0.309017};

static const double pg_I2H_a3[] = {0.5, -0.809017, 0.309017};

static const double pg_I2H_a4[] = {1, 0, 0};

static const double pg_I3H_a1[] = {0, 1, 0};

static const double pg_I3H_a2[] = {-0.587785, 0.809017, 0};

static const double pg_I3H_a3[] = {0.262866, -0.809017, 0.525731};

static const double pg_I3H_a4[] = {0.850651, 0, 0.525731};

static const double pg_I4H_a1[] = {0, -1, 0};

static const double pg_I4H_a2[] = {0.262866, 0.809017, -0.525731};

static const double pg_I4H_a3[] = {-0.587785, 0.809017, 0};

static const double pg_I4H_a4[] = {0.850651, 0, -0.525731};

#endif // POINT_GROUP_H
