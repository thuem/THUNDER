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

#define PG_CI  200
#define PG_CS  201
#define PG_CN  202
#define PG_CNV 203
#define PG_CNH 204
#define PG_SN  205
#define PG_DN  206
#define PG_DNV 207
#define PG_DNH 208
#define PG_T   209
#define PG_TD  210
#define PG_TH  211
#define PG_O   212
#define PG_OH  213
#define PG_I   214
#define PG_IH  215

#define PG_I1   216 
#define PG_I2   217
#define PG_I3   218
#define PG_I4   219
#define PG_I5   220

#define PG_I1H  221
#define PG_I2H  222
#define PG_I3H  223
#define PG_I4H  224
#define PG_I5H  225

static const double pg_T_a1[] = {-0.942809,
                                 0,
                                 0};
/* 3-fold Axis 1 <--> 3-fold Axis 2 */

static const double pg_T_a2[] = {0.471405,
                                 0.272165,
                                 0.7698};
/* 3-fold Axis 2 <--> 3-fold Axis 3 */

static const double pg_T_a3[] = {0.471405,
                                 0.816497,
                                 0};
/* 3-fold Axis 3 <--> 3-fold Axis 1 */

static const double pg_TD_a1[] = {-0.942809,
                                  0,
                                  0};
/* 2-fold Axis 1 <--> 3-fold Axis 2 */

static const double pg_TD_a2[] = {0.471405,
                                  0.272165,
                                  0.7698};
/* 3-fold Axis 2 <--> 3-fold Axis 5 */

static const double pg_TD_a3[] = {0,
                                  0.471405,
                                  -0.666667};
/* 3-fold Axis 5 <--> 2-fold Axis 1 */

static const double pg_TH_a1[] = {-0.816496,
                                  0,
                                  0};
/* 3-fold Axis 1 <--> 2-fold Axis 1 */

static const double pg_TH_a2[] = {0.707107,
                                  0.408248,
                                  -0.57735};
/* 2-fold Axis 1 <--> 2-fold Axis 2 */

static const double pg_TH_a3[] = {-0.408248,
                                  -0.707107,
                                  0};
/* 2-fold Axis 2 <--> 3-fold Axis 1 */

static const double pg_O_a1[] = {0, -1, 1};
/* 3-fold Axis 1 <--> 3-fold Axis 2 */

static const double pg_O_a2[] = {1, 1, 0};
/* 3-fold Axis 1 <--> 4-fold Axis */

static const double pg_O_a3[] = {-1, 1, 0};
/* 4-fold Axis <--> 3-fold Axis 1 */

static const double pg_I2_a1[] = {0, 1, 0};
/* 5-fold Axis 1 <--> 5-fold Axis 2 */

static const double pg_I2_a2[] = {-0.5,
                                  -0.809017007,
                                  0.309016986};
/* 5-fold Axis 2 <--> 3-fold Axis */

static const double pg_I2_a3[] = {0.5,
                                  -0.809017007,
                                  0.309016986}; 
/* 3-fold Axis <--> 5-fold Axis 1 */

#endif // POINT_GROUP_H
