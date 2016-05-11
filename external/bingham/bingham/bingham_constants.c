
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "bingham/util.h"
#include "bingham/bingham_constants.h"
#include "bingham/bingham_constant_tables.h"

const double BINGHAM_MIN_CONCENTRATION = -900;


/** Note: It is assumed that z1 < z2 < z3 < 0 for all the bingham functions. **/


#define EPSILON 1e-8
#define ITERATION_MULT 10
#define MIN_ITERATIONS 10



static kdtree_t *dY_tree_3d = NULL;  // dY = d(logF) = dF/F
static int **dY_indices_3d = NULL;  // map from the indices of dY to indices (i,j,k) of F, dF*, etc.


/*
 * Initialize the KD-trees for fast constant lookups.
 */
void bingham_constants_init()
{
  if (dY_tree_3d)  // already initialized
    return;

  double t0 = get_time_ms();

  int i, j, k;
  const int n = BINGHAM_TABLE_LENGTH;

  dY_indices_3d = new_matrix2i(n*n*n, 3);

  // build dY3d vectors
  double **dY3d = new_matrix2(n*n*n, 3);
  int cnt = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++) {
      for (k = 0; k <= j; k++) {
	//dY3d[cnt][0] = bingham_dY1_table[i][j][k];
	//dY3d[cnt][1] = bingham_dY2_table[i][j][k];
	//dY3d[cnt][2] = bingham_dY3_table[i][j][k];
	dY3d[cnt][0] = bingham_dF1_table_3d[i][j][k] / bingham_F_table_3d[i][j][k];
	dY3d[cnt][1] = bingham_dF2_table_3d[i][j][k] / bingham_F_table_3d[i][j][k];
	dY3d[cnt][2] = bingham_dF3_table_3d[i][j][k] / bingham_F_table_3d[i][j][k];
	dY_indices_3d[cnt][0] = i;
	dY_indices_3d[cnt][1] = j;
	dY_indices_3d[cnt][2] = k;
	cnt++;
      }
    }
  }

  // create a KD-tree from the vectors in dY3d
  dY_tree_3d = kdtree(dY3d, cnt, 3);

  free_matrix2(dY3d);

  fprintf(stderr, "Initialized bingham constants in %.0f ms\n", get_time_ms() - t0);
}


static double bingham_dY_params_3d_slow_eval(double *err, double *Z, double *dY)
{
  double F = bingham_F_lookup_3d(Z);
  double dF[3];
  bingham_dF_lookup_3d(dF, Z);
  err[0] = dF[0]/F - dY[0];
  err[1] = dF[1]/F - dY[1];
  err[2] = dF[2]/F - dY[2];

  double g = err[0]*err[0] + err[1]*err[1] + err[2]*err[2];

  return isnan(g) ? DBL_MAX : g;
}

//dbug: improve this later!!!!
static void bingham_dY_params_3d_slow(double *Z, double *F, double *dY)
{
  if (dY_tree_3d == NULL)
    bingham_constants_init();

  int nn_index = kdtree_NN(dY_tree_3d, dY);

  int i = dY_indices_3d[nn_index][0];
  int j = dY_indices_3d[nn_index][1];
  int k = dY_indices_3d[nn_index][2];

  double r0 = bingham_table_range[i];
  double r1 = bingham_table_range[j];
  double r2 = bingham_table_range[k];

  Z[0] = -r0*r0;
  Z[1] = -r1*r1;
  Z[2] = -r2*r2;

  // perform gradient descent to improve Z estimate
  double dz = .01;
  double delta = 1000;
  double err[3];
  int iter = 0;
  int max_iter = 50;
  for (iter = 0; iter < max_iter; iter++) {
    double g = bingham_dY_params_3d_slow_eval(err, Z, dY);
    //printf("Z = [%.2f, %.2f, %.2f], g = %.10f\n", Z[0], Z[1], Z[2], g);  //dbug

    Z[0] += dz;
    double g0 = bingham_dY_params_3d_slow_eval(err, Z, dY);
    Z[0] -= dz;
    double dgdZ0 = (g0 - g)/dz;

    Z[1] += dz;
    double g1 = bingham_dY_params_3d_slow_eval(err, Z, dY);
    Z[1] -= dz;
    double dgdZ1 = (g1 - g)/dz;
    
    Z[2] += dz;
    double g2 = bingham_dY_params_3d_slow_eval(err, Z, dY);
    Z[2] -= dz;
    double dgdZ2 = (g2 - g)/dz;

    //printf("dgdZ = [%.8f, %.8f, %.8f]\n", dgdZ0, dgdZ1, dgdZ2);  //dbug

    // simple line search for delta
    double Z2[3];
    double new_delta[5] = {delta*.36, delta*.6, delta, delta*1.6, delta*2.6};
    int a, amin = -1;
    for (a = 0; a < 5; a++) {
      Z2[0] = Z[0] - new_delta[a]*dgdZ0;
      Z2[1] = Z[1] - new_delta[a]*dgdZ1;
      Z2[2] = Z[2] - new_delta[a]*dgdZ2;
      double g2 = bingham_dY_params_3d_slow_eval(err, Z2, dY);
      if (g2 < g) {
	g = g2;
	amin = a;
      }
    }

    if (amin >= 0) {
      delta = new_delta[amin];
      Z[0] -= delta*dgdZ0;
      Z[1] -= delta*dgdZ1;
      Z[2] -= delta*dgdZ2;
    }
    else
      delta *= .2;
  }

  *F = bingham_F_lookup_3d(Z);

  //dbug
  //double g = bingham_dY_params_3d_slow_eval(err, Z, dY);
  //printf("Z = [%.2f, %.2f, %.2f], g = %.10f\n", Z[0], Z[1], Z[2], g);
}


/*
 * Look up concentration params Z and normalization constant F given dY.
 */
void bingham_dY_params_3d(double *Z, double *F, double *dY)
{
  //dbug
  bingham_dY_params_3d_slow(Z, F, dY);

  /*
  if (dY_tree_3d == NULL)
    bingham_constants_init();

  int nn_index = kdtree_NN(dY_tree_3d, dY);

  int i = dY_indices_3d[nn_index][0];
  int j = dY_indices_3d[nn_index][1];
  int k = dY_indices_3d[nn_index][2];

  double r0 = bingham_table_range[i];
  double r1 = bingham_table_range[j];
  double r2 = bingham_table_range[k];

  Z[0] = -r0*r0;
  Z[1] = -r1*r1;
  Z[2] = -r2*r2;

  *F = bingham_F_table_3d[i][j][k];
  */
}


double bingham_F_table_get(int i, int j, int k)
{
  if (i >= j) {
    if (j >= k)
      return bingham_F_table_3d[i][j][k];
    else if (i >= k)
      return bingham_F_table_3d[i][k][j];
    else
      return bingham_F_table_3d[k][i][j];
  }
  else {
    if (k >= j)
      return bingham_F_table_3d[k][j][i];
    else if (k >= i)
      return bingham_F_table_3d[j][k][i];
    else
      return bingham_F_table_3d[j][i][k];
  }

  return 0.0;
}


double bingham_dF1_table_get(int i, int j, int k)
{
  if (i >= j) {
    if (j >= k)
      return bingham_dF1_table_3d[i][j][k];
    else if (i >= k)
      return bingham_dF1_table_3d[i][k][j];
    else
      return bingham_dF1_table_3d[k][i][j];
  }
  else {
    if (k >= j)
      return bingham_dF1_table_3d[k][j][i];
    else if (k >= i)
      return bingham_dF1_table_3d[j][k][i];
    else
      return bingham_dF1_table_3d[j][i][k];
  }

  return 0.0;
}


double bingham_dF2_table_get(int i, int j, int k)
{
  if (i >= j) {
    if (j >= k)
      return bingham_dF2_table_3d[i][j][k];
    else if (i >= k)
      return bingham_dF2_table_3d[i][k][j];
    else
      return bingham_dF2_table_3d[k][i][j];
  }
  else {
    if (k >= j)
      return bingham_dF2_table_3d[k][j][i];
    else if (k >= i)
      return bingham_dF2_table_3d[j][k][i];
    else
      return bingham_dF2_table_3d[j][i][k];
  }

  return 0.0;
}


double bingham_dF3_table_get(int i, int j, int k)
{
  if (i >= j) {
    if (j >= k)
      return bingham_dF3_table_3d[i][j][k];
    else if (i >= k)
      return bingham_dF3_table_3d[i][k][j];
    else
      return bingham_dF3_table_3d[k][i][j];
  }
  else {
    if (k >= j)
      return bingham_dF3_table_3d[k][j][i];
    else if (k >= i)
      return bingham_dF3_table_3d[j][k][i];
    else
      return bingham_dF3_table_3d[j][i][k];
  }

  return 0.0;
}


double bingham_dF_table_get(int a, int i, int j, int k)
{
  switch (a) {
  case 0:
    return bingham_dF1_table_get(i, j, k);
  case 1:
    return bingham_dF2_table_get(i, j, k);
  }
  return bingham_dF3_table_get(i, j, k);
}


/*
 * Look up normalization constant F given concentration params Z
 * via trilinear interpolation.
 */
double bingham_F_lookup_3d(double *Z)
{
  double y0 = sqrt(-Z[0]);
  double y1 = sqrt(-Z[1]);
  double y2 = sqrt(-Z[2]);

  int n = BINGHAM_TABLE_LENGTH;
  double ymin = bingham_table_range[0];
  double ymax = bingham_table_range[n-1];

  // get the Z-table cell between (i0,i1), (j0,j1), and (k0,k1)

  int i0, j0, k0, i1, j1, k1;

  if (y0 <= ymin)
    i1 = 1;
  else if (y0 >= ymax)
    i1 = n-1;
  else
    i1 = binary_search(y0, (double *)bingham_table_range, n);
  i0 = i1-1;

  if (y1 <= ymin)
    j1 = 1;
  else if (y1 >= ymax)
    j1 = n-1;
  else
    j1 = binary_search(y1, (double *)bingham_table_range, n);
  j0 = j1-1;

  if (y2 <= ymin)
    k1 = 1;
  else if (y2 >= ymax)
    k1 = n-1;
  else
    k1 = binary_search(y2, (double *)bingham_table_range, n);
  k0 = k1-1;

  // use trilinear interpolation between the 8 corners
  y0 -= bingham_table_range[i0];
  y1 -= bingham_table_range[j0];
  y2 -= bingham_table_range[k0];
  double d0 = bingham_table_range[i1] - bingham_table_range[i0];
  double d1 = bingham_table_range[j1] - bingham_table_range[j0];
  double d2 = bingham_table_range[k1] - bingham_table_range[k0];

  double F000 = bingham_F_table_get(i0, j0, k0);
  double F001 = bingham_F_table_get(i0, j0, k1);
  double F010 = bingham_F_table_get(i0, j1, k0);
  double F011 = bingham_F_table_get(i0, j1, k1);
  double F100 = bingham_F_table_get(i1, j0, k0);
  double F101 = bingham_F_table_get(i1, j0, k1);
  double F110 = bingham_F_table_get(i1, j1, k0);
  double F111 = bingham_F_table_get(i1, j1, k1);

  // interpolate over k
  double F00 = F000 + y2*(F001 - F000)/d2;
  double F01 = F010 + y2*(F011 - F010)/d2;
  double F10 = F100 + y2*(F101 - F100)/d2;
  double F11 = F110 + y2*(F111 - F110)/d2;

  // interpolate over j
  double F0 = F00 + y1*(F01 - F00)/d1;
  double F1 = F10 + y1*(F11 - F10)/d1;

  // interpolate over i
  double F = F0 + y0*(F1 - F0)/d0;

  return F;
}


/*
 * Look up partial derivatives of the normalization constant F
 * given concentration params Z via trilinear interpolation.
 */
void bingham_dF_lookup_3d(double *dF, double *Z)
{
  double y0 = sqrt(-Z[0]);
  double y1 = sqrt(-Z[1]);
  double y2 = sqrt(-Z[2]);

  int n = BINGHAM_TABLE_LENGTH;
  double ymin = bingham_table_range[0];
  double ymax = bingham_table_range[n-1];

  // get the Z-table cell between (i0,i1), (j0,j1), and (k0,k1)

  int i0, j0, k0, i1, j1, k1;

  if (y0 <= ymin)
    i1 = 1;
  else if (y0 >= ymax)
    i1 = n-1;
  else
    i1 = binary_search(y0, (double *)bingham_table_range, n);
  i0 = i1-1;

  if (y1 <= ymin)
    j1 = 1;
  else if (y1 >= ymax)
    j1 = n-1;
  else
    j1 = binary_search(y1, (double *)bingham_table_range, n);
  j0 = j1-1;

  if (y2 <= ymin)
    k1 = 1;
  else if (y2 >= ymax)
    k1 = n-1;
  else
    k1 = binary_search(y2, (double *)bingham_table_range, n);
  k0 = k1-1;

  // use trilinear interpolation between the 8 corners
  y0 -= bingham_table_range[i0];
  y1 -= bingham_table_range[j0];
  y2 -= bingham_table_range[k0];
  double d0 = bingham_table_range[i1] - bingham_table_range[i0];
  double d1 = bingham_table_range[j1] - bingham_table_range[j0];
  double d2 = bingham_table_range[k1] - bingham_table_range[k0];

  int a;
  for (a = 0; a < 3; a++) {
    double dF000 = bingham_dF_table_get(a, i0, j0, k0);
    double dF001 = bingham_dF_table_get(a, i0, j0, k1);
    double dF010 = bingham_dF_table_get(a, i0, j1, k0);
    double dF011 = bingham_dF_table_get(a, i0, j1, k1);
    double dF100 = bingham_dF_table_get(a, i1, j0, k0);
    double dF101 = bingham_dF_table_get(a, i1, j0, k1);
    double dF110 = bingham_dF_table_get(a, i1, j1, k0);
    double dF111 = bingham_dF_table_get(a, i1, j1, k1);

    // interpolate over k
    double dF00 = dF000 + y2*(dF001 - dF000)/d2;
    double dF01 = dF010 + y2*(dF011 - dF010)/d2;
    double dF10 = dF100 + y2*(dF101 - dF100)/d2;
    double dF11 = dF110 + y2*(dF111 - dF110)/d2;

    // interpolate over j
    double dF0 = dF00 + y1*(dF01 - dF00)/d1;
    double dF1 = dF10 + y1*(dF11 - dF10)/d1;

    // interpolate over i
    dF[a] = dF0 + y0*(dF1 - dF0)/d0;
  }
}



///////////////////////////////////////////////////////////////////
//                                                               //
//****************  Hypergeometric 1F1 Functions  ***************//
//                                                               //
///////////////////////////////////////////////////////////////////



//-------------------  1F1 Canonical form  -------------------//


/*
 * Computes the hypergeometric function 1F1(a;b;z) in canonical form (z > 0)
 */
static double compute_1F1_1d_canon(double a, double b, double z, int iter)
{
  //printf("compute_1F1_1d_canon(%f, %f, %f, %d)\n", a, b, z, iter);

  int i;
  double g, F = 0.0, logz = log(z);

  for (i = 0; i < iter; i++) {
    g = lgamma(i+a) - lgamma(i+b) + i*logz - lfact(i);
    if (i > z && exp(g) < EPSILON * F)  // exp(g) < 1e-8 * F
      break;
    F += exp(g);
  }

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the hypergeometric function 1F1(a;b;z1,z2) in canonical form (z1 > z2 > 0)
 */
static double compute_1F1_2d_canon(double a, double b, double z1, double z2, int iter)
{
  int i, j;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2);

  for (i = 0; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      g = lgamma(i+a) + lgamma(j+a) - lgamma(i+j+b) + i*logz1 + j*logz2 - lfact(i) - lfact(j);
      if ((i > z1 || j > z2) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	break;
      F += exp(g);
    }
  }

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the hypergeometric function 1F1(a;b;z1,z2,z3) in canonical form (z1 > z2 > z3 > 0)
 */
static double compute_1F1_3d_canon(double a, double b, double z1, double z2, double z3, int iter)
{
  int i, j, k;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2), logz3 = log(z3);

  for (i = 0; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      for (k = 0; k < iter; k++) {
	g = lgamma(i+a) + lgamma(j+a) + lgamma(k+a) - lgamma(i+j+k+b) + i*logz1 + j*logz2 + k*logz3 - lfact(i) - lfact(j) - lfact(k);
	if ((i > z1 || j > z2 || k > z3) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	  break;
	F += exp(g);
      }
    }
  }

  return 2*sqrt(M_PI)*F;
}



//-------------------  1F1 General form  -------------------//


/*
 * Computes the hypergeometric function 1F1(a;b;z) with a = 1/2, b = (dim+1)/2
 */
static double compute_1F1_1d(int dim, double z, int iter)
{
  if (fabs(z) < EPSILON)  // uniform
    return surface_area_sphere(dim);

  if (z < 0)
    return exp(z)*compute_1F1_1d(dim, -z, iter);

  return compute_1F1_1d_canon(.5, .5*(dim+1), z, iter);
}


/*
 * Computes the hypergeometric function 1F1(a;b;z1,z2) with a = 1/2, b = (dim+1)/2
 */
static double compute_1F1_2d(int dim, double z1, double z2, int iter)
{
  if (fabs(z1) < EPSILON)  // uniform
    return surface_area_sphere(dim);

  if (fabs(z2) < EPSILON)  // z2 = 0
    return sqrt(M_PI)*compute_1F1_1d(dim, z1, iter);

  if (z1 < 0)
    return exp(z1)*compute_1F1_2d(dim, -z1, z2-z1, iter);

  return compute_1F1_2d_canon(.5, .5*(dim+1), z1, z2, iter);
}


/*
 * Computes the hypergeometric function 1F1(a;b;z1,z2,z3) with a = 1/2, b = (dim+1)/2
 */
static double compute_1F1_3d(int dim, double z1, double z2, double z3, int iter)
{
  if (fabs(z1) < EPSILON)  // uniform
    return surface_area_sphere(dim);

  if (fabs(z3) < EPSILON)  // z3 = 0
    return sqrt(M_PI)*compute_1F1_2d(dim, z1, z2, iter);

  if (z1 < 0)
    return exp(z1)*compute_1F1_3d(dim, -z1, z3-z1, z2-z1, iter);

  return compute_1F1_3d_canon(.5, .5*(dim+1), z1, z2, z3, iter);
}



//--------------  1F1 Partial derivatives (general form)  --------------//


//.....(((  1D  ))).....

/*
 * Computes the partial derivative w.r.t. z of the hypergeometric
 * function 1F1(a;b;z) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz_1d(int dim, double z, int iter)
{
  //printf("compute_d1F1_dz1_1d(%d, %f, %d)\n", dim, z, iter);

  if (fabs(z) < EPSILON)  // uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (z < 0)
    return compute_1F1_1d(dim, z, iter) - exp(z)*compute_d1F1_dz_1d(dim, -z, iter);

  int i;
  double g, F = 0.0, logz = log(z);
  double a = .5, b = .5*(dim+1);

  for (i = 1; i < iter; i++) {
    g = lgamma(i+a) - lgamma(i+b) + (i-1)*logz - lfact(i-1);
    if (i > z && exp(g) < EPSILON * F)  // exp(g) < 2e-9
      break;
    F += exp(g);
  }

  //printf("  --> %f\n", 2*sqrt(M_PI)*F);

  return 2*sqrt(M_PI)*F;
}


//.....(((  2D  ))).....

static double compute_d1F1_dz2_2d(int dim, double z1, double z2, int iter);


/*
 * Computes the partial derivative w.r.t. z1 of the hypergeometric
 * function 1F1(a;b;z1,z2) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz1_2d(int dim, double z1, double z2, int iter)
{
  //printf("compute_d1F1_dz1_2d(%d, %f, %f, %d)\n", dim, z1, z2, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z2) < EPSILON) {  // z2 = 0
    double retval = sqrt(M_PI)*compute_d1F1_dz_1d(dim, z1, iter);
    //printf("  --> %f\n", retval);
    return retval;
  }

  if (z1 < 0)
    return compute_1F1_2d(dim, z1, z2, iter) -
      exp(z1)*(compute_d1F1_dz1_2d(dim, -z1, z2-z1, iter) +
	       compute_d1F1_dz2_2d(dim, -z1, z2-z1, iter));

  int i, j;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2);
  double a = .5, b = .5*(dim+1);

  for (i = 1; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      g = lgamma(i+a) + lgamma(j+a) - lgamma(i+j+b) + (i-1)*logz1 + j*logz2 - lfact(i-1) - lfact(j);
      if ((i > z1 || j > z2) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	break;
      F += exp(g);
    }
  }

  //printf("  --> %f\n", 2*sqrt(M_PI)*F);

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the partial derivative w.r.t. z2 of the hypergeometric
 * function 1F1(a;b;z1,z2) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz2_2d(int dim, double z1, double z2, int iter)
{
  //printf("compute_d1F1_dz2_2d(%d, %f, %f, %d)\n", dim, z1, z2, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z2) < EPSILON)  // z2 = 0
    return .5*sqrt(M_PI)*compute_1F1_1d(dim+2, z1, iter);

  if (z1 < 0)
    return exp(z1)*compute_d1F1_dz2_2d(dim, -z1, z2-z1, iter);

  int i, j;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2);
  double a = .5, b = .5*(dim+1);

  for (i = 0; i < iter; i++) {
    for (j = 1; j < iter; j++) {
      g = lgamma(i+a) + lgamma(j+a) - lgamma(i+j+b) + i*logz1 + (j-1)*logz2 - lfact(i) - lfact(j-1);
      if ((i > z1 || j > z2) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	break;
      F += exp(g);
    }
  }

  return 2*sqrt(M_PI)*F;
}


//.....(((  3D  ))).....

static double compute_d1F1_dz2_3d(int dim, double z1, double z2, double z3, int iter);
static double compute_d1F1_dz3_3d(int dim, double z1, double z2, double z3, int iter);


/*
 * Computes the partial derivative w.r.t. z1 of the hypergeometric
 * function 1F1(a;b;z1,z2,z3) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz1_3d(int dim, double z1, double z2, double z3, int iter)
{
  //printf("compute_d1F1_dz1_3d(%d, %f, %f, %f, %d)\n", dim, z1, z2, z3, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = z3 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z3) < EPSILON) {  // z3 = 0
    double retval = sqrt(M_PI)*compute_d1F1_dz1_2d(dim, z1, z2, iter);
    //printf("  --> %f\n", retval);
    return retval;
  }

  if (z1 < 0)
    return compute_1F1_3d(dim, z1, z2, z3, iter) -
      exp(z1)*(compute_d1F1_dz1_3d(dim, -z1, z3-z1, z2-z1, iter) +
	       compute_d1F1_dz2_3d(dim, -z1, z3-z1, z2-z1, iter) +
	       compute_d1F1_dz3_3d(dim, -z1, z3-z1, z2-z1, iter));

  int i, j, k;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2), logz3 = log(z3);
  double a = .5, b = .5*(dim+1);

  for (i = 1; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      for (k = 0; k < iter; k++) {
	g = lgamma(i+a) + lgamma(j+a) + lgamma(k+a) - lgamma(i+j+k+b) + (i-1)*logz1 + j*logz2 + k*logz3 - lfact(i-1) - lfact(j) - lfact(k);
	if ((i > z1 || j > z2 || k > z3) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	  break;
	F += exp(g);
      }
    }
  }

  //printf("  --> %f\n", 2*sqrt(M_PI)*F);

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the partial derivative w.r.t. z2 of the hypergeometric
 * function 1F1(a;b;z1,z2,z3) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz2_3d(int dim, double z1, double z2, double z3, int iter)
{
  //printf("compute_d1F1_dz2_3d(%d, %f, %f, %f, %d)\n", dim, z1, z2, z3, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = z3 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z3) < EPSILON)  // z3 = 0
    return sqrt(M_PI)*compute_d1F1_dz2_2d(dim, z1, z2, iter);

  if (z1 < 0)
    return exp(z1)*compute_d1F1_dz3_3d(dim, -z1, z3-z1, z2-z1, iter);

  int i, j, k;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2), logz3 = log(z3);
  double a = .5, b = .5*(dim+1);

  for (i = 0; i < iter; i++) {
    for (j = 1; j < iter; j++) {
      for (k = 0; k < iter; k++) {
	g = lgamma(i+a) + lgamma(j+a) + lgamma(k+a) - lgamma(i+j+k+b) + i*logz1 + (j-1)*logz2 + k*logz3 - lfact(i) - lfact(j-1) - lfact(k);
	if ((i > z1 || j > z2 || k > z3) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	  break;
	F += exp(g);
      }
    }
  }

  return 2*sqrt(M_PI)*F;
}


/*
 * Computes the partial derivative w.r.t. z3 of the hypergeometric
 * function 1F1(a;b;z1,z2,z3) with a = 1/2, b = (dim+1)/2
 */
static double compute_d1F1_dz3_3d(int dim, double z1, double z2, double z3, int iter)
{
  //printf("compute_d1F1_dz3_3d(%d, %f, %f, %f, %d)\n", dim, z1, z2, z3, iter);

  if (fabs(z1) < EPSILON)  // z1 = z2 = z3 = 0 --> uniform
    return surface_area_sphere(dim)/(double)(dim+1);

  if (fabs(z3) < EPSILON)  // z3 = 0
    return .5*sqrt(M_PI)*compute_1F1_2d(dim+2, z1, z2, iter);

  if (z1 < 0)
    return exp(z1)*compute_d1F1_dz2_3d(dim, -z1, z3-z1, z2-z1, iter);

  int i, j, k;
  double g, F = 0.0, logz1 = log(z1), logz2 = log(z2), logz3 = log(z3);
  double a = .5, b = .5*(dim+1);

  for (i = 0; i < iter; i++) {
    for (j = 0; j < iter; j++) {
      for (k = 1; k < iter; k++) {
	g = lgamma(i+a) + lgamma(j+a) + lgamma(k+a) - lgamma(i+j+k+b) + i*logz1 + j*logz2 + (k-1)*logz3 - lfact(i) - lfact(j) - lfact(k-1);
	if ((i > z1 || j > z2 || k > z3) && exp(g) < EPSILON * F)  // exp(g) < 2e-9
	  break;
	F += exp(g);
      }
    }
  }

  return 2*sqrt(M_PI)*F;
}






//---------------- Bingham normalizing constants F(z) and partial derivatives ------------------//

double bingham_F_1d(double z)
{
  int iter = MAX((int)fabs(z)*ITERATION_MULT, MIN_ITERATIONS);
  return compute_1F1_1d(1, z, iter);
}

double bingham_dF_1d(double z)
{
  int iter = MAX((int)fabs(z)*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz_1d(1, z, iter);
}

double bingham_F_2d(double z1, double z2)
{
  int iter = MAX((int)MAX(fabs(z1), fabs(z2))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_1F1_2d(2, z1, z2, iter);
}

double bingham_dF1_2d(double z1, double z2)
{
  int iter = MAX((int)MAX(fabs(z1), fabs(z2))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz1_2d(2, z1, z2, iter);
}

double bingham_dF2_2d(double z1, double z2)
{
  int iter = MAX((int)MAX(fabs(z1), fabs(z2))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz2_2d(2, z1, z2, iter);
}

double bingham_F_3d(double z1, double z2, double z3)
{
  int iter = MAX((int)MAX(MAX(fabs(z1), fabs(z2)), fabs(z3))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_1F1_3d(3, z1, z2, z3, iter);
}

double bingham_dF1_3d(double z1, double z2, double z3)
{
  int iter = MAX((int)MAX(MAX(fabs(z1), fabs(z2)), fabs(z3))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz1_3d(3, z1, z2, z3, iter);
}

double bingham_dF2_3d(double z1, double z2, double z3)
{
  int iter = MAX((int)MAX(MAX(fabs(z1), fabs(z2)), fabs(z3))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz2_3d(3, z1, z2, z3, iter);
}

double bingham_dF3_3d(double z1, double z2, double z3)
{
  int iter = MAX((int)MAX(MAX(fabs(z1), fabs(z2)), fabs(z3))*ITERATION_MULT, MIN_ITERATIONS);
  return compute_d1F1_dz3_3d(3, z1, z2, z3, iter);
}





//----------------- Bingham F(z) "compute_all" tools --------------------//




void compute_all_bingham_F_2d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step)
{
  double z1, z2;

  printf("F = [ ...\n");
  for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
    for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
      printf("%f ", z1 <= z2 ? bingham_F_2d(z1, z2) : 0.0);
    printf("; ...\n");
  }
  printf("];\n\n");
}

void compute_all_bingham_dF1_2d(double z1_min, double z1_max, double z1_step,
				double z2_min, double z2_max, double z2_step)
{
  double z1, z2;

  printf("dF1 = [ ...\n");
  for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
    for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
      printf("%f ", z1 <= z2 ? bingham_dF1_2d(z1, z2) : 0.0);
    printf("; ...\n");
  }
  printf("];\n\n");
}

void compute_all_bingham_dF2_2d(double z1_min, double z1_max, double z1_step,
			       double z2_min, double z2_max, double z2_step)
{
  double z1, z2;

  printf("dF2 = [ ...\n");
  for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
    for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
      printf("%f ", z1 <= z2 ? bingham_dF2_2d(z1, z2) : 0.0);
    printf("; ...\n");
  }
  printf("];\n\n");
}

void compute_all_bingham_F_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step)
{
  double z1, z2, z3;
  int k=0;

  printf("F = [];\n\n");
  for (z3 = z3_min; z3 <= z3_max; z3 += z3_step) {
    k++;
    printf("F(:,:,%d) = [ ...\n", k);
    for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
      for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_F_3d(z1, z2, z3) : 0.0);
      printf("; ...\n");
    }
    printf("];\n\n");
  }
}

void compute_all_bingham_dF1_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step)
{
  double z1, z2, z3;
  int k=0;

  printf("dF1 = [];\n\n");
  for (z3 = z3_min; z3 <= z3_max; z3 += z3_step) {
    k++;
    printf("dF1(:,:,%d) = [ ...\n", k);
    for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
      for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF1_3d(z1, z2, z3) : 0.0);
      printf("; ...\n");
    }
    printf("];\n\n");
  }
}

void compute_all_bingham_dF2_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step)
{
  double z1, z2, z3;
  int k=0;

  printf("dF2 = [];\n\n");
  for (z3 = z3_min; z3 <= z3_max; z3 += z3_step) {
    k++;
    printf("dF2(:,:,%d) = [ ...\n", k);
    for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
      for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF2_3d(z1, z2, z3) : 0.0);
      printf("; ...\n");
    }
    printf("];\n\n");
  }
}

void compute_all_bingham_dF3_3d(double z1_min, double z1_max, double z1_step,
			      double z2_min, double z2_max, double z2_step,
			      double z3_min, double z3_max, double z3_step)
{
  double z1, z2, z3;
  int k=0;

  printf("dF3 = [];\n\n");
  for (z3 = z3_min; z3 <= z3_max; z3 += z3_step) {
    k++;
    printf("dF3(:,:,%d) = [ ...\n", k);
    for (z1 = z1_min; z1 <= z1_max; z1 += z1_step) {
      for (z2 = z2_min; z2 <= z2_max; z2 += z2_step)
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF3_3d(z1, z2, z3) : 0.0);
      printf("; ...\n");
    }
    printf("];\n\n");
  }
}


void compute_range_bingham_F_2d(double *y, int n)
{
  int i, j;
  double z1, z2;

  printf("F = [ ...\n");
  fprintf(stderr, "F");
  for (i = 0; i < n; i++) {
    fprintf(stderr, ".");
    fflush(stderr);
    z1 = -y[i]*y[i];
    for (j = 0; j < n; j++) {
      z2 = -y[j]*y[j];
      printf("%f ", z1 <= z2 ? bingham_F_2d(z1, z2) : 0.0);
    }
    printf("; ...\n");
  }
  printf("];\n\n");
  fprintf(stderr, "\n");
}

void compute_range_bingham_dF1_2d(double *y, int n)
{
  int i, j;
  double z1, z2;

  printf("dF1 = [ ...\n");
  fprintf(stderr, "dF1");
  for (i = 0; i < n; i++) {
    fprintf(stderr, ".");
    fflush(stderr);
    z1 = -y[i]*y[i];
    for (j = 0; j < n; j++) {
      z2 = -y[j]*y[j];
      printf("%f ", z1 <= z2 ? bingham_dF1_2d(z1, z2) : 0.0);
    }
    printf("; ...\n");
  }
  printf("];\n\n");
  fprintf(stderr, "\n");
}

void compute_range_bingham_dF2_2d(double *y, int n)
{
  int i, j;
  double z1, z2;

  printf("dF2 = [ ...\n");
  fprintf(stderr, "dF2");
  for (i = 0; i < n; i++) {
    fprintf(stderr, ".");
    fflush(stderr);
    z1 = -y[i]*y[i];
    for (j = 0; j < n; j++) {
      z2 = -y[j]*y[j];
      printf("%f ", z1 <= z2 ? bingham_dF1_2d(z1, z2) : 0.0);
    }
    printf("; ...\n");
  }
  printf("];\n\n");
  fprintf(stderr, "\n");
}


void compute_range_bingham_F_3d(double *y, int n, int k0, int k1)
{
  int i, j, k;
  double z1, z2, z3;

  //printf("F = [];\n\n");
  fprintf(stderr, "F");
  for (k = k0; k < k1; k++) {
    z3 = -y[k]*y[k];
    printf("F(:,:,%d) = [ ...\n", k);
    for (i = 0; i < n; i++) {
      fprintf(stderr, ".");
      fflush(stderr);
      z1 = -y[i]*y[i];
      for (j = 0; j < n; j++) {
	z2 = -y[j]*y[j];
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_F_3d(z1, z2, z3) : 0.0);
      }
      printf("; ...\n");
    }
    printf("];\n\n");
    fprintf(stderr, "\n");
  }
}

void compute_range_bingham_dF1_3d(double *y, int n, int k0, int k1)
{
  int i, j, k;
  double z1, z2, z3;

  //printf("dF1 = [];\n\n");
  fprintf(stderr, "dF1");
  for (k = k0; k < k1; k++) {
    z3 = -y[k]*y[k];
    printf("F(:,:,%d) = [ ...\n", k);
    for (i = 0; i < n; i++) {
      fprintf(stderr, ".");
      fflush(stderr);
      z1 = -y[i]*y[i];
      for (j = 0; j < n; j++) {
	z2 = -y[j]*y[j];
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF1_3d(z1, z2, z3) : 0.0);
      }
      printf("; ...\n");
    }
    printf("];\n\n");
    fprintf(stderr, "\n");
  }
}

void compute_range_bingham_dF2_3d(double *y, int n, int k0, int k1)
{
  int i, j, k;
  double z1, z2, z3;

  //printf("dF2 = [];\n\n");
  fprintf(stderr, "dF2");
  for (k = k0; k < k1; k++) {
    z3 = -y[k]*y[k];
    printf("F(:,:,%d) = [ ...\n", k);
    for (i = 0; i < n; i++) {
      fprintf(stderr, ".");
      fflush(stderr);
      z1 = -y[i]*y[i];
      for (j = 0; j < n; j++) {
	z2 = -y[j]*y[j];
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF2_3d(z1, z2, z3) : 0.0);
      }
      printf("; ...\n");
    }
    printf("];\n\n");
    fprintf(stderr, "\n");
  }
}

void compute_range_bingham_dF3_3d(double *y, int n, int k0, int k1)
{
  int i, j, k;
  double z1, z2, z3;

  //printf("dF3 = [];\n\n");
  fprintf(stderr, "dF3");
  for (k = k0; k < k1; k++) {
    z3 = -y[k]*y[k];
    printf("F(:,:,%d) = [ ...\n", k);
    for (i = 0; i < n; i++) {
      fprintf(stderr, ".");
      fflush(stderr);
      z1 = -y[i]*y[i];
      for (j = 0; j < n; j++) {
	z2 = -y[j]*y[j];
	printf("%f ", z1 <= z2 && z2 <= z3 ? bingham_dF3_3d(z1, z2, z3) : 0.0);
      }
      printf("; ...\n");
    }
    printf("];\n\n");
    fprintf(stderr, "\n");
  }
}

