#ifndef BINGHAM_GAUSS_MIX_H
#define BINGHAM_GAUSS_MIX_H


#ifdef __cplusplus
extern "C" {
#endif 


#include "bingham.h"



  typedef struct {
    double *weights;  // mixing coeffs
    double **means;   // means
    double ***covs;   // covariances
    int n;            // num components
    int d;            // dimensions
  } gauss_mix_t;
  

  gauss_mix_t *new_gauss_mix(int dims, int k);
  gauss_mix_t *gauss_mix_clone(gauss_mix_t *gm);
  void free_gauss_mix(gauss_mix_t *gm);

  gauss_mix_t *fit_gauss_mix(double **X, int npoints, int dims, double *w, unsigned int kmin, unsigned int kmax, double regularize, double th);




#ifdef __cplusplus
}
#endif 


#endif
