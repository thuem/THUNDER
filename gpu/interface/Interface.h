#ifndef INTERFACE_H
#define INTERFACE_H

#include "mpi.h"
#include "Image.h"
#include "ImageFunctions.h"
#include "Particle.h"
#include "Volume.h"
#include "Symmetry.h"
#include "Database.h"
#include "Typedef.h"

#define RECONSTRUCTOR_NORMALISE_T_F
#define RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
#define RECONSTRUCTOR_WIENER_FILTER_FSC

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
#define RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
#endif

void ExpectPrecal(vector<CTFAttr>& ctfAttr,
                  double* def,
                  double* k1,
                  double* k2,
                  const int *iCol, 
                  const int *iRow,
                  int idim, 
                  int npxl,
                  int imgNum);

void ExpectGlobal2D(Complex* vol,
                    Complex* datP,
                    double* ctfP,
                    double* sigRcpP,
                    double* trans,
                    double* wC,
                    double* wR,
                    double* wT,
                    double* rot,
                    const int *iCol, 
                    const int *iRow,
                    int nK,
                    int nR,
                    int nT,
                    int pf,
                    int interp,
                    int idim, 
                    int vdim, 
                    int npxl,
                    int imgNum);

void ExpectGlobal3D(Complex* vol,
                    Complex* datP,
                    double* ctfP,
                    double* sigRcpP,
                    double* trans,
                    double* wC,
                    double* wR,
                    double* wT,
                    double* rot,
                    const int *iCol, 
                    const int *iRow,
                    int nK,
                    int nR,
                    int nT,
                    int pf,
                    int interp,
                    int idim, 
                    int vdim, 
                    int npxl,
                    int imgNum);

void InsertF(Volume& F3D,
             Volume& T3D,
             MPI_Comm& hemi,
             Complex* datP,
             double* ctfP,
             double* sigRcpP,
             CTFAttr* ctfaData,
             double *offS,
             double *w,
             double *nR,
             double *nT,
             double *nD,
             const int *iCol,
             const int *iRow, 
             double pixelSize,
             bool cSearch,
             int rSize,
             int tSize,
             int dSize,
             int npxl,
             int mReco,
             int idim,
             int imgNum);

void ExposePT(Volume& T3D,
	          const Symmetry& sym,
	          int maxRadius,
	          int pf,
	          vec FSC,
	          bool joinHalf,
              const int wienerF);

void ExposeWT(Volume& C3D,
              Volume& T3D,
              Volume& W3D,
              TabFunction& kernelRL,
              int maxRadius,
              int pf,
              double a,
              double alpha,
              int maxIter,
              int minIter,
              int size);

void ExposeWT(Volume& T3D,
              Volume& W3D,
              int maxRadius,
              int pf);

void ExposePF(Volume& F3D,
              Volume& W3D,
              const Symmetry& sym,
              int maxRadius,
              int pf,
              double sf,
              int size);

void ExposeCorrF(Volume& padDst,
                 Volume& dst,
                 double a,
                 double alpha,
                 int pf,
                 int size);

void ExposeCorrF(Volume& F3D,
                 Volume& dst,
                 double nf,
                 double a,
                 double alpha,
                 int pf,
                 int size);

#endif
