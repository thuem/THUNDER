/**************************************************************
 * FileName: cuthunder.cuh
 * Author  : Kunpeng WANG, Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#ifndef CUTHUNDER_H
#define CUTHUNDER_H

#include "easylogging++.h"
#include "cufft.h"
#include "nccl.h"

#include <mpi.h>
#include <vector>
#include <unistd.h>

namespace cuthunder {

using std::vector;

///////////////////////////////////////////////////////////////

class Complex;
class CTFAttr;


/* Volume create kind */
typedef enum {
    DEFAULT     = 0,
    HOST_ONLY   = 1,
    DEVICE_ONLY = 2,
    HD_SYNC     = 4,
    HD_BOTH     = HOST_ONLY | DEVICE_ONLY
} VolCrtKind;

/**
 * Test routines.
 *
 * ...
 */
void addTest();

/**
 * @brief Pre-calculation in expectation.
 *
 * @param
 * @param
 */
void expectPrecal(vector<CTFAttr*>& ctfaData,
                  double* def,
                  double* k1,
                  double* k2,
                  const int *iCol,
                  const int *iRow,
                  int idim,
                  int npxl,
                  int imgNum);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectGlobal2D(Complex* volume,
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

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectGlobal3D(Complex* volume,
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

/**
 * @brief Insert a number of images into model F.
 *
 * @param
 * @param
 * @param
 */
void InsertF(Complex *F3D,
             double *T3D,
             MPI_Comm& hemi,
             Complex *datP,
             double *ctfP,
             double *sigRcpP,
             CTFAttr *ctfaData,
             double *offS,
             double *w,
             double *nR,
             double *nT,
             double *nD,
             const int *iCol,
             const int *iRow,
             double pixelSize,
             bool cSearch,
             int npxl,
             int rSize,
             int tSize,
             int dSize,
             int mReco,
             int imgNum,
             int idim,
             int vdim);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void PrepareT(double *T3D,
              const int dim,
              double *FSC,
              const int fscMatsize,
              const double *symMat,
              const int nSymmetryElement,
              const int interp,
              const bool joinHalf,
              const int maxRadius,
              const int pf,
              const int wienerF,
              const double sf);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW(double *T3D,
                double *W3D,
                const int dim,
                const int r);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW(Complex *C3D,
                double *T3D,
                double *W3D,
                double *tabdata,
                double begin,
                double end,
                double step,
                int tabsize,
                const int dim,
                const int r,
                const double nf,
                const int maxIter,
                const int minIter,
                const int padSize);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void PrepareF(Complex *F3D,
              double *W3D,
              const double sf,
              const int nSymmetryElement,
              const double *symMat,
              const int interp,
              const int maxRadius,
              const int edgeWidth,
              const int pf,
              const int dim,
              const int size);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CorrSoftMaskF(double *dst,
                   double *mkbRL,
                   const int dim);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CorrSoftMaskF(double *dst,
                   double *mkbRL,
                   double nf,
                   const int dim,
                   const int size,
                   const int edgeWidth);


///////////////////////////////////////////////////////////////

} // end namespace cunthunder

///////////////////////////////////////////////////////////////
#endif
