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
#include "Config.cuh"

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
                  RFLOAT* def,
                  RFLOAT* k1,
                  RFLOAT* k2,
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
                    RFLOAT* ctfP,
                    RFLOAT* sigRcpP,
                    double* trans,
                    RFLOAT* wC,
                    RFLOAT* wR,
                    RFLOAT* wT,
                    double* pR,
                    double* pT,
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
                    RFLOAT* ctfP,
                    RFLOAT* sigRcpP,
                    double* trans,
                    RFLOAT* wC,
                    RFLOAT* wR,
                    RFLOAT* wT,
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
void expectRotran(Complex* traP,
                  double* trans,
                  double* rot,
                  double* rotMat,
                  const int *iCol,
                  const int *iRow,
                  int nR,
                  int nT,
                  int idim,
                  int npxl);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectProject(Complex* volume,
                   Complex* rotP,
                   double* rotMat,
                   const int *iCol,
                   const int *iRow,
                   int nR,
                   int pf,
                   int interp,
                   int idim,
                   int vdim,
                   int npxl);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectGlobal3D(Complex* rotP,
                    Complex* traP,
                    Complex* datP,
                    RFLOAT* ctfP,
                    RFLOAT* sigRcpP,
                    RFLOAT* wC,
                    RFLOAT* wR,
                    RFLOAT* wT,
                    double* pR,
                    double* pT,
                    RFLOAT* baseL,
                    int kIdx,
                    int nK,
                    int nR,
                    int nT,
                    int idim,
                    int npxl,
                    int imgNum);

/**
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void InsertI2D(Complex *F2D,
               RFLOAT *T2D,
               double *O2D,
               int *counter,
               MPI_Comm& hemi,
               MPI_Comm& slav,
               Complex *datP,
               RFLOAT *ctfP,
               RFLOAT *sigRcpP,
               RFLOAT *w,
               double *offS,
               int *nC,
               double *nR,
               double *nT,
               double *nD,
               CTFAttr *ctfaData,
               const int *iCol,
               const int *iRow,
               RFLOAT pixelSize,
               bool cSearch,
               int nk,
               int opf,
               int npxl,
               int mReco,
               int idim,
               int vdim,
               int imgNum);

/**
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void InsertFT(Complex *F3D,
              RFLOAT *T3D,
              double *O3D,
              int *counter,
              MPI_Comm& hemi,
              MPI_Comm& slav,
              Complex *datP,
              RFLOAT *ctfP,
              RFLOAT *sigRcpP,
              CTFAttr *ctfaData,
              double *offS,
              RFLOAT *w,
              double *nR,
              double *nT,
              double *nD,
              int *nC,
              const int *iCol,
              const int *iRow,
              RFLOAT pixelSize,
              bool cSearch,
              int opf,
              int npxl,
              int mReco,
              int imgNum,
              int idim,
              int vdim);

/**
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void InsertFT(Complex *F3D,
              RFLOAT *T3D,
              double *O3D,
              int *counter,
              MPI_Comm& hemi,
              MPI_Comm& slav,
              Complex *datP,
              RFLOAT *ctfP,
              RFLOAT *sigRcpP,
              CTFAttr *ctfaData,
              double *offS,
              RFLOAT *w,
              double *nR,
              double *nT,
              double *nD,
              const int *iCol,
              const int *iRow,
              RFLOAT pixelSize,
              bool cSearch,
              int opf,
              int npxl,
              int mReco,
              int imgNum,
              int idim,
              int vdim);

/**
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void InsertF(Complex *F3D,
             RFLOAT *T3D,
             MPI_Comm& hemi,
             Complex *datP,
             RFLOAT *ctfP,
             RFLOAT *sigRcpP,
             CTFAttr *ctfaData,
             double *offS,
             RFLOAT *w,
             double *nR,
             double *nT,
             double *nD,
             const int *iCol,
             const int *iRow,
             RFLOAT pixelSize,
             bool cSearch,
             int opf,
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
void PrepareTF(int gpuIdx,
               Complex *F3D,
               RFLOAT *T3D,
               const double *symMat,
               const RFLOAT sf,
               const int nSymmetryElement,
               const int interp,
               const int dim,
               const int r);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CalculateT2D(int gpuIdx,
                  RFLOAT *T2D,
                  RFLOAT *FSC,
                  const int fscMatsize,
                  const bool joinHalf,
                  const int maxRadius,
                  const int wienerF,
                  const int dim,
                  const int pf);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CalculateT(int gpuIdx,
                RFLOAT *T3D,
                RFLOAT *FSC,
                const int fscMatsize,
                const bool joinHalf,
                const int maxRadius,
                const int wienerF,
                const int dim,
                const int pf);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW2D(int gpuIdx,
                  RFLOAT *T2D,
                  RFLOAT *W2D,
                  const int dim,
                  const int r);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW(int gpuIdx,
                RFLOAT *T3D,
                RFLOAT *W3D,
                const int dim,
                const int r);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW2D(int gpuIdx,
                  RFLOAT *T2D,
                  RFLOAT *W2D,
                  RFLOAT *tabdata,
                  RFLOAT begin,
                  RFLOAT end,
                  RFLOAT step,
                  int tabsize, 
                  const int dim,
                  const int r,
                  const RFLOAT nf,
                  const int maxIter,
                  const int minIter,
                  const int padSize);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW(int gpuIdx,
                RFLOAT *T3D,
                RFLOAT *W3D,
                RFLOAT *tabdata,
                RFLOAT begin,
                RFLOAT end,
                RFLOAT step,
                int tabsize,
                const int dim,
                const int r,
                const RFLOAT nf,
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
void CalculateF2D(int gpuIdx,
                  Complex *padDst,
                  Complex *F2D,
                  RFLOAT *padDstR,
                  RFLOAT *W2D,
                  const int r,
                  const int pdim,
                  const int fdim);

/**
 * @brief
 *
 * @param
 * @param
 * @param
 */
void CalculateF(int gpuIdx,
                Complex *padDst,
                Complex *F3D,
                RFLOAT *padDstR,
                RFLOAT *W3D,
                const int r,
                const int pdim,
                const int fdim);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CorrSoftMaskF2D(int gpuIdx,
                     RFLOAT *dstI,
                     Complex *dst,
                     RFLOAT *mkbRL,
                     RFLOAT nf,
                     const int dim);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CorrSoftMaskF(int gpuIdx,
                   Complex *dst,
                   RFLOAT *dstN,
                   RFLOAT *mkbRL,
                   RFLOAT nf,
                   const int dim);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void TranslateI2D(int gpuIdx,
                  Complex* src,
                  RFLOAT ox,
                  RFLOAT oy,
                  int r,
                  int dim);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void TranslateI(int gpuIdx,
                Complex* ref,
                RFLOAT ox,
                RFLOAT oy,
                RFLOAT oz,
                int r,
                int dim);

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CorrSoftMaskF(int gpuIdx,
                   RFLOAT *dst,
                   RFLOAT *mkbRL,
                   RFLOAT nf,
                   const int dim,
                   const int size,
                   const int edgeWidth);

///////////////////////////////////////////////////////////////

} // end namespace cunthunder

///////////////////////////////////////////////////////////////
#endif
