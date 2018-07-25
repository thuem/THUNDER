/**************************************************************
 * FileName: cuthunder.h
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

#include "Config.h"
#include "Macro.h"
#include "Precision.h"

#include "ManagedArrayTexture.h"
#include "ManagedCalPoint.h"

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
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void getAviDevice(vector<int>& gpus);

void __host__checkHardware(int& nGPU,
                           vector<int>& iGPU);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectPreidx(int gpuIdx,
                  int** deviCol,
                  int** deviRow,
                  int* iCol,
                  int* iRow,
                  int npxl);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectPrefre(int gpuIdx,
                  RFLOAT** devfreQ,
                  RFLOAT* freQ,
                  int npxl);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalIn(int gpuIdx,
                   Complex** devdatP,
                   RFLOAT** devctfP,
                   RFLOAT** devdefO,
                   RFLOAT** devsigP,
                   int npxl,
                   int cpyNum,
                   int cSearch);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalP(int gpuIdx,
                  Complex* devdatP,
                  RFLOAT* devctfP,
                  RFLOAT* devdefO,
                  RFLOAT* devsigP,
                  Complex* datP,
                  RFLOAT* ctfP,
                  RFLOAT* defO,
                  RFLOAT* sigRcpP,
                  int threadId,
                  int imgId,
                  int npxl,
                  int cSearch);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalV2D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int dimSize);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalV3D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int vdim);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalHostA(int gpuIdx,
                      RFLOAT** wC,
                      RFLOAT** wR,
                      RFLOAT** wT,
                      RFLOAT** wD,
                      double** oldR,
                      double** oldT,
                      double** oldD,
                      double** trans,
                      double** rot,
                      double** dpara,
                      int mR,
                      int mT,
                      int mD,
                      int cSearch);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalRTD(int gpuIdx,
                    ManagedCalPoint* mcp,
                    double* oldR,
                    double* oldT,
                    double* oldD,
                    double* trans,
                    double* rot,
                    double* dpara);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalPreI2D(int gpuIdx,
                       int datShift,
                       ManagedArrayTexture* mgr,
                       ManagedCalPoint* mcp,
                       RFLOAT* devdefO,
                       RFLOAT* devfreQ,
                       int *deviCol,
                       int *deviRow,
                       RFLOAT phaseShift,
                       RFLOAT conT,
                       RFLOAT k1,
                       RFLOAT k2,
                       int pf,
                       int idim,
                       int vdim,
                       int npxl,
                       int interp);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalPreI3D(int gpuIdx,
                       int datShift,
                       ManagedArrayTexture* mgr,
                       ManagedCalPoint* mcp,
                       RFLOAT* devdefO,
                       RFLOAT* devfreQ,
                       int *deviCol,
                       int *deviRow,
                       RFLOAT phaseShift,
                       RFLOAT conT,
                       RFLOAT k1,
                       RFLOAT k2,
                       int pf,
                       int idim,
                       int vdim,
                       int npxl,
                       int interp);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalM(int gpuIdx,
                  int datShift,
                  //int l,
                  //RFLOAT* dvpA,
                  //RFLOAT* baseL,
                  ManagedCalPoint* mcp,
                  Complex* devdatP,
                  RFLOAT* devctfP,
                  RFLOAT* devsigP,
                  RFLOAT* wC,
                  RFLOAT* wR,
                  RFLOAT* wT,
                  RFLOAT* wD,
                  double oldC,
                  int npxl);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalHostF(int gpuIdx,
                      RFLOAT** wC,
                      RFLOAT** wR,
                      RFLOAT** wT,
                      RFLOAT** wD,
                      double** oldR,
                      double** oldT,
                      double** oldD,
                      double** trans,
                      double** rot,
                      double** dpara,
                      int cSearch);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalFin(int gpuIdx,
                    Complex** devdatP,
                    RFLOAT** devctfP,
                    RFLOAT** devdefO,
                    RFLOAT** devfreQ,
                    RFLOAT** devsigP,
                    int cSearch);

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectFreeIdx(int gpuIdx,
                   int** deviCol,
                   int** deviRow);

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
void allocDevicePoint(int gpuIdx,
                      Complex** dev_C,
                      RFLOAT** dev_W,
                      RFLOAT** dev_T,
                      RFLOAT** dev_tab,
                      RFLOAT** devDiff,
                      RFLOAT** devMax,
                      int** devCount,
                      void** stream,
                      int streamNum,
                      int tabSize,
                      int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void hostDeviceInit(int gpuIdx,
                    Complex* C3D,
                    RFLOAT* W3D,
                    RFLOAT* T3D,
                    RFLOAT* tab,
                    RFLOAT* dev_W,
                    RFLOAT* dev_T,
                    RFLOAT* dev_tab,
                    void** stream,
                    int streamNum,
                    int tabSize,
                    int r,
                    int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateC(int gpuIdx,
                Complex *C3D,
                Complex *dev_C,
                RFLOAT *dev_T,
                RFLOAT *dev_W,
                void** stream,
                int streamNum,
                const int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void ConvoluteC(int gpuIdx,
                RFLOAT *C3D,
                RFLOAT* dev_C,
                RFLOAT* dev_tab,
                void** stream,
                RFLOAT begin,
                RFLOAT end,
                RFLOAT step,
                int tabsize, 
                const RFLOAT nf,
                int streamNum,
                const int padSize,
                const int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void UpdateWC(int gpuIdx,
              Complex *C3D,
              Complex *dev_C,
              RFLOAT *diff,
              RFLOAT *cmax,
              RFLOAT *dev_W,
              RFLOAT *devDiff,
              RFLOAT *devMax,
              int *devCount,
              int *counter,
              void** stream,
              RFLOAT &diffC, 
              int streamNum, 
              const int r,
              const int dim);

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void freeDevHostPoint(int gpuIdx,
                      Complex** dev_C,
                      RFLOAT** dev_W,
                      RFLOAT** dev_T,
                      RFLOAT** dev_tab,
                      RFLOAT** devDiff,
                      RFLOAT** devMax,
                      int** devCount,
                      void** stream,
                      Complex* C3D,
                      RFLOAT* volumeW,
                      RFLOAT* volumeT,
                      int streamNum,
                      int dim);

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
void CalculateFW(int gpuIdx,
                 Complex *padDst,
                 Complex *F3D,
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
 * @brief ReMask.
 *
 * @param
 * @param
 */
void reMask(vector<Complex*>& imgData,
            RFLOAT maskRadius,
            RFLOAT pixelSize,
            RFLOAT ew,
            int idim,
            int imgNum);

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
