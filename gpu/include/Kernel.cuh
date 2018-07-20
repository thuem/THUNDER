/**************************************************************
 * FileName: Kernel.cuh
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cstdlib>
#include <iostream>

#include "Config.h"
#include "Precision.h"

#include "Mat33.cuh"
#include "Vec3.cuh"
#include "Complex.cuh"
#include "Image.cuh"
#include "Volume.cuh"
#include "Weilume.cuh"
#include "TabFunction.cuh"
#include "Constructor.cuh"

namespace cuthunder {

///////////////////////////////////////////////////////////////
//                          MACROS
//

#define BUFF_SIZE                   256
#define HOST_PGLK_MEM_SIZE    BUFF_SIZE
#define DEV_CONST_MAT_SIZE    BUFF_SIZE


///////////////////////////////////////////////////////////////
//                  GLOBAL CONSTANT VARIABLES
//

//__constant__ extern double dev_mat_data[][DEV_CONST_MAT_SIZE * 9];
__constant__ extern RFLOAT dev_ws_data[][DEV_CONST_MAT_SIZE];
//__constant__ extern double devRot[][DEV_CONST_MAT_SIZE * 4];

///////////////////////////////////////////////////////////////
//                     KERNEL ROUTINES
//

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getTextureC(RFLOAT iCol,
                               RFLOAT iRow,
                               RFLOAT iSlc,
                               const int dim,
                               cudaTextureObject_t texObject);

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getTextureC2D(RFLOAT iCol,
                                 RFLOAT iRow,
                                 const int dim,
                                 cudaTextureObject_t texObject);

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getByInterp2D(RFLOAT iCol,
                                 RFLOAT iRow,
                                 const int interp,
                                 const int dim,
                                 cudaTextureObject_t texObject);

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getByInterpolationFTC(RFLOAT iCol,
                                         RFLOAT iRow,
                                         RFLOAT iSlc,
                                         const int interp,
                                         const int dim,
                                         cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ExpectPrectf(CTFAttr* dev_ctfa,
                                    RFLOAT* dev_def,
                                    RFLOAT* dev_k1,
                                    RFLOAT* dev_k2,
                                    int* deviCol,
                                    int* deviRow,
                                    int npxl);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_TranslateL(Complex* devtraP,
                                  double* devnT,
                                  int* deviCol,
                                  int* deviRow,
                                  int idim,
                                  int npxl);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(Complex* devtraP,
                                 double* dev_trans,
                                 int* deviCol,
                                 int* deviRow,
                                 int idim,
                                 int npxl);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalCTFL(RFLOAT* devctfP,
                               RFLOAT* devdefO,
                               RFLOAT* devfreQ,
                               double* devdP,
                               RFLOAT phaseShift,
                               RFLOAT conT,
                               RFLOAT k1,
                               RFLOAT k2,
                               int npxl);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRotMatL(double* devRotm,
                                  double* devnR,
                                  int nR);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRotMat(double* devRotm,
                                 double* devnR,
                                 int nR);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Project3DL(Complex* priRotP,
                                  double* devRotm,
                                  int* deviCol,
                                  int* deviRow,
                                  int pf,
                                  int vdim,
                                  int npxl,
                                  int interp,
                                  cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Project3D(Complex* priRotP,
                                 double* devRotm,
                                 int* deviCol,
                                 int* deviRow,
                                 int pf,
                                 int vdim,
                                 int npxl,
                                 int interp,
                                 cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Project3D(Complex* priRotP,
                                 double* devRotm,
                                 int* deviCol,
                                 int* deviRow,
                                 int shift,
                                 int pf,
                                 int vdim,
                                 int npxl,
                                 int interp,
                                 cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Project2DL(Complex* priRotP,
                                  double* devnR,
                                  int* deviCol,
                                  int* deviRow,
                                  int pf,
                                  int vdim,
                                  int npxl,
                                  int interp,
                                  cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Project2D(Complex* priRotP,
                                 double* devnR,
                                 int* deviCol,
                                 int* deviRow,
                                 int shift,
                                 int pf,
                                 int vdim,
                                 int npxl,
                                 int interp,
                                 cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_logDataVSL(Complex* priRotP,
                                  Complex* devtraP,
                                  Complex* devdatP,
                                  RFLOAT* devctfP,
                                  RFLOAT* devsigP,
                                  RFLOAT* devDvp,
                                  //RFLOAT* devre,
                                  int nT,
                                  int npxl);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_logDataVSLC(Complex* priRotP,
                                   Complex* devtraP,
                                   Complex* devdatP,
                                   RFLOAT* devctfP,
                                   RFLOAT* devsigP,
                                   RFLOAT* devDvp,
                                   int nT,
                                   int nD,
                                   int npxl);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_logDataVS(Complex* devdatP,
                                 Complex* priRotP,
                                 Complex* devtraP,
                                 RFLOAT* devctfP,
                                 RFLOAT* devsigP,
                                 RFLOAT* devDvp,
                                 int r,
                                 int nR,
                                 int nT,
                                 int rbatch,
                                 int npxl);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getMaxBaseL(RFLOAT* devBaseL,
                                   RFLOAT* devDvp,
                                   int angleNum);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getMaxBase(RFLOAT* devbaseL,
                                  RFLOAT* devDvp,
                                  int angleNum);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_setBaseLine(RFLOAT* devcomP,
                                   RFLOAT* devbaseL,
                                   RFLOAT* devwC,
                                   RFLOAT* devwR,
                                   RFLOAT* devwT,
                                   int nK,
                                   int nR,
                                   int nT);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateW(RFLOAT* devDvp,
                               RFLOAT* devbaseL,
                               RFLOAT* devwC,
                               RFLOAT* devwR,
                               RFLOAT* devwT,
                               double* devpR,
                               double* devpT,
                               int kIdx,
                               int nK,
                               int nR,
                               int rSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateWL(RFLOAT* devDvp,
                                RFLOAT* devBaseL,
                                RFLOAT* devwC,
                                RFLOAT* devwR,
                                RFLOAT* devwT,
                                RFLOAT* devwD,
                                double* devR,
                                double* devT,
                                double* devD,
                                double oldC,
                                int nT,
                                int l);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateWL(RFLOAT* devDvp,
                                RFLOAT* devBaseL,
                                RFLOAT* devwC,
                                RFLOAT* devwR,
                                RFLOAT* devwT,
                                RFLOAT* devwD,
                                double* devR,
                                double* devT,
                                double* devD,
                                double oldC,
                                int nT);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateWLC(RFLOAT* devDvp,
                                 RFLOAT* devBaseL,
                                 RFLOAT* devwC,
                                 RFLOAT* devwR,
                                 RFLOAT* devtT,
                                 RFLOAT* devtD,
                                 double* devR,
                                 double* devT,
                                 double* devD,
                                 double oldC,
                                 int nT,
                                 int nD);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ReduceW(RFLOAT* devw,
                               RFLOAT* devt);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateW3D(RFLOAT* devDvp,
                                 RFLOAT* devbaseL,
                                 RFLOAT* devwC,
                                 RFLOAT* devwR,
                                 RFLOAT* devwT,
                                 int rIdx,
                                 int nK,
                                 int nR,
                                 int nT,
                                 int rSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateW2D(RFLOAT* devDvp,
                                 RFLOAT* devbaseL,
                                 RFLOAT* devwC,
                                 RFLOAT* devwR,
                                 RFLOAT* devwT,
                                 int kIdx,
                                 int rIdx,
                                 int nK,
                                 int nR,
                                 int nT,
                                 int rSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRandomCTD(double* dev_nt,
                                    double* dev_tran,
                                    double* dev_nd,
                                    double* dev_ramD,
                                    double* dev_nr,
                                    double* dev_ramR,
                                    unsigned int out,
                                    int rSize,
                                    int tSize,
                                    int dSize
                                    );

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRandomCTD(double* dev_nt,
                                    double* dev_tran,
                                    double* dev_nr,
                                    double* dev_ramR,
                                    unsigned int out,
                                    int rSize,
                                    int tSize
                                    );

/**
 * @brief A Kernel for Calcuating Rotation Matrices From Quaternions
 * 
 * each block should be an image
 * each thread of a block should be an insertation of an image
 *
 * @param dev_mat rotation matrices
 * @param dev_ramR quaternions
 */
__global__ void kernel_getRandomR(double* dev_mat,
                                  double* dev_ramR,
                                  int* dev_nc);

/**
 * @brief 
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRandomR(double* dev_mat,
                                  double* dev_ramR);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(Complex* devdatP,
                                 Complex* devtranP,
                                 double* dev_tran,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(Complex* devdatP,
                                 Complex* devtranP,
                                 double* dev_offS,
                                 double* dev_tran,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(Complex* devdatP,
                                 Complex* devtranP,
                                 double* dev_tran,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Translate(Complex* devdatP,
                                 Complex* devtranP,
                                 double* dev_offS,
                                 double* dev_tran,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateCTF(RFLOAT* devctfP,
                                    CTFAttr* dev_ctfas,
                                    double* dev_ramD,
                                    int* dev_nc,
                                    int* deviCol,
                                    int* deviRow,
                                    RFLOAT pixel,
                                    int insertIdx,
                                    int opf,
                                    int npxl,
                                    int mReco);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateCTF(RFLOAT* devctfP,
                                    CTFAttr* dev_ctfas,
                                    double* dev_ramD,
                                    int* deviCol,
                                    int* deviRow,
                                    RFLOAT pixel,
                                    int insertIdx,
                                    int opf,
                                    int npxl,
                                    int mReco);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT2D(RFLOAT* devDataT,
                                 RFLOAT* devctfP,
                                 RFLOAT* devsigRcpP,
                                 double* dev_nr,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int npxl,
                                 int mReco,
                                 int vdim,
                                 int vdimSize,
                                 int smidx);

__global__ void kernel_InsertF2D(Complex* devDataF,
                                 Complex* devtranP,
                                 RFLOAT* devctfP,
                                 RFLOAT* devsigRcpP,
                                 double* dev_nr,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int npxl,
                                 int mReco,
                                 int vdim,
                                 int vdimSize,
                                 int smidx);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT2D(RFLOAT* devDataT,
                                 RFLOAT* devctfP,
                                 double* dev_nr,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int npxl,
                                 int mReco,
                                 int vdim,
                                 int vdimSize,
                                 int smidx);

__global__ void kernel_InsertF2D(Complex* devDataF,
                                 Complex* devtranP,
                                 RFLOAT* devctfP,
                                 double* dev_nr,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int npxl,
                                 int mReco,
                                 int vdim,
                                 int vdimSize,
                                 int smidx);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO2D(double* devO,
                                 int* devC,
                                 double* dev_nr,
                                 double* dev_nt,
                                 double* dev_offs,
                                 int* dev_nc,
                                 int insertIdx,
                                 int mReco);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO2D(double* devO,
                                 int* devC,
                                 double* dev_nr,
                                 double* dev_nt,
                                 int* dev_nc,
                                 int insertIdx,
                                 int mReco);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(RFLOAT* devDataT,
                               RFLOAT* devctfP,
                               RFLOAT* devsigRcpP,
                               double* dev_mat,
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);

__global__ void kernel_InsertF(Complex* devDataF,
                               Complex* devtranP,
                               RFLOAT* devctfP,
                               RFLOAT* devsigRcpP,
                               double* dev_mat,
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);
/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(RFLOAT* devDataT,
                               RFLOAT* devctfP,
                               double* dev_mat,
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);

__global__ void kernel_InsertF(Complex* devDataF,
                               Complex* devtranP,
                               RFLOAT* devctfP,
                               double* dev_mat,
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO3D(double* devO,
                                 int* devC,
                                 double* dev_mat,
                                 double* dev_nt,
                                 double* dev_offs,
                                 int* dev_nc,
                                 int insertIdx,
                                 int mReco,
                                 int batchSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO3D(double* devO,
                                 int* devC,
                                 double* dev_mat,
                                 double* dev_nt,
                                 int* dev_nc,
                                 int insertIdx,
                                 int mReco,
                                 int batchSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(RFLOAT* devDataT,
                               RFLOAT* devctfP,
                               RFLOAT* devsigRcpP,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);

__global__ void kernel_InsertF(Complex* devDataF,
                               Complex* devtranP,
                               RFLOAT* devctfP,
                               RFLOAT* devsigRcpP,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);
/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(RFLOAT* devDataT,
                               RFLOAT* devctfP,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);

__global__ void kernel_InsertF(Complex* devDataF,
                               Complex* devtranP,
                               RFLOAT* devctfP,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO3D(double* devO,
                                 int* devC,
                                 double* dev_mat,
                                 double* dev_nt,
                                 double* dev_offs,
                                 int insertIdx,
                                 int mReco,
                                 int batchSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertO3D(double* devO,
                                 int* devC,
                                 double* dev_mat,
                                 double* dev_nt,
                                 int insertIdx,
                                 int mReco,
                                 int batchSize);

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_SetSF2D(RFLOAT *devDataT, 
                               RFLOAT *sf,
                               int dimSize);

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeTF2D(Complex *devDataF,
                                     RFLOAT *devDataT, 
                                     RFLOAT *sf,
                                     int kIdx);

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeTF(Complex *devDataF,
                                   RFLOAT *devDataT,
                                   const size_t dimSize,
                                   const size_t num,
                                   const RFLOAT sf);

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeT(RFLOAT *devDataT,
                                  const int dimSize,
                                  const int num,
                                  const RFLOAT sf);

/**
 * @brief Symmetrize T3D
 *
 * @param devDataT : the pointer of T3D
 * @param devSym : the pointer of Volume
 * @param devSymmat : the Symmetry Matrix
 * @param numSymMat : the size of the Symmetry Matrix
 * @param r : the range of T3D elements need to be symmetrized
 * @param interp : the way of interpolating
 * @param dim : the length of one side of T3D
 */
__global__ void kernel_SymmetrizeT(RFLOAT *devDataT,
                                   double *devSymmat, 
                                   const int numSymMat,
                                   const int r, 
                                   const int interp,
                                   const size_t num,
                                   const int dim,
                                   const size_t dimSize,
                                   cudaTextureObject_t texObject);

/**
 * @brief Normalize F: F = F * sf
 *
 * @param devDataF : the pointer of F3D
 * @param length : F3D's size
 * @param sf : the coefficient to Normalize F
 **/
__global__ void kernel_NormalizeF(Complex *devDataF, 
	                              const int dimSize, 
                                  const int num, 
	                              const RFLOAT sf);

/**
 * @brief Symmetrize F3D
 *
 * @param devDataF : the pointer of F3D
 * @param devSym : the pointer of Volume
 * @param devSymmat : the Symmetry Matrix
 * @param numSymMat : the size of the Symmetry Matrix
 * @param r : the range of F3D elements need to be symmetrized
 * @param interp : the way of interpolating
 * @param dim : the length of one side of F3D
 **/
__global__ void kernel_SymmetrizeF(Complex *devDataF,
                                   double *devSymmat, 
                                   const int numSymMat,
                                   const int r, 
                                   const int interp,
                                   const size_t num,
                                   const int dim,
                                   const size_t dimSize,
                                   cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ShellAverage2D(RFLOAT *devAvg2D, 
                                      int *devCount2D, 
                                      RFLOAT *devDataT,
                                      int dim, 
                                      int r);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ShellAverage(RFLOAT *devAvg2D, 
                                    int *devCount2D, 
                                    RFLOAT *devDataT,
                                    int dim, 
                                    int r,
                                    size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateAvg(RFLOAT *devAvg2D,
                                    int *devCount2D,
                                    RFLOAT *devAvg,
                                    int *devCount,
                                    int dim,
                                    int r);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateFSC2D(RFLOAT *devDataT,
                                      RFLOAT *devFSC,
                                      RFLOAT *devAvg,
                                      bool joinHalf, 
                                      int fscMatsize,
                                      int wiener, 
                                      int dim,
                                      int pf,
                                      int r);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateFSC(RFLOAT *devDataT,
                                    RFLOAT *devFSC,
                                    RFLOAT *devAvg,
                                    int fscMatsize,
                                    bool joinHalf, 
                                    int wiener, 
                                    int r, 
                                    int pf, 
                                    size_t num,
                                    int dim,
                                    size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_WienerConst(RFLOAT *devDataT,
                                   int wiener,
                                   int r,
                                   int num, 
                                   int dim,
                                   int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateW2D(RFLOAT *devDataW,  
                                    RFLOAT *devDataT,  
                                    const int dim,
                                    const int r);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateW(RFLOAT *devDataW,  
                                  RFLOAT *devDataT,  
                                  const size_t length,
                                  const size_t num,
                                  const int dim,
                                  const int r);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InitialW2D(RFLOAT *devDataW,  
                                  int initWR, 
                                  int dim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InitialW(RFLOAT *devDataW,  
                                int initWR,
                                int dim,
                                size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InitialW(RFLOAT *devDataW,  
                                int initWR,
                                int shift, 
                                int dim,
                                size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_DeterminingC(Complex *devDataC,
                                    RFLOAT *devDataT, 
                                    RFLOAT *devDataW,
                                    const size_t length);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_convoluteC2D(RFLOAT *devDoubleC,
                                    TabFunction tabfunc,
                                    RFLOAT nf,
                                    int padSize,
                                    int dim,
                                    int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ConvoluteC(RFLOAT *devDataC,
                                  TabFunction tabfunc,
                                  RFLOAT nf,
                                  int dim,
                                  size_t shift,
                                  int padSize,
                                  size_t batch);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_convoluteC(RFLOAT *devDoubleC,
                                  TabFunction tabfunc,
                                  RFLOAT nf,
                                  int padSize,
                                  int dim,
                                  size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_RecalculateW2D(RFLOAT *devDataW,
                                      Complex *devDataC,  
                                      int initWR, 
                                      int dim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_RecalculateW(Complex *devDataC,
                                    RFLOAT *devDataW,  
                                    int initWR, 
                                    size_t shift, 
                                    int dim,
                                    size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_RecalculateW(RFLOAT *devDataW,
                                    Complex *devDataC,  
                                    int initWR, 
                                    int dim,
                                    size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCAVG2D(RFLOAT *diff,
                                   int *counter,
                                   Complex *devDataC,  
                                   int r, 
                                   int dim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCAVG(RFLOAT *diff,
                                 int *counter,
                                 Complex *devDataT,  
                                 int r, 
                                 int dim,
                                 size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCMAX2D(RFLOAT *devMax,
                                   Complex *devDataC,  
                                   int r, 
                                   int dim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCMAX(RFLOAT *devMax,
                                 Complex *devDataC,  
                                 int r, 
                                 int dim,
                                 size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_NormalizeFW2D(Complex *devDst,
                                     Complex *devDataF, 
                                     RFLOAT *devDataW,
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
__global__ void kernel_NormalizeP2D(RFLOAT *devDstR, 
                                    int dimSize);

/**
 * @brief 
 *
 * @param 
 * @param 
 * @param 
 */
__global__ void kernel_NormalizeP(RFLOAT *devDstR, 
                                  size_t length,
                                  size_t shift,
                                  size_t dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_NormalizeFW(Complex *devDst,
                                   Complex *devDataF, 
	                               RFLOAT *devDataW, 
	                               const size_t dimSize, 
	                               const size_t shift,
                                   const int r,
                                   const int pdim,
                                   const int fdim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_LowpassF(Complex *devDataF, 
	                            RFLOAT thres,
                                RFLOAT ew,
                                const int num,
	                            const int dim,
                                const int dimSize); 

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF2D(RFLOAT *devDstI, 
                                  RFLOAT *devMkb,
                                  RFLOAT nf,
                                  const int dim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF(RFLOAT *devDst, 
                                RFLOAT *devMkb,
                                RFLOAT nf,
                                const int dim, 
                                const size_t dimSize,
                                const size_t shift); 

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF2D(RFLOAT *devDstI, 
                                  RFLOAT *devTik,
                                  const int dim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF(RFLOAT *devDst, 
                                RFLOAT *devTik,
                                const int dim, 
                                const size_t dimSize,
                                const size_t shift);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_TranslateI2D(Complex* devSrc, 
                                    RFLOAT ox,
                                    RFLOAT oy,
                                    int r,
                                    int shift,
                                    int dim);
 
/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_TranslateI(Complex* devRef, 
                                  RFLOAT ox,
                                  RFLOAT oy,
                                  RFLOAT oz,
                                  int r,
                                  size_t shift,
                                  int dim,
                                  size_t batch);
 
/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_SoftMask(RFLOAT *devMask,
                                RFLOAT r,
                                RFLOAT ew,
                                int dim,
                                size_t imgSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_MulMask(RFLOAT *dev_image,
                               RFLOAT *devMask,
                               int imgIdx,
                               int dim,
                               size_t imgSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Background(RFLOAT *devDst,
                                  RFLOAT *devSumG,
                                  RFLOAT *devSumWG,
                                  RFLOAT r,
                                  RFLOAT edgeWidth,
                                  const int dim,
                                  const int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateBg(RFLOAT *devSumG,
                                   RFLOAT *devSumWG,
                                   RFLOAT *bg,
                                   int dim);
/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_SoftMaskD(RFLOAT *devDst,
                                 RFLOAT *bg,
                                 RFLOAT r,
                                 RFLOAT edgeWidth,
                                 const int dim,
                                 const int dimSize,
                                 const int shift);


///////////////////////////////////////////////////////////////

} // end namespace cuthunder

///////////////////////////////////////////////////////////////

#endif
