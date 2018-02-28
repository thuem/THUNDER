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
__constant__ extern double dev_ws_data[][DEV_CONST_MAT_SIZE];
//__constant__ extern double devRot[][DEV_CONST_MAT_SIZE * 4];

///////////////////////////////////////////////////////////////
//                     KERNEL ROUTINES
//

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getTextureC(double iCol,
                               double iRow,
                               double iSlc,
                               const int dim,
                               cudaTextureObject_t texObject);

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getTextureC2D(double iCol,
                                 double iRow,
                                 const int dim,
                                 cudaTextureObject_t texObject);

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getByInterp2D(double iCol,
                                 double iRow,
                                 const int interp,
                                 const int dim,
                                 cudaTextureObject_t texObject);

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getByInterpolationFTC(double iCol,
                                         double iRow,
                                         double iSlc,
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
                                    double* dev_def,
                                    double* dev_k1,
                                    double* dev_k2,
                                    int* deviCol,
                                    int* deviRow,
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
__global__ void kernel_getRotMat(double* devRotm,
                                 double* devnR,
                                 int nR);

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
__global__ void kernel_logDataVS(Complex* devdatP,
                                 Complex* priRotP,
                                 Complex* devtraP,
                                 double* devctfP,
                                 double* devsigP,
                                 double* devDvp,
                                 int nT,
                                 int rbatch,
                                 int npxl);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_UpdateW3D(double* devDvp,
                                 double* devbaseL,
                                 double* devwC,
                                 double* devwR,
                                 double* devwT,
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
__global__ void kernel_UpdateW2D(double* devDvp,
                                 double* devbaseL,
                                 double* devwC,
                                 double* devwR,
                                 double* devwT,
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
 * @brief ...
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
                                 double* dev_offS,
                                 double* dev_tran,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int npxl,
                                 int mReco,
                                 int idim);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateCTF(double* devctfP,
                                    CTFAttr* dev_ctfas,
                                    double* dev_ramD,
                                    int* deviCol,
                                    int* deviRow,
                                    double pixel,
                                    int insertIdx,
                                    int npxl,
                                    int mReco);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InsertT(double* devDataT,
                               double* devctfP,
                               double* devsigRcpP,
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
                               double* devctfP,
                               double* devsigRcpP,
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
__global__ void kernel_InsertT(double* devDataT,
                               double* devctfP,
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
                               double* devctfP,
                               double* dev_mat,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx);

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeT(double *devDataT,
                                  const int dimSize,
                                  const int num,
                                  const double sf);

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
__global__ void kernel_SymmetrizeT(double *devDataT,
                                   double *devSymmat, 
                                   const int numSymMat,
                                   const double r, 
                                   const int interp,
                                   const int num,
                                   const int dim,
                                   const int dimSize,
                                   cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ShellAverage(double *devAvg, 
                                    int *devCount, 
                                    double *devDataT,
                                    int dim, 
                                    int r,
                                    int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateAvg(double *devAvg2D,
                                    int *devCount2D,
                                    double *devAvg,
                                    int *devCount,
                                    int dim,
                                    int r);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateFSC(double *devDataT,
                                    double *devFSC,
                                    double *devAvg,
                                    int fscMatsize,
                                    bool joinHalf, 
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
__global__ void kernel_WienerConst(double *devDataT,
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
__global__ void kernel_CalculateW(double *devDataW,  
                                  double *devDataT,  
                                  const int length,
                                  const int num,
                                  const int dim,
                                  const int r);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InitialW(double *devDataW,  
                                int initWR, 
                                int dim,
                                int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_DeterminingC(Complex *devDataC,
                                    double *devDataT, 
                                    double *devDataW,
                                    const int length);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_convoluteC(double *devDoubleC,
                                  TabFunction tabfunc,
                                  double nf,
                                  int padSize,
                                  int dim,
                                  int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_RecalculateW(double *devDataW,
                                    Complex *devDataC,  
                                    int initWR, 
                                    int dim,
                                    int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCAVG(double *diff,
                                 double *counter,
                                 Complex *devDataT,  
                                 int r, 
                                 int dim,
                                 int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCMAX(double *devMax,
                                 Complex *devDataC,  
                                 int r, 
                                 int dim,
                                 int dimSize);
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
	                              const double sf);

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
                                   const double r, 
                                   const int interp,
                                   const int num,
                                   const int dim,
                                   const int dimSize,
                                   cudaTextureObject_t texObject);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_NormalizeFW(Complex *devDataF, 
	                               double *devDataW, 
	                               const int dimSize, 
	                               const int shiftF,
                                   const int shiftW);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_LowpassF(Complex *devDataF, 
	                            double thres,
                                double ew,
                                const int num,
	                            const int dim,
                                const int dimSize); 

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF(double *devDst, 
                                double *devMkb,
                                double nf,
                                const int dim, 
                                const int dimSize,
                                const int shift); 

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF(double *devDst, 
                                double *devTik,
                                const int dim, 
                                const int dimSize,
                                const int shift);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Background(double *devDst,
                                  double *devSumG,
                                  double *devSumWG,
                                  double r,
                                  double edgeWidth,
                                  const int dim,
                                  const int dimSize);

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateBg(double *devSumG,
                                   double *devSumWG,
                                   double *bg,
                                   int dim);
/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_SoftMaskD(double *devDst,
                                 double *bg,
                                 double r,
                                 double edgeWidth,
                                 const int dim,
                                 const int dimSize,
                                 const int shift);


///////////////////////////////////////////////////////////////

} // end namespace cuthunder

///////////////////////////////////////////////////////////////

#endif
