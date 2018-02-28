/**************************************************************
 * FileName: Constructor.cuh
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#ifndef CONSTRUCTOR_CUH
#define CONSTRUCTOR_CUH

#include "Device.cuh"
#include "Volume.cuh"
#include "Weilume.cuh"
#include "Image.cuh"
#include "Mat33.cuh"
#include "Vec2.cuh"
#include "Vec3.cuh"
#include "Vec4.cuh"
#include "TabFunction.cuh"
#include "curand_kernel.h"

namespace cuthunder {

#define NORM2(a, b) norm3d((double)a, (double)b, 0.0)

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef PI_2
#define PI_2 6.28318530717959
#endif

#ifndef divPI2
#define divPI2 1.57079632679489661923132169164
#endif

struct insertParam
{
    bool cSearch;
    double rU;
    double rL;
    double w1;
    double w2;
    double pixelSize;
    int nC;
    int nR;
    int nT;
    int nD;
    int mReco;
    int pf;
};

class Constructor
{
    public:

        struct BiTuple { int x; int y; };

        enum TaskKind { FULL, PART, HALF };
        
        HD_CALLABLE Constructor() {}

        HD_CALLABLE ~Constructor() {}

        /**
         * @brief Constructor for insertF task.
         *
         * @param ...
         * @param ...
         */
        HD_CALLABLE Constructor(Volume vol,
                                ImageStream imgss,
                                TabFunction tabfunc,
                                double dimsize,
                                double weight,
                                double alpha,
                                double pf);
        
        /**
         * @brief Initialize insider variable.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void init(int tid, int batchSize);

        /**
         * @brief Distribute task along one dimensional side.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void assignTask1D(int dim);

        /**
         * @brief Distribute task along two dimensional sides.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void assignTask2D(int nCol, int nRow, TaskKind type);

        /**
         * @brief Distribute task along three dimensional sides.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void assignTask3D(int dim, TaskKind type);
        D_CALLABLE void assignTask3DX(int dim, TaskKind type);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void addSurfaceD(double value,
                                    int iCol,
                                    int iRow,
                                    int iSlc,
                                    const int dim,
                                    cudaSurfaceObject_t surfObject) const;
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void addSurfaceC(Complex value,
                                    int iCol,
                                    int iRow,
                                    int iSlc,
                                    const int dim,
                                    cudaSurfaceObject_t surfObject) const;
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void addFTD(double* devDataT,
                               double value,
                               double iCol,
                               double iRow,
                               double iSlc,
                               const int dim) const;
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void addFTC(Complex* devDataF,
                               Complex& value,
                               double iCol,
                               double iRow,
                               double iSlc,
                               const int dim) const;
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void addFTCD(Complex* devDataF,
                                double* devDataT,
                                Complex& cvalue,
                                double dvalue,
                                double iCol,
                                double iRow,
                                double iSlc,
                                const int dim) const;
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void expectPrectf(CTFAttr* dev_ctfa,
                                     double* dev_def,
                                     double* dev_k1,
                                     double* dev_k2,
                                     int* deviCol,
                                     int* deviRow,
                                     int imgidx,
                                     int blockSize,
                                     int npxl);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void translate(Complex* devtraP,
                                  double* dev_trans,
                                  int* deviCol,
                                  int* deviRow,
                                  int idim,
                                  int npxl,
                                  int imgidx,
                                  int blockSize);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void getRotMat(double* devRotm,
                                  double* matS,
                                  double* devnR,
                                  int nR,
                                  int threadId);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void project(Complex* priRotP,
                                double* devRotm,
                                int* deviCol,
                                int* deviRow,
                                int shift,
                                int pf,
                                int vdim,
                                int npxl,
                                int interp,
                                int rIndex,
                                int blockSize,
                                cudaTextureObject_t texObject);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void logDataVS(Complex* devdatP,
                                  Complex* priRotP,
                                  Complex* devtraP,
                                  double* devctfP,
                                  double* devsigP,
                                  double* devDvp,
                                  double* result,
                                  int nT,
                                  int rbatch,
                                  int blockId,
                                  int blockSize,
                                  int npxl);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void updateW(double* devDvp,
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
         * @param ...
         * @param ...
         */
        D_CALLABLE void getRandomCTD(double* dev_nt,
                                     double* dev_tran,
                                     double* dev_nd,
                                     double* dev_ramD,
                                     double* dev_nr,
                                     double* dev_ramR,
                                     int* dev_ramC,
                                     unsigned long out,
                                     int nC,
                                     int nR,
                                     int nT,
                                     int nD,
                                     int blockId);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void getRandomCTD(double* dev_nt,
                                     double* dev_tran,
                                     double* dev_nr,
                                     double* dev_ramR,
                                     int* dev_ramC,
                                     unsigned long out,
                                     int nC,
                                     int nR,
                                     int nT,
                                     int blockId);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void getRandomR(double* dev_mat,
                                   double* matS,
                                   double* dev_ramR,
                                   int threadId);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void translate(Complex* dev_images,
                                  Complex* tranImgs,
                                  double* dev_offs,
                                  double* tran,
                                  int* deviCol,
                                  int* deviRow,
                                  int* deviPxl,
                                  int insertIdx,
                                  int npxl,
                                  int mReco,
                                  int idim,
                                  int imgidx,
                                  int blockSize,
                                  int imgSize);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void translate(Complex* dev_images,
                                  Complex* tranImgs,
                                  double* tran,
                                  int* deviCol,
                                  int* deviRow,
                                  int* deviPxl,
                                  int insertIdx,
                                  int npxl,
                                  int mReco,
                                  int idim,
                                  int imgidx,
                                  int blockSize,
                                  int imgSize);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void calculateCTF(Complex* dev_ctfs,
                                     CTFAttr* dev_ctfas,
                                     double* dev_ramD,
                                     int* deviCol,
                                     int* deviRow,
                                     int* deviPxl,
                                     double pixel,
                                     double w1,
                                     double w2,
                                     int insertIdx,
                                     int npxl,
                                     int mReco,
                                     int imgidx,
                                     int blockSize,
                                     int imgSize);
        
        /**
         * @brief Perform insertF task.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void insertT(double* devDataT,
                                Complex* dev_ctfs,
                                double* dev_ws,
                                double* dev_mat,
                                double* rotMat,
                                int* deviCol,
                                int* deviRow,
                                int* deviPxl,
                                int insertIdx,
                                int npxl,
                                int mReco,
                                int pf,
                                int vdim,
                                int imgidx,
                                int blockSize,
                                int imgSize);

        D_CALLABLE void insertF(Complex* devDataF,
                                double* devDataT,
                                Complex* tranImgs,
                                Complex* dev_ctfs,
                                double* dev_ws,
                                double* dev_mat,
                                double* rotMat,
                                int* dev_ramC,
                                int* deviCol,
                                int* deviRow,
                                int* deviPxl,
                                int insertIdx,
                                int npxl,
                                int mReco,
                                int pf,
                                int vdim,
                                int imgidx,
                                int blockSize,
                                int imgSize);

        D_CALLABLE void insertC();

        D_CALLABLE void symmetrizeF();

        D_CALLABLE void symmetrizeC();

        D_CALLABLE void normalizeW();

        D_CALLABLE void normalizeF();

        D_CALLABLE void checkC();

        D_CALLABLE void construct();

        /**
         * @brief For normalizeF,
         *            normalizeT,
         *            CheckCAVG,
         *            CheckCMAX.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void init(int tid);
        
        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void normalizeT(double *devDataT, 
                                   const int dimSize, 
                                   const int num,
                                   const double sf);

        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE double getTexture(double iCol, 
                                     double iRow, 
                                     double iSlc,
                                     const int dim,
                                     cudaTextureObject_t texObject) const;

        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE double getByInterpolationFT(double iCol, 
                                               double iRow, 
                                               double iSlc, 
                                               const int interp,
                                               const int dim,
                                               cudaTextureObject_t texObject) const;
        
        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void symmetrizeT(double *devDataT,
                                    double *matBase, 
                                    double r, 
                                    int numSymMat,
                                    int interp,
                                    int num,
                                    int dim,
                                    int dimSize,
                                    cudaTextureObject_t texObject);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void shellAverage(double *devAvg2D, 
                                     int *devCount2D, 
                                     double *devDataT,
                                     double *sumAvg,
                                     int *sumCount, 
                                     int r,
                                     int dim,
                                     int dimSize,
                                     int indexDiff,
                                     int blockId);

        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void calculateAvg(double *devAvg2D, 
                                     int *devCount2D,
                                     double *devAvg,
                                     int *devCount,
                                     int dim, 
                                     int r);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void calculateFSC(double *devFSC,
                                     double *devAvg, 
                                     double *devDataT,
                                     int num,
                                     int dim,
                                     int dimSize,
                                     int fscMatsize,
                                     int wiener,
                                     int r,
                                     bool joinHalf);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void wienerConst(double *devDataT,
                                    int wiener,
                                    int r,
                                    int num,
                                    int dim,
                                    int dimSize);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void calculateW(double *devDataW, 
                                   double *devDataT,
                                   const int length,
                                   const int num, 
                                   const int dim, 
                                   const int r);
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void initialW(double *devDataW,
                                 int initWR,
                                 int dim,
                                 int dimSize);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void determiningC(Complex *devDataC,
                                     double *devDataT,
                                     double *devDataW,
                                     int length);
        
        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void convoluteC(double *devDoubleC,
                                   TabFunction &tabfunc,
                                   double nf,
                                   int padSize,
                                   int dim,
                                   int dimSize);
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void recalculateW(Complex *devDataC,
                                     double *devDataW,
                                     int initWR,
                                     int dim,
                                     int dimSize);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void checkCAVG(double *sumDiff,
                                  double *sumCount,
                                  double *diff,
                                  double *counter,
                                  Complex *devDataC,
                                  int r,
                                  int dim,
                                  int dimSize,
                                  int indexDiff,
                                  int blockId);

        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void checkCMAX(double *singleMax,
                                  double *devMax,
                                  Complex *devDataC,
                                  int r,
                                  int dim,
                                  int dimSize,
                                  int indexDiff,
                                  int blockId);

        /**
         * @brief For normalize F (F * sf).
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void normalizeF(Complex *devDataF, 
                                   const int dimSize, 
                                   const int num,
                                   const double sf);
        
        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE Complex getTextureC(double iCol, 
                                       double iRow, 
                                       double iSlc,
                                       const int dim,
                                       cudaTextureObject_t texObject) const;

        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE Complex getByInterpolationFTC(double iCol, 
                                                 double iRow, 
                                                 double iSlc, 
                                                 const int interp,
                                                 const int dim,
                                                 cudaTextureObject_t texObject) const;
        
        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void symmetrizeF(Complex *devDataF,
                                    double *matBase, 
                                    double r, 
                                    int numSymMat,
                                    int interp,
                                    int num,
                                    int dim,
                                    int dimSize,
                                    cudaTextureObject_t texObject);
        
        /**
         * @brief For normalize FW.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void normalizeFW(Complex *F3D, 
                                    double *devDataW, 
                                    const int dimSize, 
                                    const int shiftF,
                                    const int shiftW);
        
        /**
         * @brief For LowpassFilter F.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void lowpassF(Complex *devDataF, 
                                 const double thres, 
                                 const double ew,
                                 const int num,
                                 const int dim,
                                 const int dimSize);
        
        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void correctF(double *devDst,
                                 double *devMkb,
                                 double nf,
                                 int dim,
                                 int dimSize,
                                 int shift); 

        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void correctF(double *devDst,
                                 double *devTik,
                                 int dim,
                                 int dimSize,
                                 int shift);

        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void background(double *devDst,
                                   double *sumWeight,
                                   double *sumS,
                                   double *sumGlobal,
                                   double *sumweightG,
                                   double r,
                                   double ew,
                                   int dim,
                                   int dimSize,
                                   int indexDiff,
                                   int blockId);

        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void calculateBg(double *devSumG, 
                                    double *devSumWG,
                                    double *bg,
                                    int dim);
        
        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void softMaskD(double *devDst,
                                  double *bg,
                                  double r,
                                  double ew,
                                  int dim,
                                  int dimSize,
                                  int shift);
        

        /* For debug and performance analysis */
        
        D_CALLABLE void check();

        D_CALLABLE void insertOne();

    private:

        /* Thread attribution */

        int _tid;

        int _dim;

        /* Calculation variables */

        Volume _F3D, _T3D;

        ImageStream _images;

        Mat33 _mat;

        TabFunction _MKBessel;

        Weilume _W3D, _C3D;

        int _maxRadius;

        double _w  = 1.0;
        double _a  = 1.9;
        int _pf = 2;
/*
        bool _cSearch;
        double _rU;
        double _rL;
        double _w1;
        double _w2;
        double _pixelSize;
        int _nC; 
        int _nR; 
        int _nT; 
        int _nD;
        int _mReco; 
*/

        /* Task assignment control variables */

        BiTuple _workerDim, _workerIdx;

        bool _isWorker = false;

        int _batchSize = 0;

        int _left   = 0;
        int _right  = 0;
        int _top    = 0;
        int _bottom = 0;
        int _front  = 0;
        int _back   = 0;
};

} // end namespace cuthunder

#endif
