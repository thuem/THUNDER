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
                                int pf);
        
        /**
         * @brief Initialize insider variable.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void init(int tid, int batchSize);

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
        D_CALLABLE void expectPrectf(CTFAttr* dev_ctfa,
                                     RFLOAT* dev_def,
                                     RFLOAT* dev_k1,
                                     RFLOAT* dev_k2,
                                     int* deviCol,
                                     int* deviRow,
                                     int imgidx,
                                     int blockSize,
                                     int npxl);
        
        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void normalizeTF(Complex *devDataF,
                                    RFLOAT *devDataT, 
                                    const int dimSize, 
                                    const int num,
                                    const RFLOAT sf);

        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void normalizeT(RFLOAT *devDataT, 
                                   const int dimSize, 
                                   const int num,
                                   const RFLOAT sf);

        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE RFLOAT getTexture(RFLOAT iCol, 
                                     RFLOAT iRow, 
                                     RFLOAT iSlc,
                                     const int dim,
                                     cudaTextureObject_t texObject) const;

        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE RFLOAT getByInterpolationFT(RFLOAT iCol, 
                                               RFLOAT iRow, 
                                               RFLOAT iSlc, 
                                               const int interp,
                                               const int dim,
                                               cudaTextureObject_t texObject) const;
        
        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void symmetrizeT(RFLOAT *devDataT,
                                    double *matBase, 
                                    int r, 
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
        D_CALLABLE void shellAverage(RFLOAT *devAvg2D, 
                                     int *devCount2D, 
                                     RFLOAT *devDataT,
                                     RFLOAT *sumAvg,
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
        D_CALLABLE void calculateAvg(RFLOAT *devAvg2D, 
                                     int *devCount2D,
                                     RFLOAT *devAvg,
                                     int *devCount,
                                     int dim, 
                                     int r);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void calculateFSC(RFLOAT *devFSC,
                                     RFLOAT *devAvg, 
                                     RFLOAT *devDataT,
                                     int num,
                                     int dim,
                                     int dimSize,
                                     int fscMatsize,
                                     int wiener,
                                     int r,
                                     int pf,
                                     bool joinHalf);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void wienerConst(RFLOAT *devDataT,
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
        D_CALLABLE void calculateW(RFLOAT *devDataW, 
                                   RFLOAT *devDataT,
                                   const int length,
                                   const int num, 
                                   const int dim, 
                                   const int r);
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void initialW(RFLOAT *devDataW,
                                 int initWR,
                                 int dim,
                                 int dimSize);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void determiningC(Complex *devDataC,
                                     RFLOAT *devDataT,
                                     RFLOAT *devDataW,
                                     int length);
        
        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void convoluteC(RFLOAT *devDoubleC,
                                   TabFunction &tabfunc,
                                   RFLOAT nf,
                                   int padSize,
                                   int dim,
                                   int dimSize);
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void recalculateW(Complex *devDataC,
                                     RFLOAT *devDataW,
                                     int initWR,
                                     int dim,
                                     int dimSize);
        
        /**
         * @brief ...
         * @param ...
         * @param ...
         */
        D_CALLABLE void checkCAVG(RFLOAT *sumDiff,
                                  int *sumCount,
                                  RFLOAT *diff,
                                  int *counter,
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
        D_CALLABLE void checkCMAX(RFLOAT *singleMax,
                                  RFLOAT *devMax,
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
                                   const RFLOAT sf);
        
        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE Complex getTextureC(RFLOAT iCol, 
                                       RFLOAT iRow, 
                                       RFLOAT iSlc,
                                       const int dim,
                                       cudaTextureObject_t texObject) const;

        /**
         * @brief ...
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE Complex getByInterpolationFTC(RFLOAT iCol, 
                                                 RFLOAT iRow, 
                                                 RFLOAT iSlc, 
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
                                    int r, 
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
        D_CALLABLE void normalizeFW(Complex *devDst,
                                    Complex *devDataF, 
                                    RFLOAT *devDataW, 
                                    const int dimSize, 
                                    const int num,
                                    const int r,
                                    const int pdim,
                                    const int fdim);
        
        /**
         * @brief For LowpassFilter F.
         *
         * @param ...
         * @param ...
         */
        D_CALLABLE void lowpassF(Complex *devDataF, 
                                 const RFLOAT thres, 
                                 const RFLOAT ew,
                                 const int num,
                                 const int dim,
                                 const int dimSize);
        
        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void correctF(RFLOAT *devDst,
                                 RFLOAT *devMkb,
                                 RFLOAT nf,
                                 int dim,
                                 int dimSize,
                                 int shift); 

        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void correctF(RFLOAT *devDst,
                                 RFLOAT *devTik,
                                 int dim,
                                 int dimSize,
                                 int shift);

        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void background(RFLOAT *devDst,
                                   RFLOAT *sumWeight,
                                   RFLOAT *sumS,
                                   RFLOAT *sumGlobal,
                                   RFLOAT *sumweightG,
                                   RFLOAT r,
                                   RFLOAT ew,
                                   int dim,
                                   int dimSize,
                                   int indexDiff,
                                   int blockId);

        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void calculateBg(RFLOAT *devSumG, 
                                    RFLOAT *devSumWG,
                                    RFLOAT *bg,
                                    int dim);
        
        /**
         * @brief 
         * @param ...
         * @param ...
         */
        D_CALLABLE void softMaskD(RFLOAT *devDst,
                                  RFLOAT *bg,
                                  RFLOAT r,
                                  RFLOAT ew,
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
