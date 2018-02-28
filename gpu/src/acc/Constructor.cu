/**************************************************************
 * FileName: Constructor.cu
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#include "Constructor.cuh"
#include "Config.cuh"
namespace cuthunder {

//////////////////////////////////////////////////////////////////////////

static __inline__ D_CALLABLE double fetch_double(int hi, int lo){
    
    return __hiloint2double(hi, lo);
}

HD_CALLABLE Constructor::Constructor(Volume vol,
                                     ImageStream imgss,
                                     TabFunction tabfunc,
                                     double dimsize,
                                     double weight,
                                     double alpha,
                                     double pf)
    : _F3D(vol), _images(imgss), _MKBessel(tabfunc)
{
    _w  = weight;
    _a  = alpha;
    _pf = pf;

    _maxRadius = dimsize / pf - alpha;
}

D_CALLABLE void Constructor::init(int tid, int batchSize)
{
    _tid = tid;
    _batchSize = batchSize;
}

D_CALLABLE __forceinline__ int getIndexHalf(const int i,
                                            const int j,
                                            const int k,
                                            const int dim)
{
   return (k >= 0 ? k : k + dim) * (dim / 2 + 1) * dim
        + (j >= 0 ? j : j + dim) * (dim / 2 + 1)
        + i; 
}

//////////////////////////////////////////////////////////////////////////

/**
 * @brief Assign a (dim * dim) size layer of volume to a thread
 *
 * @param dim: volume dimension.
 */
D_CALLABLE void Constructor::assignTask1D(int dim)
{
    if (blockDim.x < dim / 2 + 1) {
        printf("*** Launched threads grid error in assignTask1D.\n");
        return ;
    }

    _tid = threadIdx.x;

    if (threadIdx.x < dim / 2 + 1) {
        _bottom = - dim / 2;
        _top = dim / 2;

        _left = - dim / 2;
        _right = dim / 2;

        _front = threadIdx.x;
        _back = _front + 1;

        _isWorker = true;
    }else {
        _isWorker = false;
    }
}

/**
 * @brief Assign one pixel of image to a thread.
 *
 * @param nCol: column of store dimension.
 * @param nRow: row of store dimension.
 * @param type: FULL means column range equals [-maxRadius, -maxRadius]
 *              HALF means column range equals [         0, -maxRadius]
 */
D_CALLABLE void Constructor::assignTask2D(int nCol, int nRow, TaskKind type)
{
    int nR_H = _maxRadius;
    int nR_V = nR_H;

    int L = 0, R = 0, T = 0, B = 0;

    if (nR_H >= nCol)
        nR_H = nCol - 1;

    if (type == TaskKind::FULL)
        L = -nR_H - 1;
    else if (type == TaskKind::HALF)
        L = 0;
    R = nR_H + 1;

    if (nR_V >= nRow / 2 - 1)
        nR_V = nRow / 2 - 2;

    T =  nR_V + 1;
    B = -nR_V - 1;

    _workerDim.x = R - L;
    _workerDim.y = T - B;

    if (_tid < _workerDim.x * _workerDim.y) {
        _workerIdx.y = _tid / _workerDim.x;
        _workerIdx.x = _tid - _workerIdx.y * _workerDim.x;

        _left = L + _workerIdx.x;
        _right = (_left + 1 > R ? R : _left + 1);

        _top = T - _workerIdx.y;
        _bottom = (_top - 1 < B ? B : _top - 1);

        _isWorker = true;
    }else {
        _isWorker = false;
    }
}

/**
 * @brief Assign a voxel for a thread when symmetrize C(HALF), symmetrize F
 *        (HALF) and normalize F(HALF).
 *
 * @param dim : volume dimension.
 * @param type: FULL means range of column equals [-nCol / 2, nCol / 2 - 1]
 *              PART means range of column equals [0, nCol / 2 - 1]
 *              HALF means range of column equals [0, nCol / 2]
 */
D_CALLABLE void Constructor::assignTask3D(int dim, TaskKind type)
{
    /* One thread handles one voxle:
     *   gridDim.y  = (nCol / 2 + 1), Index: i, Range: _front  --> _back
     *   gridDim.x  = nRow,           Index: j, Range: _left   --> _right
     *   blockDim.x = nSlc,           Index: k, Range: _bottom --> _top
     */

    if (type == TaskKind::FULL)
    {
        /* Test total number of threads first. */
        if (gridDim.y < dim || gridDim.x < dim || blockDim.x < dim) {
            printf("*** Launched threads grid error in assignTask3D FULL mode.\n");
            return ;
        }

        if (blockIdx.y < dim && blockIdx.x < dim && threadIdx.x < dim)
        {
            _bottom = threadIdx.x - dim / 2;
            _top = _bottom + 1;

            _left = blockIdx.x - dim / 2;
            _right = _left + 1;

            _front = blockIdx.y - dim / 2;
            _back  = _front + 1;

            _isWorker = true;
        }else {
            _isWorker = false;
        }
    }
    else if (type == TaskKind::PART)
    {
        /* Test total number of threads first. */
        if (gridDim.y < (dim / 2) || gridDim.x < dim || blockDim.x < dim) {
            printf("***Launched threads grid error in assignTask3D PART mode.\n");
            return ;
        }

        if (blockIdx.y < (dim / 2) && blockIdx.x < dim && threadIdx.x < dim) {
            _bottom = threadIdx.x - dim / 2;
            _top = _bottom + 1;

            _left = blockIdx.x - dim / 2;
            _right = _left + 1;

            _front = blockIdx.y;
            _back  = _front + 1;

            _isWorker = true;
        }else {
            _isWorker = false;
        }
    }
    else if (type == TaskKind::HALF)
    {
        /* Test total number of threads first. */
        if (gridDim.y < (dim / 2 + 1) || gridDim.x < dim || blockDim.x < dim)
        {
            printf("***Threads number error in assignTask3D HALF mode.\n");
            return ;
        }

        if (blockIdx.y < (dim / 2 + 1) && blockIdx.x < dim && threadIdx.x < dim)
        {
            _bottom = threadIdx.x - dim / 2;
            _top = _bottom + 1;

            _left = blockIdx.x - dim / 2;
            _right = _left + 1;

            _front = blockIdx.y;
            _back  = _front + 1;

            _isWorker = true;
        }else {
            _isWorker = false;
        }
    }
    else
    {
        printf("***Error task type in assignTask3D!\n");
    }
}

/**
 * @brief Assign a voxel for a thread when symmetrize C(HALF), symmetrize F
 *        (HALF) and normalize F(HALF).
 *
 * @param dim : volume dimension.
 * @param type: FULL means range of column equals [-nCol / 2, nCol / 2 - 1]
 *              PART means range of column equals [0, nCol / 2 - 1]
 *              HALF means range of column equals [0, nCol / 2]
 */
D_CALLABLE void Constructor::assignTask3DX(int dim, TaskKind type)
{
    /* One thread handles one voxle:
     *   blockDim.x = (nCol / 2 + 1), Index: i, Range: _front  --> _back
     *   gridDim.x  = nRow,           Index: j, Range: _left   --> _right
     *   gridDim.y  = nSlc,           Index: k, Range: _bottom --> _top
     */

    if (type == TaskKind::FULL)
    {
        /* Test total number of threads first. */
        if (gridDim.y < dim || gridDim.x < dim || blockDim.x < dim) {
            printf("*** Launched threads grid error in assignTask3D FULL mode.\n");
            return ;
        }

        if (blockIdx.y < dim && blockIdx.x < dim && threadIdx.x < dim)
        {
            _bottom = blockIdx.y - dim / 2;
            _top = _bottom + 1;

            _left = blockIdx.x - dim / 2;
            _right = _left + 1;

            _front = threadIdx.x - dim / 2;
            _back  = _front + 1;

            _isWorker = true;
        }else {
            _isWorker = false;
        }
    }
    else if (type == TaskKind::PART)
    {
        /* Test total number of threads first. */
        if (blockDim.x < (dim / 2) || gridDim.x < dim || gridDim.y < dim) {
            printf("***Launched threads grid error in assignTask3D PART mode.\n");
            return ;
        }

        if (threadIdx.x < (dim / 2) && blockIdx.x < dim && blockIdx.y < dim) {
            _bottom = blockIdx.y;
            _top = _bottom + 1;

            _left = blockIdx.x - dim / 2;
            _right = _left + 1;

            _front = threadIdx.x - dim / 2;
            _back  = _front + 1;

            _isWorker = true;
        }else {
            _isWorker = false;
        }
    }
    else if (type == TaskKind::HALF)
    {
        /* Test total number of threads first. */
        if (blockDim.x < (dim / 2 + 1) || gridDim.x < dim || gridDim.y < dim)
        {
            printf("***Threads number error in assignTask3D HALF mode.\n");
            return ;
        }

        if (threadIdx.x < (dim / 2 + 1) && blockIdx.x < dim && blockIdx.y < dim)
        {
            _bottom = blockIdx.y;
            _top = _bottom + 1;

            _left = blockIdx.x - dim / 2;
            _right = _left + 1;

            _front = threadIdx.x - dim / 2;
            _back  = _front + 1;

            _isWorker = true;
        }else {
            _isWorker = false;
        }
    }
    else
    {
        printf("***Error task type in assignTask3D!\n");
    }
}


/* @brief For normalizeF,
 *            normalizeT,
 *            CheckCAVG,
 *            CheckCMAX. 
 */
D_CALLABLE void Constructor::init(int tid)
{
    _tid = tid;
}

//////////////////////////////////////////////////////////////////////////

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::addSurfaceD(double value,
                                         int iCol,
                                         int iRow,
                                         int iSlc,
                                         const int dim,
                                         cudaSurfaceObject_t surfObject) const
{
    if (iRow < 0) iRow += dim;
    if (iSlc < 0) iSlc += dim;

    int2 dval;
    dval.y = __double2hiint(value);
    dval.x = __double2loint(value);

    surf3Dwrite(dval, surfObject, iCol, iRow, iSlc, cudaBoundaryModeTrap);
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::addSurfaceC(Complex value,
                                         int iCol,
                                         int iRow,
                                         int iSlc,
                                         const int dim,
                                         cudaSurfaceObject_t surfObject) const
{
    if (iRow < 0) iRow += dim;
    if (iSlc < 0) iSlc += dim;

    int4 cval;
    cval.w = __double2hiint(value.real());
    cval.z = __double2loint(value.imag());
    cval.y = __double2hiint(value.real());
    cval.x = __double2loint(value.imag());

    surf3Dwrite(cval, surfObject, iCol, iRow, iSlc, cudaBoundaryModeTrap);
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::addFTD(double* devDataT,
                                    double value,
                                    double iCol,
                                    double iRow,
                                    double iSlc,
                                    const int dim) const
{
    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
    }

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};
    int index;

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {    
                index = getIndexHalf(x0[0] + i,
                                     x0[1] + j,
                                     x0[2] + k,
                                     dim);
                //if (index < 0 || index >= dim * dim * (dim / 2 + 1))
                //    printf("index error!\n");
                atomicAdd(&devDataT[index], value * w[k][j][i]);
            } 
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::addFTC(Complex* devDataF,
                                    Complex& value,
                                    double iCol,
                                    double iRow,
                                    double iSlc,
                                    const int dim) const
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
        conjug = true;
    }

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};
    int index;

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    conjug ? value.conj() : value;

    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
                index = getIndexHalf(x0[0] + i,
                                     x0[1] + j,
                                     x0[2] + k,
                                     dim);

                //if (index < 0 || index >= dim * dim * (dim / 2 + 1))
                //    printf("index error!\n");
                atomicAdd(devDataF[index].realAddr(), value.real() * w[k][j][i]);
                atomicAdd(devDataF[index].imagAddr(), value.imag() * w[k][j][i]);
            } 
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::addFTCD(Complex* devDataF,
                                     double* devDataT,
                                     Complex& cvalue,
                                     double dvalue,
                                     double iCol,
                                     double iRow,
                                     double iSlc,
                                     const int dim) const
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
        conjug = true;
    }

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};
    int index;

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    conjug ? cvalue.conj() : cvalue;

    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++)
            {
                index = getIndexHalf(x0[0] + i,
                                     x0[1] + j,
                                     x0[2] + k,
                                     dim);
                if (index < 0 || index >= dim * dim * (dim / 2 + 1))
                    printf("index error!\n");
                atomicAdd(devDataF[index].realAddr(), cvalue.real() * w[k][j][i]);
                atomicAdd(devDataF[index].imagAddr(), cvalue.imag() * w[k][j][i]);
                atomicAdd(&devDataT[index], dvalue * w[k][j][i]);
            } 
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::expectPrectf(CTFAttr* dev_ctfa,
                                          double* dev_def,
                                          double* dev_k1,
                                          double* dev_k2,
                                          int* deviCol,
                                          int* deviRow,
                                          int imgidx,
                                          int blockSize,
                                          int npxl)
{
    int i, j;
    int shift = imgidx * npxl; 
    double lambda, angle;
    
    lambda = 12.2643274 / sqrt(dev_ctfa[imgidx].voltage 
                        * (1 + dev_ctfa[imgidx].voltage * 0.978466e-6));
    dev_k1[imgidx] = PI * lambda;
    dev_k2[imgidx] = PI / 2 * dev_ctfa[imgidx].Cs * lambda * lambda * lambda;
    
    for (int itr = _tid; itr < npxl; itr += blockSize)
    {
        i = deviCol[itr];
        j = deviRow[itr];
        
        angle = atan2((double)j, (double)i) - dev_ctfa[imgidx].defocusTheta;
        
        dev_def[shift + itr] = -(dev_ctfa[imgidx].defocusU 
                                 + dev_ctfa[imgidx].defocusV 
                                 + (dev_ctfa[imgidx].defocusU - dev_ctfa[imgidx].defocusV) 
                                 * cos(2 * angle)) / 2;
    }

}
        
/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::translate(Complex* devtraP,
                                       double* dev_trans,
                                       int* deviCol,
                                       int* deviRow,
                                       int idim,
                                       int npxl,
                                       int imgidx,
                                       int blockSize)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::getRotMat(double* devRotm,
                                       double* matS,
                                       double* devnR,
                                       int nR,
                                       int threadId)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::project(Complex* priRotP,
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
                                     cudaTextureObject_t texObject)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::logDataVS(Complex* devdatP,
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
                                       int npxl)
{
}
        
/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::updateW(double* devDvp,
                                     double* devbaseL,
                                     double* devwC,
                                     double* devwR,
                                     double* devwT,
                                     int rIdx,
                                     int nK,
                                     int nR,
                                     int nT,
                                     int rSize)
{
}
        
/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::getRandomCTD(double* dev_nt,
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
                                          int blockId)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::getRandomCTD(double* dev_nt,
                                          double* dev_tran,
                                          double* dev_nr,
                                          double* dev_ramR,
                                          int* dev_ramC,
                                          unsigned long out,
                                          int nC,
                                          int nR,
                                          int nT,
                                          int blockId)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::getRandomR(double* dev_mat,
                                        double* matS,
                                        double* dev_ramR,
                                        int threadId)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::translate(Complex* dev_images,
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
                                       int imgSize)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::translate(Complex* dev_images,
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
                                       int imgSize)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::calculateCTF(Complex* dev_ctfs,
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
                                          int imgSize)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::insertT(double* devDataT,
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
                                     int imgSize)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::insertF(Complex* devDataF,
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
                                     int imgSize)
{
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::normalizeT(double *devDataT, 
                                        const int length,
                                        const int num, 
                                        const double sf)
{
    int index = _tid;
    while(index < length)
    {
        devDataT[index + num] *= sf;
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE double Constructor::getTexture(double iCol,
                                          double iRow,
                                          double iSlc,
                                          const int dim,
                                          cudaTextureObject_t texObject) const
{
    if (iRow < 0) iRow += dim;
    if (iSlc < 0) iSlc += dim;

    int2 dval = tex3D<int2>(texObject, iCol, iRow, iSlc);

    return fetch_double(dval.y, dval.x);
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE double Constructor::getByInterpolationFT(double iCol,
                                                    double iRow,
                                                    double iSlc,
                                                    const int interp,
                                                    const int dim,
                                                    cudaTextureObject_t texObject) const
{
    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
    }

    if(interp == 0)
    {
        double result = getTexture(iCol, 
                                   iRow,
                                   iSlc,
                                   dim,
                                   texObject);
        return result;
    }

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    double result = 0.0;
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++){
                result += getTexture((double)x0[0] + i, 
                                     (double)x0[1] + j, 
                                     (double)x0[2] + k, 
                                     dim, 
                                     texObject)
                        * w[k][j][i];
            } 
    return result;
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::symmetrizeT(double *devDataT,
                                         double *matBase, 
                                         double r, 
                                         int numSymmat,
                                         int interp,
                                         int num,
                                         int dim,
                                         int dimSize,
                                         cudaTextureObject_t texObject)
{
    int i, j, k;
    int index = _tid;

    while(index < dimSize)
    {
        i = (index + num) % (dim / 2 + 1);
        j = ((index + num) / (dim / 2 + 1)) % dim;
        k = ((index + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        for (int m = 0 ; m < numSymmat; m++)
        {
            _mat.init(matBase, m);
            
            double inc = 0.0;
            
            Vec3 newCor((double)i, (double)j, (double)k);
            Vec3 oldCor = _mat * newCor;
            
            if (rint(oldCor.squaredNorm3()) < rint(r * r))
            {
                inc = getByInterpolationFT(oldCor(0),
                                           oldCor(1),
                                           oldCor(2),
                                           interp,
                                           dim,
                                           texObject);
                
                devDataT[index + num] += inc;
            }

        }        
        
        index += blockDim.x * gridDim.x;
    }

}


/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::shellAverage(double *devAvg2D, 
                                          int *devCount2D, 
                                          double *devDataT,
                                          double *sumAvg,
                                          int *sumCount, 
                                          int r,
                                          int dim,
                                          int dimSize,
                                          int indexDiff,
                                          int blockId)
{
    int i, j, k;
    int index = _tid;
    sumAvg[indexDiff] = 0;
    sumCount[indexDiff] = 0;

    __syncthreads();

    while(index < dimSize)
    {
        i = index % (dim / 2 + 1);
        j = (index / (dim / 2 + 1)) % dim;
        k = (index / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        int quad = i * i + j * j + k * k;
        
        if(quad < r * r)
        {
            int u = (int)rint(sqrt((double)quad));
            
            atomicAdd(&sumAvg[u], devDataT[index]);
            atomicAdd(&sumCount[u], 1);
        }
        index += blockDim.x * gridDim.x;
    }

    __syncthreads();

    if (indexDiff < r) 
    {
        devAvg2D[indexDiff + blockId * r] = sumAvg[indexDiff];
        devCount2D[indexDiff + blockId * r] = sumCount[indexDiff];
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::calculateAvg(double *devAvg2D, 
                                          int *devCount2D,
                                          double *devAvg,
                                          int *devCount,
                                          int dim, 
                                          int r)
{
    int index = _tid;

    devAvg[index] = 0;
    devCount[index] = 0;
    
    __syncthreads();

    for (int i = 0; i < dim; i++)
    {
        if (index < r)
        {
            devAvg[index] += devAvg2D[index + i * r];
            devCount[index] += devCount2D[index + i * r];
        }
    }

    if (index < r)
        devAvg[index] /= devCount[index];

    if(index == r - 1 ){
        devAvg[r] = devAvg[r - 1];
        devAvg[r + 1] = devAvg[r - 1];
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::calculateFSC(double *devFSC,
                                          double *devAvg, 
                                          double *devDataT,
                                          int num,
                                          int dim,
                                          int dimSize,
                                          int fscMatsize,
                                          int wiener,
                                          int r,
                                          bool joinHalf)
{
    int i, j, k;
    int index = _tid;

    while(index < dimSize)
    {
        i = (index + num) % (dim / 2 + 1);
        j = ((index + num) / (dim / 2 + 1)) % dim;
        k = ((index + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        int quad = i * i + j * j + k * k;
        
        if(quad >= wiener && quad < r)
        {
            int u = (int)rint(sqrt((double)quad));
            
            double FSC = (u / _pf >= fscMatsize)
                       ? 0 
                       : devFSC[u / _pf];
            
            FSC = fmax(1e-3, fmin(1 - 1e-3, FSC));

#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrt(2 * FSC / (1 + FSC));
#else
            if (joinHalf) 
                FSC = sqrt(2 * FSC / (1 + FSC));
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
            devDataT[index + num] += (1 - FSC) / FSC * devAvg[u];
#else
            devDataT[index + num] /= FSC;
#endif
            
        }
        index += blockDim.x * gridDim.x;
    }

}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::wienerConst(double *devDataT,
                                         int wiener,
                                         int r,
                                         int num,
                                         int dim,
                                         int dimSize)
{
    int i, j, k;
    int index = _tid;

    while(index < dimSize)
    {
        i = (index + num) % (dim / 2 + 1);
        j = ((index + num) / (dim / 2 + 1)) % dim;
        k = ((index + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        int quad = i * i + j * j + k * k;

        if (quad >= wiener && quad < r)
        {
            devDataT[index + num] = devDataT[index + num] + 1;
        }
        
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::calculateW(double *devDataW, 
                                        double *devDataT,
                                        const int length,
                                        const int num,
                                        const int dim, 
                                        const int r)
{
    int i, j, k;
    int index = _tid;

    while(index < length)
    {
        i = (index + num) % (dim / 2 + 1);
        j = ((index + num) / (dim / 2 + 1)) % dim;
        k = ((index + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        int quad = i * i + j * j + k * k;

        if (quad < r)
        {
            devDataW[index + num] = 1.0 / fmax(fabs(devDataT[index + num]), 1e-6);
        }
        
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::initialW(double *devDataW,
                                      int initWR,
                                      int dim,
                                      int dimSize)
{
    int i, j, k;
    int index = _tid;

    while(index < dimSize)
    {
        i = (index) % (dim / 2 + 1);
        j = ((index) / (dim / 2 + 1)) % dim;
        k = ((index) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        int quad = i * i + j * j + k * k;

        if (quad < initWR)
        {
            devDataW[index] = 1;
        }
        else
        {
            devDataW[index] = 0;
        }
        
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::determiningC(Complex *devDataC,
                                          double *devDataT,
                                          double *devDataW,
                                          int length)
{
    int index = _tid;
    while(index < length)
    {
        devDataC[index].set(devDataT[index] * devDataW[index], 0);
        index += blockDim.x * gridDim.x;
    }
    
}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::convoluteC(double *devDoubleC,
                                        TabFunction &tabfunc,
                                        double nf,
                                        int padSize,
                                        int dim,
                                        int dimSize)
{
    int index = _tid;
    int i, j, k;
    
    while(index < dimSize)
    {
        i = index % dim;
        j = (index / dim) % dim;
        k = (index / dim) / dim;

        if(i >= dim / 2) i = i - dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        devDoubleC[index] = devDoubleC[index] 
                          / dimSize
                          * tabfunc((i*i + j*j + k*k) / pow(padSize, 2))
                          / nf;

        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::recalculateW(Complex *devDataC,
                                          double *devDataW,
                                          int initWR,
                                          int dim,
                                          int dimSize)
{
    int i, j, k;
    int index = _tid;

    double mode = 0.0;
    while(index < dimSize)
    {
        i = (index) % (dim / 2 + 1);
        j = ((index) / (dim / 2 + 1)) % dim;
        k = ((index) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        int quad = i * i + j * j + k * k;

        if (quad < initWR)
        {
            mode = devDataC[index].real() * devDataC[index].real()
                 + devDataC[index].imag() * devDataC[index].imag();

            devDataW[index] /= fmax(sqrt(mode), 1e-6);
        }
        
        index += blockDim.x * gridDim.x;
    }

}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::checkCAVG(double *sumDiff,
                                       double *sumCount,
                                       double *diff,
                                       double *counter,
                                       Complex *devDataC,
                                       int r,
                                       int dim,
                                       int dimSize,
                                       int indexDiff,
                                       int blockId)
{
    double tempD = 0, mode = 0;
    int tempC = 0;
    int index = _tid;
    int i, j, k;
    bool flag = true;
    
    while(index < dimSize)
    {
        i = index % (dim / 2 + 1);
        j = (index / (dim / 2 + 1)) % dim;
        k = (index / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        int quad = i * i + j * j + k * k;
        
        if(quad < r)
        {
            mode = devDataC[index].real() 
                 * devDataC[index].real()
                 + devDataC[index].imag()
                 * devDataC[index].imag();

            tempD += fabs(sqrt(mode) - 1);
            tempC += 1;
        }
        index += blockDim.x * gridDim.x;
    }

    sumDiff[indexDiff] = tempD;
    sumCount[indexDiff] = tempC;

    __syncthreads();

    if (dim % 2 == 0)
    {
        i = dim / 2;
        flag = true;
    }
    else
    {
        i = dim / 2 + 1;
        flag = false;
    }
    while (i != 0) 
    {
        if (flag)
        {
            if (indexDiff < i)
            {
                sumDiff[indexDiff] += sumDiff[indexDiff + i];
                sumCount[indexDiff] += sumCount[indexDiff + i];
            }
        
        }
        else
        {
            if (indexDiff < i - 1)
            {
                sumDiff[indexDiff] += sumDiff[indexDiff + i];
                sumCount[indexDiff] += sumCount[indexDiff + i];
            }
        
        }
        
        __syncthreads();
        
        if(i % 2 != 0 && i != 1)
        {
            i++;
            flag = false;
        }
        else
            flag = true;
        
        i /= 2; 
    }
    
    if (indexDiff == 0) 
    {

        diff[blockId] = sumDiff[0];
        counter[blockId] = sumCount[0];
    }
  
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::checkCMAX(double *singleMax,
                                       double *devMax,
                                       Complex *devDataC,
                                       int r,
                                       int dim,
                                       int dimSize,
                                       int indexDiff,
                                       int blockId)
{
    int index = _tid;
    int i, j, k;
    double temp = 0.0, mode = 0.0;;
    bool flag = true;
    
    while(index < dimSize)
    {
        i = index % (dim / 2 + 1);
        j = (index / (dim / 2 + 1)) % dim;
        k = (index / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        int quad = i * i + j * j + k * k;
        
        if(quad < r)
        {
            mode = sqrt(devDataC[index].real() 
                        * devDataC[index].real()
                        + devDataC[index].imag()
                        * devDataC[index].imag());

            if (fabs(mode - 1) >= temp)
                temp = fabs(mode - 1);
        }
        index += blockDim.x * gridDim.x;
    }
    
    singleMax[indexDiff] = temp;

    __syncthreads();

    if (dim % 2 == 0)
    {
        i = dim / 2;
        flag = true;
    }
    else
    {
        i = dim / 2 + 1;
        flag = false;
    }
    
    while (i != 0) 
    {
        if (flag)
        {
            if (indexDiff < i)
            {
                if (singleMax[indexDiff] < singleMax[indexDiff + i])
                {
                    singleMax[indexDiff] = singleMax[indexDiff + i]; 
                }
            }
                
        }
        else
        {
            if (indexDiff < i - 1)
            {
                if (singleMax[indexDiff] < singleMax[indexDiff + i])
                {
                    singleMax[indexDiff] = singleMax[indexDiff + i]; 
                }
            }
        
        }
        
        __syncthreads();
        
        if(i % 2 != 0 && i != 1)
        {
            i++;
            flag = false;
        }
        else
            flag = true;
        i /= 2;
    }
    
    if (indexDiff == 0) 
    {
        devMax[blockId] = singleMax[0];
    }

}
/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::normalizeF(Complex *devDataF, 
                                        const int length, 
                                        const int num, 
                                        const double sf)
{
    int index = _tid;

    while(index < length)
    {
        devDataF[index + num] *= sf;
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex Constructor::getTextureC(double iCol,
                                            double iRow,
                                            double iSlc,
                                            const int dim,
                                            cudaTextureObject_t texObject) const
{
    if (iRow < 0) iRow += dim;
    if (iSlc < 0) iSlc += dim;

    int4 cval = tex3D<int4>(texObject, iCol, iRow, iSlc);

    return Complex(fetch_double(cval.y,cval.x),
                   fetch_double(cval.w,cval.z));
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex Constructor::getByInterpolationFTC(double iCol,
                                                      double iRow,
                                                      double iSlc,
                                                      const int interp,
                                                      const int dim,
                                                      cudaTextureObject_t texObject) const
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
        conjug = true;
    }

    if(interp == 0)
    {
        Complex result = getTextureC(iCol, 
                                     iRow,
                                     iSlc,
                                     dim,
                                     texObject);
        return conjug ? result.conj() : result;
    }

    double w[2][2][2];
    int x0[3];
    double x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    Complex result (0.0, 0.0);
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++){
                
                result += getTextureC((double)x0[0] + i, 
                                      (double)x0[1] + j, 
                                      (double)x0[2] + k, 
                                      dim, 
                                      texObject)
                       * w[k][j][i];
            } 
    return conjug ? result.conj() : result;
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::symmetrizeF(Complex *devDataF,
                                         double *matBase, 
                                         double r, 
                                         int numSymmat,
                                         int interp,
                                         int num,
                                         int dim,
                                         int dimSize,
                                         cudaTextureObject_t texObject)
{
    int i, j, k;
    int index = _tid;

    while(index < dimSize)
    {
        i = (index + num) % (dim / 2 + 1);
        j = ((index + num) / (dim / 2 + 1)) % dim;
        k = ((index + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        for (int m = 0 ; m < numSymmat; m++)
        {
            _mat.init(matBase, m);
            
            Complex inc(0.0, 0.0);
            
            Vec3 newCor((double)i, (double)j, (double)k);
            Vec3 oldCor = _mat * newCor;
            
            if (rint(oldCor.squaredNorm3()) < rint(r * r))
            {
                
                inc = getByInterpolationFTC(oldCor(0),
                                            oldCor(1),
                                            oldCor(2),
                                            interp,
                                            dim,
                                            texObject);
                
                devDataF[index + num] += inc;
            }

        }        
        index += blockDim.x * gridDim.x;
    }

}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::normalizeFW(Complex *F3D,
                                         double *devDataW,
                                         const int length, 
                                         const int shiftF,
                                         const int shiftW)
{
    int index = _tid;

    while(index < length)
    {
        F3D[index + shiftF] *= devDataW[index + shiftW];
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::lowpassF(Complex *devDataF, 
                                      const double thres, 
                                      const double ew,
                                      const int num,
                                      const int dim,
                                      const int dimSize)
{
    int i, j, k;
    int index = _tid;

    while(index < dimSize)
    {
        i = (index + num) % (dim / 2 + 1);
        j = ((index + num) / (dim / 2 + 1)) % dim;
        k = ((index + num) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        double f = norm3d((double)i / dim, 
                          (double)j / dim, 
                          (double)k / dim);
        
        if (f > thres + ew)
        {
            devDataF[index + num] = 0;
        }
        else
        {
            if (f >= thres)
            {
                devDataF[index + num] *= (cos((f - thres) * PI / ew) / 2 + 0.5);
            }
        }
        index += blockDim.x * gridDim.x;
    }

}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::correctF(double *devDst,
                                      double *devMkb,
                                      double nf,
                                      int dim,
                                      int dimSize,
                                      int shift) 
{
    int index = _tid;
    int i, j, k, mkbIndex;

    while(index < dimSize)
    {
        i = (index + shift) % dim;
        j = ((index + shift) / dim) % dim;
        k = ((index + shift) / dim) / dim;

        if(i >= dim / 2) i = dim - i;
        if(j >= dim / 2) j = dim - j;
        if(k >= dim / 2) k = dim - k;
        
        mkbIndex = k * (dim / 2 + 1) * (dim / 2 + 1) + j * (dim / 2 + 1) + i;

        devDst[index + shift] = devDst[index + shift] 
                              / devMkb[mkbIndex]
                              * nf;
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::correctF(double *devDst,
                                      double *devTik,
                                      int dim,
                                      int dimSize,
                                      int shift)
{
    int index = _tid;
    int i, j, k, mkbIndex;
    
    while(index < dimSize)
    {
        i = (index + shift) % dim;
        j = ((index + shift) / dim) % dim;
        k = ((index + shift) / dim) / dim;

        if(i >= dim / 2) i = dim - i;
        if(j >= dim / 2) j = dim - j;
        if(k >= dim / 2) k = dim - k;
        
        mkbIndex = k * (dim / 2 + 1) * (dim / 2 + 1) + j * (dim / 2 + 1) + i;
        
        devDst[index + shift] /= devTik[mkbIndex];

        index += blockDim.x * gridDim.x;
    }
}


/**
  @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::background(double *devDst,
                                        double *sumWeight,
                                        double *sumS,
                                        double *devSumG,
                                        double *devSumWG,
                                        double r,
                                        double ew,
                                        int dim,
                                        int dimSize,
                                        int indexDiff,
                                        int blockId)
{
    double weightSum = 0;
    double sumt = 0;
    int index = _tid;
    int i, j, k;
    bool flag = true;

    while(index < dimSize)
    {
        i = index % dim;
        j = (index / dim) % dim;
        k = (index / dim) / dim;

        if(i >= dim / 2) i = i - dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;
        
        double u = norm3d((double)i, 
                          (double)j, 
                          (double)k);
        
        if(u > r + ew)
        {
            weightSum += 1;
            sumt += devDst[index];
        }
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * PI); 
            
            weightSum += w;
            sumt += devDst[index] * w;
        }
        index += blockDim.x * gridDim.x;
    }

    sumWeight[indexDiff] = weightSum;
    sumS[indexDiff] = sumt;

    __syncthreads();

    if (dim % 2 == 0)
    {
        i = dim / 2;
        flag = true;
    
    }
    else
    {
         i = dim / 2 + 1;
         flag = false;
    
    } 

    while (i != 0) 
    {
        if (flag)
        {
            if (indexDiff < i)
            {
                sumWeight[indexDiff] += sumWeight[indexDiff + i];
                sumS[indexDiff] += sumS[indexDiff + i];
            }
        }
        else
        {
            if (indexDiff < i - 1)
            {
                sumWeight[indexDiff] += sumWeight[indexDiff + i];
                sumS[indexDiff] += sumS[indexDiff + i];
            }
        
        }
        __syncthreads();
        
        if(i % 2 != 0 && i != 1)
        {
             i++;
             flag = false;
        }
        else
            flag = true;

        i /= 2; 
    }
    
    if (indexDiff == 0) 
    {
        devSumG[blockId] = sumS[0];
        devSumWG[blockId] = sumWeight[0];
    }
    
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::calculateBg(double *devSumG, 
                                         double *devSumWG,
                                         double *bg,
                                         int dim)
{
    int index = _tid;
    int i = 0;
    int flag =true;

    if (dim % 2 == 0)
    {
        i = dim / 2;
        flag = true;
    
    }
    else
    {
         i = dim / 2 + 1;
         flag = false;
    
    } 

    while (i != 0) 
    {
        if (flag)
        {
            if (index < i)
            {
                devSumWG[index] += devSumWG[index + i];
                devSumG[index] += devSumG[index + i];
            }
        }
        else
        {
            if (index < i - 1)
            {
                devSumWG[index] += devSumWG[index + i];
                devSumG[index] += devSumG[index + i];
            }
        
        }
        __syncthreads();
        
        if(i % 2 != 0 && i != 1)
        {
             i++;
             flag = false;
        }
        else
            flag = true;

        i /= 2; 
    }
    
    if (index == 0) 
    {
        bg[0] = devSumG[0] / devSumWG[0];
    }
}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::softMaskD(double *devDst,
                                       double *bg,
                                       double r,
                                       double ew,
                                       int dim,
                                       int dimSize,
                                       int shift)
{
    int index = _tid;
    int i, j, k;
    
    while(index < dimSize)
    {
        i = (index + shift) % dim;
        j = ((index + shift) / dim) % dim;
        k = ((index + shift) / dim) / dim;

        if(i >= dim / 2) i = dim - i;
        if(j >= dim / 2) j = dim - j;
        if(k >= dim / 2) k = dim - k;
        
        double u = norm3d((double)i, (double)j, (double)k);
        
        if (u > r + ew)
            devDst[index + shift] = bg[0];
        else if (u >= r)
        {
            double w = 0.5 - 0.5 * cos((u - r) / ew * PI);
            devDst[index + shift] = bg[0] * w + devDst[index + shift] * (1 - w);
        }
        
        index += blockDim.x * gridDim.x;
    }
}

//////////////////////////////////////////////////////////////////////////

} // end namespace cuthunder

//////////////////////////////////////////////////////////////////////////
