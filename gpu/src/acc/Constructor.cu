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

HD_CALLABLE Constructor::Constructor(Volume vol,
                                     ImageStream imgss,
                                     TabFunction tabfunc,
                                     double dimsize,
                                     double weight,
                                     double alpha,
                                     int pf)
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
D_CALLABLE void Constructor::expectPrectf(CTFAttr* dev_ctfa,
                                          RFLOAT* dev_def,
                                          RFLOAT* dev_k1,
                                          RFLOAT* dev_k2,
                                          int* deviCol,
                                          int* deviRow,
                                          int imgidx,
                                          int blockSize,
                                          int npxl)
{
    int i, j;
    int shift = imgidx * npxl; 
    double lambda, angle;
    
#ifdef SINGLE_PRECISION
    lambda = 12.2643274 / sqrtf(dev_ctfa[imgidx].voltage 
                        * (1 + dev_ctfa[imgidx].voltage * 0.978466e-6));
#else
    lambda = 12.2643274 / sqrt(dev_ctfa[imgidx].voltage 
                        * (1 + dev_ctfa[imgidx].voltage * 0.978466e-6));
#endif    
    dev_k1[imgidx] = PI * lambda;
    dev_k2[imgidx] = PI / 2 * dev_ctfa[imgidx].Cs * lambda * lambda * lambda;
    
    for (int itr = _tid; itr < npxl; itr += blockSize)
    {
        i = deviCol[itr];
        j = deviRow[itr];
        
#ifdef SINGLE_PRECISION
        angle = atan2f((float)j, (float)i) - dev_ctfa[imgidx].defocusTheta;
        
        dev_def[shift + itr] = -(dev_ctfa[imgidx].defocusU 
                                 + dev_ctfa[imgidx].defocusV 
                                 + (dev_ctfa[imgidx].defocusU - dev_ctfa[imgidx].defocusV) 
                                 * cosf(2 * angle)) / 2;
#else
        angle = atan2((double)j, (double)i) - dev_ctfa[imgidx].defocusTheta;
        
        dev_def[shift + itr] = -(dev_ctfa[imgidx].defocusU 
                                 + dev_ctfa[imgidx].defocusV 
                                 + (dev_ctfa[imgidx].defocusU - dev_ctfa[imgidx].defocusV) 
                                 * cos(2 * angle)) / 2;
#endif    
    }

}
        
/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::normalizeTF(Complex *devDataF,
                                         RFLOAT *devDataT, 
                                         const int length,
                                         const int num, 
                                         const RFLOAT sf)
{
    int index = _tid;
    while(index < length)
    {
        devDataF[index + num] *= sf;
        devDataT[index] *= sf;
        
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::normalizeT(RFLOAT *devDataT, 
                                        const int length,
                                        const int num, 
                                        const RFLOAT sf)
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
D_CALLABLE RFLOAT Constructor::getTexture(RFLOAT iCol,
                                          RFLOAT iRow,
                                          RFLOAT iSlc,
                                          const int dim,
                                          cudaTextureObject_t texObject) const
{
    if (iRow < 0) iRow += dim;
    if (iSlc < 0) iSlc += dim;

#ifdef SINGLE_PRECISION
    float dval = tex3D<float>(texObject, iCol, iRow, iSlc);
    return dval;
#else
    int2 dval = tex3D<int2>(texObject, iCol, iRow, iSlc);
    return __hiloint2double(dval.y, dval.x);
#endif    
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE RFLOAT Constructor::getByInterpolationFT(RFLOAT iCol,
                                                    RFLOAT iRow,
                                                    RFLOAT iSlc,
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
        RFLOAT result = getTexture(iCol, 
                                   iRow,
                                   iSlc,
                                   dim,
                                   texObject);
        return result;
    }

    RFLOAT w[2][2][2];
    int x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    RFLOAT result = 0.0;
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++){
                result += getTexture((RFLOAT)x0[0] + i, 
                                     (RFLOAT)x0[1] + j, 
                                     (RFLOAT)x0[2] + k, 
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
D_CALLABLE void Constructor::symmetrizeT(RFLOAT *devDataT,
                                         double *matBase, 
                                         int r, 
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
            
            RFLOAT inc = 0.0;
            
            Vec3 newCor((double)i, (double)j, (double)k);
            Vec3 oldCor = _mat * newCor;
            
#ifdef SINGLE_PRECISION
            if ((int)floorf(oldCor.squaredNorm3()) < r)
            {
                inc = getByInterpolationFT((RFLOAT)oldCor(0),
                                           (RFLOAT)oldCor(1),
                                           (RFLOAT)oldCor(2),
                                           interp,
                                           dim,
                                           texObject);
                
                devDataT[index + num] += inc;
            }
#else
            if ((int)floor(oldCor.squaredNorm3()) < r)
            {
                inc = getByInterpolationFT((RFLOAT)oldCor(0),
                                           (RFLOAT)oldCor(1),
                                           (RFLOAT)oldCor(2),
                                           interp,
                                           dim,
                                           texObject);
                
                devDataT[index + num] += inc;
            }
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
D_CALLABLE void Constructor::shellAverage(RFLOAT *devAvg2D, 
                                          int *devCount2D, 
                                          RFLOAT *devDataT,
                                          RFLOAT *sumAvg,
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
#ifdef SINGLE_PRECISION
            int u = (int)rintf(sqrtf((float)quad));
#else
            int u = (int)rint(sqrt((double)quad));
#endif    
            if (u < r)
            {
                atomicAdd(&sumAvg[u], devDataT[index]);
                atomicAdd(&sumCount[u], 1);
            }
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
D_CALLABLE void Constructor::calculateAvg(RFLOAT *devAvg2D, 
                                          int *devCount2D,
                                          RFLOAT *devAvg,
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
D_CALLABLE void Constructor::calculateFSC(RFLOAT *devFSC,
                                          RFLOAT *devAvg, 
                                          RFLOAT *devDataT,
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
#ifdef SINGLE_PRECISION
            int u = (int)rintf(sqrtf((float)quad));
            float FSC = (u / _pf >= fscMatsize)
                      ? 0 
                      : devFSC[u / _pf];
            
            FSC = fmaxf(1e-3, fminf(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
            FSC = sqrtf(2 * FSC / (1 + FSC));
#else
            if (joinHalf) 
                FSC = sqrtf(2 * FSC / (1 + FSC));
#endif

#else
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
D_CALLABLE void Constructor::wienerConst(RFLOAT *devDataT,
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
D_CALLABLE void Constructor::calculateW(RFLOAT *devDataW, 
                                        RFLOAT *devDataT,
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
#ifdef SINGLE_PRECISION
            devDataW[index + num] = 1.0 / fmaxf(fabsf(devDataT[index + num]), 1e-6);
#else
            devDataW[index + num] = 1.0 / fmax(fabs(devDataT[index + num]), 1e-6);
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
D_CALLABLE void Constructor::initialW(RFLOAT *devDataW,
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
                                          RFLOAT *devDataT,
                                          RFLOAT *devDataW,
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
D_CALLABLE void Constructor::convoluteC(RFLOAT *devDoubleC,
                                        TabFunction &tabfunc,
                                        RFLOAT nf,
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
                          * tabfunc((RFLOAT)(i*i + j*j + k*k) / (padSize * padSize))
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
                                          RFLOAT *devDataW,
                                          int initWR,
                                          int dim,
                                          int dimSize)
{
    int i, j, k;
    int index = _tid;

    RFLOAT mode = 0.0, u, x, y;
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
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[index].real());
            y = fabsf(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            devDataW[index] /= fmaxf(mode, 1e-6);
#else
            x = fabs(devDataC[index].real());
            y = fabs(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            devDataW[index] /= fmax(mode, 1e-6);
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
D_CALLABLE void Constructor::checkCAVG(RFLOAT *sumDiff,
                                       int *sumCount,
                                       RFLOAT *diff,
                                       int *counter,
                                       Complex *devDataC,
                                       int r,
                                       int dim,
                                       int dimSize,
                                       int indexDiff,
                                       int blockId)
{
    RFLOAT mode = 0, u, x, y;
    int index = _tid;
    int i, j, k;
    bool flag = true;
    
    sumDiff[indexDiff] = 0;
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
        
        if(quad < r)
        {
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[index].real());
            y = fabsf(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            sumDiff[indexDiff] += fabsf(mode - 1);
#else
            x = fabs(devDataC[index].real());
            y = fabs(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            sumDiff[indexDiff] += fabs(mode - 1);
#endif    
            sumCount[indexDiff] += 1;
        }
        index += blockDim.x * gridDim.x;
    }

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
D_CALLABLE void Constructor::checkCMAX(RFLOAT *singleMax,
                                       RFLOAT *devMax,
                                       Complex *devDataC,
                                       int r,
                                       int dim,
                                       int dimSize,
                                       int indexDiff,
                                       int blockId)
{
    int index = _tid;
    int i, j, k;
    RFLOAT temp = 0.0, mode = 0.0, u, x, y;
    bool flag = true;
    
    singleMax[indexDiff] = 0;

    __syncthreads();

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
#ifdef SINGLE_PRECISION
            x = fabsf(devDataC[index].real());
            y = fabsf(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrtf(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrtf(1 + u * u);
                }
            }

            if (fabsf(mode - 1) >= temp)
                temp = fabsf(mode - 1);
#else
            x = fabs(devDataC[index].real());
            y = fabs(devDataC[index].imag());
            if (x < y)
            {
                if (x == 0)
                    mode = y;
                else
                {
                    u = x / y;
                    mode = y * sqrt(1 + u * u);
                }
            }
            else
            {
                if (y == 0)
                    mode = x;
                else
                {
                    u = y / x;
                    mode = x * sqrt(1 + u * u);
                }
            }

            if (fabs(mode - 1) >= temp)
                temp = fabs(mode - 1);
#endif    
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
                                        const RFLOAT sf)
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
D_CALLABLE Complex Constructor::getTextureC(RFLOAT iCol,
                                            RFLOAT iRow,
                                            RFLOAT iSlc,
                                            const int dim,
                                            cudaTextureObject_t texObject) const
{
    if (iRow < 0) iRow += dim;
    if (iSlc < 0) iSlc += dim;

#ifdef SINGLE_PRECISION
    float2 cval = tex3D<float2>(texObject, iCol, iRow, iSlc);
    return Complex(cval.x, cval.y);
#else
    int4 cval = tex3D<int4>(texObject, iCol, iRow, iSlc);
    return Complex(__hiloint2double(cval.y,cval.x),
                   __hiloint2double(cval.w,cval.z));
#endif    
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex Constructor::getByInterpolationFTC(RFLOAT iCol,
                                                      RFLOAT iRow,
                                                      RFLOAT iSlc,
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

    RFLOAT w[2][2][2];
    int x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};

    WG_TRI_LINEAR_INTERPF(w, x0, x);

    Complex result (0.0, 0.0);
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 2; i++){
                
                result += getTextureC((RFLOAT)x0[0] + i, 
                                      (RFLOAT)x0[1] + j, 
                                      (RFLOAT)x0[2] + k, 
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
                                         int r, 
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
#ifdef SINGLE_PRECISION
            if ((int)floorf(oldCor.squaredNorm3()) < r)
            {
                inc = getByInterpolationFTC((RFLOAT)oldCor(0),
                                            (RFLOAT)oldCor(1),
                                            (RFLOAT)oldCor(2),
                                            interp,
                                            dim,
                                            texObject);
                
                devDataF[index + num] += inc;
            }
#else
            if ((int)floor(oldCor.squaredNorm3()) < r)
            {
                inc = getByInterpolationFTC((RFLOAT)oldCor(0),
                                            (RFLOAT)oldCor(1),
                                            (RFLOAT)oldCor(2),
                                            interp,
                                            dim,
                                            texObject);
                
                devDataF[index + num] += inc;
            }
#endif    
        }        
        index += blockDim.x * gridDim.x;
    }

}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::normalizeFW(Complex *devDst,
                                         Complex *devDataF,
                                         RFLOAT *devDataW,
                                         const int dimSize, 
                                         const int num,
                                         const int r,
                                         const int pdim,
                                         const int fdim)
{
    int i, j, k, pj, pk, quad;
    int index = _tid;
    int dIdx;

    while(index < dimSize)
    {
        i = (index + num) % (fdim / 2 + 1);
        j = ((index + num) / (fdim / 2 + 1)) % fdim;
        k = ((index + num) / (fdim / 2 + 1)) / fdim;
        
        if(j >= fdim / 2) 
        {
            j = j - fdim;
            pj = j + pdim;
        }
        else
            pj = j;
        
        if(k >= fdim / 2) 
        {
            k = k - fdim;
            pk = k + pdim;
        }
        else
            pk = k;
        
        dIdx = i + pj * (pdim / 2 + 1)
                 + pk * (pdim / 2 + 1) * pdim; 
        
        quad = i * i + j * j + k * k;

        if (quad < r)
        {
            devDst[dIdx] = devDataF[index] 
                         * devDataW[index];
        }

        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::lowpassF(Complex *devDataF, 
                                      const RFLOAT thres, 
                                      const RFLOAT ew,
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
        
#ifdef SINGLE_PRECISION
        float f = norm3df((float)i / dim, 
                          (float)j / dim, 
                          (float)k / dim);
        
        if (f > thres + ew)
        {
            devDataF[index + num] = 0;
        }
        else
        {
            if (f >= thres)
            {
                devDataF[index + num] *= (cosf((f - thres) * PI / ew) / 2 + 0.5);
            }
        }
#else
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
#endif    
        index += blockDim.x * gridDim.x;
    }

}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::correctF(RFLOAT *devDst,
                                      RFLOAT *devMkb,
                                      RFLOAT nf,
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
#ifdef RECONSTRUCTOR_REMOVE_NEG
        if (devDst[index + shift] < 0)
            devDst[index + shift] = 0; 
#endif
        index += blockDim.x * gridDim.x;
    }
}

/**
 * @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::correctF(RFLOAT *devDst,
                                      RFLOAT *devTik,
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

#ifdef RECONSTRUCTOR_REMOVE_NEG
        if (devDst[index + shift] < 0)
            devDst[index + shift] = 0; 
#endif

        index += blockDim.x * gridDim.x;
    }
}


/**
  @brief 
 * @param ...
 * @param ...
 */
D_CALLABLE void Constructor::background(RFLOAT *devDst,
                                        RFLOAT *sumWeight,
                                        RFLOAT *sumS,
                                        RFLOAT *devSumG,
                                        RFLOAT *devSumWG,
                                        RFLOAT r,
                                        RFLOAT ew,
                                        int dim,
                                        int dimSize,
                                        int indexDiff,
                                        int blockId)
{
    RFLOAT weightSum = 0;
    RFLOAT sumt = 0;
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
        
#ifdef SINGLE_PRECISION
        float u = norm3df((float)i / dim, 
                          (float)j / dim, 
                          (float)k / dim);
#else
        double u = norm3d((double)i / dim, 
                          (double)j / dim, 
                          (double)k / dim);
#endif    
        if(u > r + ew)
        {
            weightSum += 1;
            sumt += devDst[index];
        }
        else if (u >= r)
        {
#ifdef SINGLE_PRECISION
            float w = 0.5 - 0.5 * cosf((u - r) / ew * PI); 
#else
            double w = 0.5 - 0.5 * cos((u - r) / ew * PI); 
#endif    
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
D_CALLABLE void Constructor::calculateBg(RFLOAT *devSumG, 
                                         RFLOAT *devSumWG,
                                         RFLOAT *bg,
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
D_CALLABLE void Constructor::softMaskD(RFLOAT *devDst,
                                       RFLOAT *bg,
                                       RFLOAT r,
                                       RFLOAT ew,
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
        
#ifdef SINGLE_PRECISION
        float u = norm3df((float)i / dim, 
                          (float)j / dim, 
                          (float)k / dim);
#else
        double u = norm3d((double)i / dim, 
                          (double)j / dim, 
                          (double)k / dim);
#endif    
        
        if (u > r + ew)
            devDst[index + shift] = bg[0];
        else if (u >= r)
        {
#ifdef SINGLE_PRECISION
            float w = 0.5 - 0.5 * cosf((u - r) / ew * PI); 
#else
            double w = 0.5 - 0.5 * cos((u - r) / ew * PI); 
#endif    
            devDst[index + shift] = bg[0] * w + devDst[index + shift] * (1 - w);
        }
        index += blockDim.x * gridDim.x;
    }
}


//////////////////////////////////////////////////////////////////////////

} // end namespace cuthunder

//////////////////////////////////////////////////////////////////////////
