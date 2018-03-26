/**************************************************************
 * FileName: Kernel.cu
 * Author  : Kunpeng WANG,Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#include "Kernel.cuh"

namespace cuthunder {

///////////////////////////////////////////////////////////////
//                     GLOBAL VARIABLES
//

/* Constant memory on device for rotation and symmetry matrix */
//__constant__ double dev_mat_data[3][DEV_CONST_MAT_SIZE * 9];
__constant__ RFLOAT dev_ws_data[3][DEV_CONST_MAT_SIZE];


///////////////////////////////////////////////////////////////
//                     KERNEL ROUTINES
//

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE __forceinline__ int getIndexHalf(const int i,
                                            const int j,
                                            const int k,
                                            const int dim)
{
   return (k >= 0 ? k : k + dim) * (dim / 2 + 1) * dim
        + (j >= 0 ? j : j + dim) * (dim / 2 + 1)
        + i; 
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void addFTD(RFLOAT* devDataT,
                       RFLOAT value,
                       RFLOAT iCol,
                       RFLOAT iRow,
                       RFLOAT iSlc,
                       const int dim)
{
    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
    }

    RFLOAT w[2][2][2];
    int x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};
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
D_CALLABLE void addFTC(Complex* devDataF,
                       Complex& value,
                       RFLOAT iCol,
                       RFLOAT iRow,
                       RFLOAT iSlc,
                       const int dim)
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        iSlc *= -1;
        conjug = true;
    }

    RFLOAT w[2][2][2];
    int x0[3];
    RFLOAT x[3] = {iCol, iRow, iSlc};
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
D_CALLABLE Complex getTextureC(RFLOAT iCol,
                               RFLOAT iRow,
                               RFLOAT iSlc,
                               const int dim,
                               cudaTextureObject_t texObject)
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
D_CALLABLE Complex getTextureC2D(RFLOAT iCol,
                                 RFLOAT iRow,
                                 const int dim,
                                 cudaTextureObject_t texObject)
{
    if (iRow < 0) iRow += dim;

#ifdef SINGLE_PRECISION
    float2 cval = tex2D<float2>(texObject, iCol, iRow);

    return Complex(cval.x, cval.y);
#else
    int4 cval = tex2D<int4>(texObject, iCol, iRow);

    return Complex(__hiloint2double(cval.y,cval.x),
                   __hiloint2double(cval.w,cval.z));
#endif    
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE Complex getByInterp2D(RFLOAT iCol,
                                 RFLOAT iRow,
                                 const int interp,
                                 const int dim,
                                 cudaTextureObject_t texObject)
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        conjug = true;
    }

    if(interp == 0)
    {
        Complex result = getTextureC2D(iCol, 
                                       iRow,
                                       dim,
                                       texObject);
        return conjug ? result.conj() : result;
    }

    RFLOAT w[2][2];
    int x0[2];
    RFLOAT x[2] = {iCol, iRow};

    WG_BI_LINEAR_INTERP(w, x0, x);

    Complex result (0.0, 0.0);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++){
            
            result += getTextureC2D((RFLOAT)x0[0] + i, 
                                    (RFLOAT)x0[1] + j, 
                                    dim, 
                                    texObject)
                   * w[j][i];
        } 
    return conjug ? result.conj() : result;
}

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
                                         cudaTextureObject_t texObject)
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
                                    int npxl)
{
    Constructor constructor;
        
    int tid = threadIdx.x;

    constructor.init(tid);

    constructor.expectPrectf(dev_ctfa,
                             dev_def,
                             dev_k1,
                             dev_k2,
                             deviCol,
                             deviRow, 
                             blockIdx.x,
                             blockDim.x,
                             npxl);
}

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
                                 int npxl)
{
    int i, j;
    int tranoff = blockIdx.x * 2;   
    int poff = blockIdx.x * npxl;   
    
    Complex imgTemp(0.0, 0.0);
    RFLOAT phase, col, row;

    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = deviCol[itr];
        j = deviRow[itr];
        
        col = (RFLOAT)(2 * PI * dev_trans[tranoff] / idim);
        row = (RFLOAT)(2 * PI * dev_trans[tranoff + 1] / idim);
        phase = -1 * (i * col + j * row); 
        
#ifdef SINGLE_PRECISION
        devtraP[poff + itr].set(cosf(phase), sinf(phase));
#else
        devtraP[poff + itr].set(cos(phase), sin(phase));
#endif    
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRotMat(double* devRotm,
                                 double* devnR,
                                 int nR)
{
    extern __shared__ double matS[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= nR)
        return;
    
    double *mat, *res;
    mat = matS + threadIdx.x * 18;
    res = mat  + 9;
    
    mat[0] = 0; mat[4] = 0; mat[8] = 0;
    mat[5] = devnR[tid * 4 + 1];
    mat[6] = devnR[tid * 4 + 2];
    mat[1] = devnR[tid * 4 + 3];
    mat[7] = -mat[5];
    mat[2] = -mat[6];
    mat[3] = -mat[1];
    
    for(int i = 0; i < 9; i++)
        res[i] = 0;   
    
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                res[i + j * 3] += mat[i + k * 3] * mat[k + j * 3];
    
    double scale = 2 * devnR[tid * 4];
    for (int n = 0; n < 9; n++)
    {
        mat[n] *= scale;
        mat[n] += res[n] * 2;
    }
    
    mat[0] += 1;
    mat[4] += 1;
    mat[8] += 1;
    
    for (int n = 0; n < 9; n++)
    {
        devRotm[tid * 9 + n] = mat[n]; 
    }
}

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
                                 cudaTextureObject_t texObject)
{
    //extern __shared__ double rotMat[];
    
    Mat33 mat;
    mat.init(&devRotm[(blockIdx.x + shift) * 9], 0);

    //for (int itr = threadIdx.x; itr < 9; itr += blockDim.x)
    //    rotMat[itr] = devRotm[(blockIdx.x + shift) * 9 + itr];

    //__syncthreads();
    //mat.init(rotMat, 0);
    
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        Vec3 newCor((double)(deviCol[itr] * pf), 
                    (double)(deviRow[itr] * pf), 
                    0);

        Vec3 oldCor = mat * newCor;

        priRotP[blockIdx.x * npxl + itr] = getByInterpolationFTC((RFLOAT)oldCor(0),
                                                                 (RFLOAT)oldCor(1),
                                                                 (RFLOAT)oldCor(2),
                                                                 interp,
                                                                 vdim,
                                                                 texObject);
    }
}

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
                                 cudaTextureObject_t texObject)
{
    double i, j;
    double oldi, oldj;
    extern __shared__ double rotMat[];
    for (int itr = threadIdx.x; itr < 2; itr += blockDim.x)
        rotMat[itr] = devnR[(blockIdx.x + shift) * 2 + itr];

    __syncthreads();
    
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = (double)(deviCol[itr] * pf); 
        j = (double)(deviRow[itr] * pf); 
        
        oldi = i * rotMat[0] - j * rotMat[1];
        oldj = i * rotMat[1] + j * rotMat[0];

        priRotP[blockIdx.x * npxl + itr] = getByInterp2D((RFLOAT)oldi,
                                                         (RFLOAT)oldj,
                                                         interp,
                                                         vdim,
                                                         texObject);
    }
}

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
                                 int nT,
                                 int rbatch,
                                 int npxl)
{
    extern __shared__ RFLOAT result[];
    
    result[threadIdx.x] = 0;
   
    /* One block handle one par:
     *    i: Range: ibatch 
     *    j: Range: rbatch * nT
     *    blockId = i * rbatch * nT + j
     */
    int nrIdx = (blockIdx.x % (rbatch * nT)) / nT;
    int ntIdx = (blockIdx.x % (rbatch * nT)) % nT; 
    int imgIdx = blockIdx.x / (rbatch * nT);

    Complex temp(0.0, 0.0), tempC(0.0, 0.0); 
    RFLOAT tempD = 0;
    
    nrIdx *= npxl;
    ntIdx *= npxl;
    imgIdx *= npxl;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        temp  = devtraP[ntIdx + itr] * priRotP[nrIdx + itr];
        tempC = devdatP[imgIdx + itr] - temp * devctfP[imgIdx + itr];
        tempD = tempC.real() * tempC.real() + tempC.imag() * tempC.imag();
        result[threadIdx.x] += tempD * devsigP[imgIdx + itr];  
    }
    
    __syncthreads();
   
    int i = 256;
    while (i != 0) 
    {
        if (threadIdx.x < i)
        {
            result[threadIdx.x] += result[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        devDvp[blockIdx.x] = result[0];
    }  
    
}

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
                                 int rSize)
{
   int imgBase = threadIdx.x * rSize;
   if (rIdx == 0)
   {
       devbaseL[threadIdx.x] = devDvp[imgBase];
   }

   RFLOAT offset, nf, w;
   int cBase = threadIdx.x * nK;
   int rBase = threadIdx.x * nR + rIdx;
   int tBase = threadIdx.x * nT;
   for (int itr = 0; itr < rSize; itr++)
   {
       if (devDvp[threadIdx.x * rSize + itr] > devbaseL[threadIdx.x])
       {
           offset = devDvp[threadIdx.x * rSize + itr] - devbaseL[threadIdx.x];
#ifdef SINGLE_PRECISION
           nf = expf(-offset);
#else
           nf = exp(-offset);
#endif    
           for (int c = 0; c < nK; c++)
               devwC[threadIdx.x * nK + c] *= nf;
           for (int r = 0; r < nR; r++)
               devwR[threadIdx.x * nR + r] *= nf;
           for (int t = 0; t < nT; t++)
               devwT[threadIdx.x * nT + t] *= nf;
           
           devbaseL[threadIdx.x] += offset;
       }
       
#ifdef SINGLE_PRECISION
       w = expf(devDvp[imgBase + itr] - devbaseL[threadIdx.x]);
#else
       w = exp(devDvp[imgBase + itr] - devbaseL[threadIdx.x]);
#endif    
       devwC[cBase] += w;
       devwR[rBase + itr / nT] += w;
       devwT[tBase + itr % nT] += w;
   }
}

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
                                 int rSize)
{
   int imgBase = threadIdx.x * rSize;
   if (rIdx == 0)
   {
       devbaseL[threadIdx.x] = devDvp[imgBase];
   }

   RFLOAT offset, nf, w;
   int cBase = threadIdx.x * nK + kIdx;
   int rBase = threadIdx.x * nR + rIdx;
   int tBase = threadIdx.x * nT;
   for (int itr = 0; itr < rSize; itr++)
   {
       if (devDvp[threadIdx.x * rSize + itr] > devbaseL[threadIdx.x])
       {
           offset = devDvp[threadIdx.x * rSize + itr] - devbaseL[threadIdx.x];
#ifdef SINGLE_PRECISION
           nf = expf(-offset);
#else
           nf = exp(-offset);
#endif    
           for (int c = 0; c < nK; c++)
               devwC[threadIdx.x * nK + c] *= nf;
           for (int r = 0; r < nR; r++)
               devwR[threadIdx.x * nR + r] *= nf;
           for (int t = 0; t < nT; t++)
               devwT[threadIdx.x * nT + t] *= nf;
           
           devbaseL[threadIdx.x] += offset;
       }
       
#ifdef SINGLE_PRECISION
       w = expf(devDvp[imgBase + itr] - devbaseL[threadIdx.x]);
#else
       w = exp(devDvp[imgBase + itr] - devbaseL[threadIdx.x]);
#endif    
       devwC[cBase] += w;
       devwR[rBase + itr / nT] += w;
       devwT[tBase + itr % nT] += w;
   }
}

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
                                    )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float myrand;
    
    curandState s;
    curand_init(out, tid, 0, &s);
    
    //myrand = curand_uniform(&s);
    //myrand *= (0 - nC);
    //myrand += (nC - 0);
    //dev_ramC[tid] = (int)truncf(myrand);
    
    myrand = curand_uniform(&s);
    myrand *= (0 - tSize);
    myrand += (tSize - 0);
    int t = ((int)truncf(myrand) + blockIdx.x * tSize) * 2; 
    //int t = (blockIdx.x * tSize) * 2; 
    for (int n = 0; n < 2; n++)
    {
        dev_tran[tid * 2 + n] = dev_nt[t + n];
    }
            
    myrand = curand_uniform(&s);
    myrand *= (0 - rSize);
    myrand += (rSize - 0);
    int r = ((int)truncf(myrand) + blockIdx.x * rSize) * 4;
    //int r = (blockIdx.x + blockIdx.x * rSize) * 4;
    for (int n = 0; n < 4; n++)
    {
        dev_ramR[tid * 4 + n] = dev_nr[r + n];
    }
    
    myrand = curand_uniform(&s);
    myrand *= (0 - dSize);
    myrand += (dSize - 0);
    dev_ramD[tid] = dev_nd[blockIdx.x * dSize + (int)truncf(myrand)];
    //dev_ramD[tid] = dev_nd[blockIdx.x * dSize];
}

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
                                    )
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float myrand;
    
    curandState s;
    curand_init(out, tid, 0, &s);
    
    //myrand = curand_uniform(&s);
    //myrand *= (0 - nC);
    //myrand += (nC - 0);
    //dev_ramC[tid] = (int)truncf(myrand);
    
    myrand = curand_uniform(&s);
    myrand *= (0 - tSize);
    myrand += (tSize - 0);
    int t = ((int)truncf(myrand) + blockIdx.x * tSize) * 2; 
    //int t = (blockIdx.x * tSize) * 2; 
    for (int n = 0; n < 2; n++)
    {
        dev_tran[tid * 2 + n] = dev_nt[t + n];
    }
            
    myrand = curand_uniform(&s);
    myrand *= (0 - rSize);
    myrand += (rSize - 0);
    int r = ((int)truncf(myrand) + blockIdx.x * rSize) * 4;
    //int r = (blockIdx.x + blockIdx.x * rSize) * 4;
    for (int n = 0; n < 4; n++)
    {
        dev_ramR[tid * 4 + n] = dev_nr[r + n];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRandomR(double* dev_mat,
                                  double* dev_ramR)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ double matS[];
    
    double *mat, *res;
    mat = matS + threadIdx.x * 18;
    res = mat  + 9;
    
    mat[0] = 0; mat[4] = 0; mat[8] = 0;
    mat[5] = dev_ramR[tid * 4 + 1];
    mat[6] = dev_ramR[tid * 4 + 2];
    mat[1] = dev_ramR[tid * 4 + 3];
    mat[7] = -mat[5];
    mat[2] = -mat[6];
    mat[3] = -mat[1];
    
    for(int i = 0; i < 9; i++)
        res[i] = 0;   
    
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                res[i + j * 3] += mat[i + k * 3] * mat[k + j * 3];
    
    double scale = 2 * dev_ramR[tid * 4];
    for (int n = 0; n < 9; n++)
    {
        mat[n] *= scale;
        mat[n] += res[n] * 2;
    }
    
    mat[0] += 1;
    mat[4] += 1;
    mat[8] += 1;
    
    for (int n = 0; n < 9; n++)
    {
        dev_mat[tid * 9 + n] = mat[n]; 
    }
}

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
                                 int idim)
{
    int i, j;
    int off = (blockIdx.x * mReco + insertIdx) * 2;   
    
    Complex imgTemp(0.0, 0.0);
    RFLOAT phase, col, row;

    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = deviCol[itr] / opf;
        j = deviRow[itr] / opf;
        
        col = -(RFLOAT)(dev_tran[off] - dev_offS[blockIdx.x * 2]) / idim;
        row = -(RFLOAT)(dev_tran[off + 1] - dev_offS[blockIdx.x * 2 + 1]) / idim;
        phase = PI_2 * (i * col + j * row); 
#ifdef SINGLE_PRECISION
        imgTemp.set(cosf(-phase), sinf(-phase));
#else
        imgTemp.set(cos(-phase), sin(-phase));
#endif    
        devtranP[blockIdx.x * npxl + itr] = devdatP[blockIdx.x * npxl + itr] * imgTemp;
    }
}

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
                                    int mReco)
{
    int i, j;
    int quat = blockIdx.x * mReco + insertIdx;   
    
    RFLOAT lambda, defocus, angle, k1, k2, ki, u, w1, w2;
    
#ifdef SINGLE_PRECISION
    lambda = 12.2643247 / sqrtf(dev_ctfas[blockIdx.x].voltage 
                        * (1 + dev_ctfas[blockIdx.x].voltage * 0.978466e-6));
    
    w1 = sqrtf(1 - dev_ctfas[blockIdx.x].amplitudeContrast * dev_ctfas[blockIdx.x].amplitudeContrast);
#else
    lambda = 12.2643247 / sqrt(dev_ctfas[blockIdx.x].voltage 
                        * (1 + dev_ctfas[blockIdx.x].voltage * 0.978466e-6));
    
    w1 = sqrt(1 - dev_ctfas[blockIdx.x].amplitudeContrast * dev_ctfas[blockIdx.x].amplitudeContrast);
#endif    
    w2 = dev_ctfas[blockIdx.x].amplitudeContrast;
    
    k1 = PI * lambda;
    k2 = divPI2 * dev_ctfas[blockIdx.x].Cs * lambda * lambda * lambda;
    
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = deviCol[itr] / opf;
        j = deviRow[itr] / opf;
#ifdef SINGLE_PRECISION
        u = sqrtf((i / pixel) * (i / pixel) + (j / pixel) * (j / pixel));
        angle = atan2f((float)j, (float)i) - dev_ctfas[blockIdx.x].defocusTheta;
        
        defocus = -(dev_ctfas[blockIdx.x].defocusU 
                    + dev_ctfas[blockIdx.x].defocusV 
                    + (dev_ctfas[blockIdx.x].defocusU - dev_ctfas[blockIdx.x].defocusV) 
                    * cosf(2 * angle)) * (float)dev_ramD[quat] / 2;
        
        ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[blockIdx.x].phaseShift;
        devctfP[blockIdx.x * npxl + itr] = -w1 * sinf(ki) + w2 * cosf(ki);
#else
        u = sqrt((i / pixel) * (i / pixel) + (j / pixel) * (j / pixel));
        angle = atan2((double)j, (double)i) - dev_ctfas[blockIdx.x].defocusTheta;
        
        defocus = -(dev_ctfas[blockIdx.x].defocusU 
                    + dev_ctfas[blockIdx.x].defocusV 
                    + (dev_ctfas[blockIdx.x].defocusU - dev_ctfas[blockIdx.x].defocusV) 
                    * cos(2 * angle)) * dev_ramD[quat] / 2;
        
        ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[blockIdx.x].phaseShift;
        devctfP[blockIdx.x * npxl + itr] = -w1 * sin(ki) + w2 * cos(ki);
#endif    
    }
}

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
                               int smidx)
{
    extern __shared__ double rotMat[];

    RFLOAT ctfTemp; 
    Mat33 mat;

    for (int itr = threadIdx.x; itr < 9; itr += blockDim.x)
        rotMat[itr] = dev_mat[(blockIdx.x * mReco + insertIdx) * 9 + itr];

    __syncthreads();
    
    mat.init(rotMat, 0);
    
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        Vec3 newCor((double)deviCol[itr], (double)deviRow[itr], 0);
        Vec3 oldCor = mat * newCor;

        ctfTemp = devctfP[blockIdx.x * npxl + itr] 
                * devctfP[blockIdx.x * npxl + itr]
                * devsigRcpP[blockIdx.x * npxl + itr]
                * dev_ws_data[smidx][blockIdx.x];
        
        addFTD(devDataT,
               ctfTemp,
               (RFLOAT)oldCor(0), 
               (RFLOAT)oldCor(1), 
               (RFLOAT)oldCor(2),
               vdim); 
    }
}

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
                               int smidx)
{
    extern __shared__ double rotMat[];
    
    Mat33 mat;
    
    for (int itr = threadIdx.x; itr < 9; itr += blockDim.x)
        rotMat[itr] = dev_mat[(blockIdx.x * mReco + insertIdx) * 9 + itr];

    __syncthreads();
    
    mat.init(rotMat, 0);
    
    Complex tran(0.0, 0.0); 
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        Vec3 newCor((double)deviCol[itr], (double)deviRow[itr], 0);
        Vec3 oldCor = mat * newCor;
        
        tran = devtranP[blockIdx.x * npxl + itr];
        tran *= (devctfP[blockIdx.x * npxl + itr] 
                * devsigRcpP[blockIdx.x * npxl + itr]
                * dev_ws_data[smidx][blockIdx.x]);
        
        addFTC(devDataF,
               tran,
               (RFLOAT)oldCor(0), 
               (RFLOAT)oldCor(1), 
               (RFLOAT)oldCor(2), 
               vdim);
    }
}

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
                               int smidx)
{
    extern __shared__ double rotMat[];

    RFLOAT ctfTemp; 
    Mat33 mat;

    for (int itr = threadIdx.x; itr < 9; itr += blockDim.x)
        rotMat[itr] = dev_mat[(blockIdx.x * mReco + insertIdx) * 9 + itr];

    __syncthreads();
    
    mat.init(rotMat, 0);
    
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        Vec3 newCor((double)deviCol[itr], (double)deviRow[itr], 0);
        Vec3 oldCor = mat * newCor;

        ctfTemp = devctfP[blockIdx.x * npxl + itr] 
                * devctfP[blockIdx.x * npxl + itr]
                * dev_ws_data[smidx][blockIdx.x];
       
        addFTD(devDataT,
               ctfTemp,
               (RFLOAT)oldCor(0), 
               (RFLOAT)oldCor(1), 
               (RFLOAT)oldCor(2),
               vdim);
    }
}

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
                               int smidx)
{
    extern __shared__ double rotMat[];
    
    Mat33 mat;
    
    for (int itr = threadIdx.x; itr < 9; itr += blockDim.x)
        rotMat[itr] = dev_mat[(blockIdx.x * mReco + insertIdx) * 9 + itr];

    __syncthreads();
    
    mat.init(rotMat, 0);
   
    Complex tran(0.0, 0.0); 
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        Vec3 newCor((double)deviCol[itr], (double)deviRow[itr], 0);
        Vec3 oldCor = mat * newCor;
        
        tran = devtranP[blockIdx.x * npxl + itr];
        tran *= (devctfP[blockIdx.x * npxl + itr] 
                * dev_ws_data[smidx][blockIdx.x]);
        
        addFTC(devDataF,
               tran,
               (RFLOAT)oldCor(0), 
               (RFLOAT)oldCor(1), 
               (RFLOAT)oldCor(2), 
               vdim);
    }
}

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeTF(Complex *devDataF,
                                   RFLOAT *devDataT, 
                                   const int length,
                                   const int num, 
                                   const RFLOAT sf)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.normalizeTF(devDataF,
                            devDataT, 
                            length, 
                            num, 
                            sf);
}

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_NormalizeT(RFLOAT *devDataT, 
                                  const int length,
                                  const int num, 
                                  const RFLOAT sf)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.normalizeT(devDataT, length, num, sf);
}

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
                                   const int num,
                                   const int dim,
                                   const int dimSize,
                                   cudaTextureObject_t texObject)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.symmetrizeT(devDataT, 
                            devSymmat, 
                            r,
                            numSymMat,
                            interp,
                            num,
                            dim,
                            dimSize,
                            texObject);
}

/**
 * @brief Normalize F: F = F * sf
 *
 * @param devDataF : the pointer of F3D
 * @param length : F3D's size
 * @param sf : the coefficient to Normalize F
 **/
__global__ void kernel_NormalizeF(Complex *devDataF, 
                                  const int length, 
                                  const int num, 
                                  const RFLOAT sf)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.normalizeF(devDataF, length, num, sf);
}

/**
 * @brief Symmetrize F3D
 * 
 * @param devDataF : the pointer of T3D
 * @param devSym : the pointer of Volume
 * @param devSymmat : the Symmetry Matrix
 * @param numSymMat : the size of the Symmetry Matrix
 * @param r : the range of T3D elements need to be symmetrized
 * @param interp : the way of interpolating
 * @param dim : the length of one side of F3D
 **/
__global__ void kernel_SymmetrizeF(Complex *devDataF,
                                   double *devSymmat, 
                                   const int numSymMat, 
                                   const int r, 
                                   const int interp,
                                   const int num,
                                   const int dim,
                                   const int dimSize,
                                   cudaTextureObject_t texObject)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.symmetrizeF(devDataF, 
                            devSymmat, 
                            r,
                            numSymMat,
                            interp,
                            num,
                            dim,
                            dimSize,
                            texObject);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ShellAverage(RFLOAT *devAvg, 
                                    int *devCount, 
                                    RFLOAT *devDataT,
                                    int dim, 
                                    int r,
                                    int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    extern __shared__ RFLOAT sum[];

    RFLOAT *sumAvg = sum;
    int *sumCount = (int*)&sumAvg[dim];

    Constructor constructor;
    constructor.init(tid);

    constructor.shellAverage(devAvg, 
                             devCount,
                             devDataT,
                             sumAvg,
                             sumCount, 
                             r,
                             dim,
                             dimSize,
                             threadIdx.x,
                             blockIdx.x);
}

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
                                    int r)
{
    int tid = threadIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.calculateAvg(devAvg2D, 
                             devCount2D,
                             devAvg,
                             devCount,
                             dim, 
                             r);

}

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
                                    int num, 
                                    int dim,
                                    int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.calculateFSC(devFSC, 
                             devAvg,
                             devDataT,
                             num, 
                             dim,
                             dimSize,
                             fscMatsize,
                             wiener,
                             r, 
                             joinHalf);

}

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
                                   int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    Constructor constructor;
    constructor.init(tid);
    
    constructor.wienerConst(devDataT,
                            wiener, 
                            r, 
                            num,
                            dim, 
                            dimSize);

}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateW(RFLOAT *devDataW,  
                                  RFLOAT *devDataT,  
                                  const int length,
                                  const int num,
                                  const int dim,
                                  const int r)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);
    
    constructor.calculateW(devDataW,
                           devDataT,
                           length, 
                           num,
                           dim,
                           r); 
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_InitialW(RFLOAT *devDataW,  
                                int initWR, 
                                int dim,
                                int dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);
    
    constructor.initialW(devDataW,
                         initWR,
                         dim,
                         dimSize);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_DeterminingC(Complex *devDataC,
                                    RFLOAT *devDataT, 
                                    RFLOAT *devDataW,
                                    const int length)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    Constructor constructor;
    constructor.init(tid);

    constructor.determiningC(devDataC,
                             devDataT,
                             devDataW,
                             length);

}

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
                                  int dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.convoluteC(devDoubleC,
                           tabfunc, 
                           nf,
                           padSize,
                           dim,
                           dimSize);
}

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
                                    int dimSize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);
    
    constructor.recalculateW(devDataC,
                             devDataW,
                             initWR,
                             dim,
                             dimSize);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCAVG(RFLOAT *diff,
                                 RFLOAT *counter,
                                 Complex *devDataC,  
                                 int r, 
                                 int dim,
                                 int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ RFLOAT sum[];

    RFLOAT *sumDiff = sum;
    RFLOAT *sumCount = (RFLOAT*)&sumDiff[dim];

    Constructor constructor;
    constructor.init(tid);

    constructor.checkCAVG(sumDiff,
                          sumCount,
                          diff, 
                          counter,
                          devDataC,
                          r,
                          dim,
                          dimSize, 
                          threadIdx.x,
                          blockIdx.x);
}

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
                                 int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ RFLOAT singleMax[];
    
    Constructor constructor;
    constructor.init(tid);

    constructor.checkCMAX(singleMax,
                          devMax, 
                          devDataC,
                          r,
                          dim,
                          dimSize, 
                          threadIdx.x,
                          blockIdx.x);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_NormalizeFW(Complex *devDst,
                                   Complex *devDataF, 
                                   RFLOAT *devDataW,
                                   const int length, 
                                   const int shift,
                                   const int r,
                                   const int pdim,
                                   const int fdim)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.normalizeFW(devDst,
                            devDataF, 
                            devDataW, 
                            length, 
                            shift,
                            r,
                            pdim,
                            fdim);
}

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
                                const int dimSize) 
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);
    
    constructor.lowpassF(devDataF, 
                         thres, 
                         ew, 
                         num, 
                         dim, 
                         dimSize);
}

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
                                const int dimSize, 
                                const int shift)
{
    //int tid = threadIdx.x;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.correctF(devDst, 
                         devMkb,
                         nf, 
                         dim, 
                         dimSize, 
                         shift);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CorrectF(RFLOAT *devDst, 
                                RFLOAT *devTik,
                                const int dim, 
                                const int dimSize,
                                const int shift)
{
    //int tid = threadIdx.x;
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.correctF(devDst, 
                         devTik, 
                         dim, 
                         dimSize,
                         shift);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_Background(RFLOAT *devDst,
                                  RFLOAT *devSumG,
                                  RFLOAT *devSumWG,
                                  const int dim,
                                  RFLOAT r,
                                  RFLOAT edgeWidth,
                                  const int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ RFLOAT sum[];

    RFLOAT *sumWeight = sum;
    RFLOAT *sumS = (RFLOAT*)&sumWeight[dim];

    Constructor constructor;
    constructor.init(tid);

    constructor.background(devDst, 
                           sumWeight,
                           sumS,
                           devSumG,
                           devSumWG, 
                           r, 
                           edgeWidth,
                           dim,
                           dimSize,
                           threadIdx.x,
                           blockIdx.x);
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CalculateBg(RFLOAT *devSumG,
                                   RFLOAT *devSumWG,
                                   RFLOAT *bg,
                                   int dim)
{
    int tid = threadIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.calculateBg(devSumG, 
                            devSumWG,
                            bg,
                            dim);

}
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
                                 const int shift)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    Constructor constructor;
    constructor.init(tid);

    constructor.softMaskD(devDst, 
                          bg, 
                          r, 
                          edgeWidth, 
                          dim,
                          dimSize,
                          shift);
}

///////////////////////////////////////////////////////////////

} // end namespace cuthunder

///////////////////////////////////////////////////////////////
