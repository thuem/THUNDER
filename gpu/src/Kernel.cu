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
D_CALLABLE __forceinline__ int getIndexHalf2D(const int i,
                                              const int j,
                                              const int dim)
{
   return (j >= 0 ? j : j + dim) * (dim / 2 + 1)
        + i; 
}

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
D_CALLABLE void addFTD2D(RFLOAT* devDataT,
                         RFLOAT value,
                         RFLOAT iCol,
                         RFLOAT iRow,
                         const int dim)
{
    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
    }

    RFLOAT w[2][2];
    int x0[2];
    RFLOAT x[2] = {iCol, iRow};
    int index;

    WG_BI_LINEAR_INTERPF(w, x0, x);
    
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++)
        {    
            index = getIndexHalf2D(x0[0] + i,
                                   x0[1] + j,
                                   dim);
            //if (index < 0 || index >= dim * dim * (dim / 2 + 1))
            //    printf("index error!\n");
            atomicAdd(&devDataT[index], value * w[j][i]);
        } 
}

/**
 * @brief ...
 * @param ...
 * @param ...
 */
D_CALLABLE void addFTC2D(Complex* devDataF,
                         Complex& value,
                         RFLOAT iCol,
                         RFLOAT iRow,
                         const int dim)
{
    bool conjug = false;

    if (iCol < 0)
    {
        iCol *= -1;
        iRow *= -1;
        conjug = true;
    }

    RFLOAT w[2][2];
    int x0[2];
    RFLOAT x[2] = {iCol, iRow};
    int index;

    WG_BI_LINEAR_INTERPF(w, x0, x);

    conjug ? value.conj() : value;

    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++)
        {
            index = getIndexHalf2D(x0[0] + i,
                                   x0[1] + j,
                                   dim);

            //if (index < 0 || index >= dim * dim * (dim / 2 + 1))
            //    printf("index error!\n");
            atomicAdd(devDataF[index].realAddr(), value.real() * w[j][i]);
            atomicAdd(devDataF[index].imagAddr(), value.imag() * w[j][i]);
        } 
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

    WG_BI_LINEAR_INTERPF(w, x0, x);

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
__global__ void kernel_TranslateL(Complex* devtraP,
                                  double* devnT,
                                  int* deviCol,
                                  int* deviRow,
                                  int idim,
                                  int npxl)
{
    int i, j;
    
    Complex imgTemp(0.0, 0.0);
    RFLOAT phase, col, row;
    col = (RFLOAT)(PI_2 * devnT[blockIdx.x * 2] / idim);
    row = (RFLOAT)(PI_2 * devnT[blockIdx.x * 2 + 1] / idim);

    int tIdx = blockIdx.x * npxl;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = deviCol[itr];
        j = deviRow[itr];
        phase = -1 * (i * col + j * row); 
#ifdef SINGLE_PRECISION
        devtraP[tIdx + itr].set(cosf(phase), sinf(phase));
#else
        devtraP[tIdx + itr].set(cos(phase), sin(phase));
#endif    
    }
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

    col = (RFLOAT)(PI_2 * dev_trans[tranoff] / idim);
    row = (RFLOAT)(PI_2 * dev_trans[tranoff + 1] / idim);
    
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = deviCol[itr];
        j = deviRow[itr];
        
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
__global__ void kernel_CalCTFL(RFLOAT* devctfP,
                               RFLOAT* devdefO,
                               RFLOAT* devfreQ,
                               double* devdP,
                               RFLOAT phaseShift,
                               RFLOAT conT,
                               RFLOAT k1,
                               RFLOAT k2,
                               int npxl)
{
    RFLOAT freq, ki;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        freq = devfreQ[itr] * devfreQ[itr];

        ki = k1 
           * devdefO[itr] 
           * devdP[blockIdx.x]
           * freq 
           + k2
           * freq * freq
           - phaseShift;
#ifdef SINGLE_PRECISION
        devctfP[blockIdx.x * npxl + itr] = -sqrtf(1 - conT * conT)
                                         * sinf(ki)
                                         + conT
                                         * cosf(ki);
#else
        devctfP[blockIdx.x * npxl + itr] = -sqrt(1 - conT * conT)
                                         * sin(ki)
                                         + conT
                                         * cos(ki);
#endif
    }    
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getRotMatL(double* devRotm,
                                  double* devnR,
                                  int nR)
{
    extern __shared__ double matS[];

    double *mat, *res;
    mat = matS + threadIdx.x * 18;
    res = mat  + 9;
    
    mat[0] = 0; mat[4] = 0; mat[8] = 0;
    mat[5] = devnR[threadIdx.x * 4 + 1];
    mat[6] = devnR[threadIdx.x * 4 + 2];
    mat[1] = devnR[threadIdx.x * 4 + 3];
    mat[7] = -mat[5];
    mat[2] = -mat[6];
    mat[3] = -mat[1];
    
    for(int i = 0; i < 9; i++)
        res[i] = 0;   
    
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                res[i + j * 3] += mat[i + k * 3] * mat[k + j * 3];
    
    double scale = 2 * devnR[threadIdx.x * 4];
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
        devRotm[threadIdx.x * 9 + n] = mat[n]; 
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
__global__ void kernel_Project3DL(Complex* priRotP,
                                  double* devRotm,
                                  int* deviCol,
                                  int* deviRow,
                                  int pf,
                                  int vdim,
                                  int npxl,
                                  int interp,
                                  cudaTextureObject_t texObject)
{
    Mat33 mat;
    mat.init(&devRotm[blockIdx.x * 9], 0);

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
__global__ void kernel_Project3D(Complex* priRotP,
                                 double* devRotm,
                                 int* deviCol,
                                 int* deviRow,
                                 int pf,
                                 int vdim,
                                 int npxl,
                                 int interp,
                                 cudaTextureObject_t texObject)
{
    //extern __shared__ double rotMat[];
    
    Mat33 mat;
    mat.init(&devRotm[blockIdx.x * 9], 0);

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
__global__ void kernel_Project2DL(Complex* priRotP,
                                  double* devnR,
                                  int* deviCol,
                                  int* deviRow,
                                  int pf,
                                  int vdim,
                                  int npxl,
                                  int interp,
                                  cudaTextureObject_t texObject)
{
    double i, j;
    double oldi, oldj;
    int rIdx = blockIdx.x * 2;
    
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = (double)(deviCol[itr] * pf); 
        j = (double)(deviRow[itr] * pf); 
        
        oldi = i * devnR[rIdx] - j * devnR[rIdx + 1];
        oldj = i * devnR[rIdx + 1] + j * devnR[rIdx];

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
__global__ void kernel_logDataVSL(Complex* priRotP,
                                  Complex* devtraP,
                                  Complex* devdatP,
                                  RFLOAT* devctfP,
                                  RFLOAT* devsigP,
                                  RFLOAT* devDvp,
                                  //RFLOAT* devre,
                                  int nT,
                                  int npxl)
{
    extern __shared__ RFLOAT resL[];
    
    resL[threadIdx.x] = 0;
   
    int nrIdx = blockIdx.x / nT * npxl; 
    int ntIdx = (blockIdx.x % nT) * npxl; 

    Complex temp(0.0, 0.0), tempC(0.0, 0.0); 
    RFLOAT tempD = 0;
    
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        temp  = devtraP[ntIdx + itr] * priRotP[nrIdx + itr];
        tempC = devdatP[itr] - temp * devctfP[itr];
        tempD = tempC.real() * tempC.real() + tempC.imag() * tempC.imag();
        resL[threadIdx.x] += tempD * devsigP[itr]; 
        //if (blockIdx.x == 2)
        //{
        //    devre[itr] = tempD * devsigP[itr];
        //    if (itr == 0)
        //       printf("itr:%d, block:%d, thread:%d,devre:%lf\n", itr, blockIdx.x, itr, devre[itr]);
        //} 
    }
    
    __syncthreads();
   
    int i = 32;
    while (i != 0) 
    {
        if (threadIdx.x < i)
        {
            resL[threadIdx.x] += resL[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        devDvp[blockIdx.x] = resL[0];
        //if (blockIdx.x == 2)
        //    printf("result:%lf, devDvp:%lf\n", resL[0], devDvp[blockIdx.x]);
    }  
    
}

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
                                   int npxl)
{
    extern __shared__ RFLOAT result[];
    
    result[threadIdx.x] = 0;
   
    int nrIdx = blockIdx.x / (nT * nD); 
    int ntIdx = (blockIdx.x % (nT * nD)) / nD; 
    int ndIdx = (blockIdx.x % (nT * nD)) % nD; 

    Complex temp(0.0, 0.0), tempC(0.0, 0.0); 
    RFLOAT tempD = 0;
    
    nrIdx *= npxl;
    ntIdx *= npxl;
    ndIdx *= npxl;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        temp  = devtraP[ntIdx + itr] * priRotP[nrIdx + itr];
        tempC = devdatP[itr] - temp * devctfP[ndIdx + itr];
        tempD = tempC.real() * tempC.real() + tempC.imag() * tempC.imag();
        result[threadIdx.x] += tempD * devsigP[itr]; 
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
    int dvpIdx = imgIdx * nR * nT + (r + nrIdx) * nT + ntIdx; 

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
   
    int i = 32;
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
        devDvp[dvpIdx] = result[0];
    }  
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getMaxBaseL(RFLOAT* devBaseL,
                                   RFLOAT* devDvp,
                                   int angleNum)
{
    extern __shared__ RFLOAT result[];
    
    RFLOAT temp = devDvp[threadIdx.x]; 
    for (int itr = threadIdx.x; itr < angleNum; itr += blockDim.x)
    {
        if (temp < devDvp[itr])
            temp = devDvp[itr];
    }
    result[threadIdx.x] = temp;
    
    __syncthreads();
   
    int i = 512;
    while (i != 0) 
    {
        if (threadIdx.x < i)
        {
            if (result[threadIdx.x] < result[threadIdx.x + i])
                result[threadIdx.x] = result[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        devBaseL[0] = result[0];
    }  
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_getMaxBase(RFLOAT* devbaseL,
                                  RFLOAT* devDvp,
                                  int angleNum)
{
    extern __shared__ RFLOAT result[];
    
    int imgIdx = blockIdx.x * angleNum;

    RFLOAT tempD = devDvp[imgIdx + threadIdx.x];
    
    for (int itr = threadIdx.x; itr < angleNum; itr += blockDim.x)
    {
        if (devDvp[imgIdx + itr] > tempD)
            tempD = devDvp[imgIdx + itr];
    }
   
    result[threadIdx.x] = tempD;

    __syncthreads();
   
    int i = 256;
    while (i != 0) 
    {
        if (threadIdx.x < i)
        {
            if (result[threadIdx.x] < result[threadIdx.x + i])
                result[threadIdx.x] = result[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        devbaseL[blockIdx.x] = result[0];
    }  
}

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
                                   int nT)
{
    if (devcomP[blockIdx.x] > devbaseL[blockIdx.x])
    {
        RFLOAT offset, nf;
        offset = devcomP[blockIdx.x] - devbaseL[blockIdx.x];
#ifdef SINGLE_PRECISION
        nf = expf(-offset);
#else
        nf = exp(-offset);
#endif    
        int shiftR = blockIdx.x * nK * nR;
        int shiftT = blockIdx.x * nK * nT;
        for (int c = threadIdx.x; c < nK; c+= blockDim.x)
        {
            devwC[blockIdx.x * nK + c] *= nf;
            for (int r = threadIdx.x; r < nR; r+= blockDim.x)
                devwR[shiftR + c * nR + r] *= nf;
            for (int t = threadIdx.x; t < nT; t+= blockDim.x)
                devwT[shiftT + c * nT + t] *= nf;
        }

        if (threadIdx.x == 0)
            devbaseL[blockIdx.x] = devcomP[blockIdx.x];
    }
}

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
                               int rSize)
{
    extern __shared__ RFLOAT result[];

    result[threadIdx.x] = 0;
    
    RFLOAT w;
    int rIdx = 0;

    int shiftR = blockIdx.x * nK * nR + kIdx * nR;
    int shiftT = blockIdx.x * nK * blockDim.x + kIdx * blockDim.x;
    
    for (int itr = threadIdx.x; itr < rSize; itr += blockDim.x)
    {
#ifdef SINGLE_PRECISION
        w = expf(devDvp[blockIdx.x * rSize + itr] - devbaseL[blockIdx.x]);
#else
        w = exp(devDvp[blockIdx.x * rSize + itr] - devbaseL[blockIdx.x]);
#endif    
        result[threadIdx.x] += w * devpR[rIdx] * devpT[threadIdx.x];
        devwT[shiftT + threadIdx.x] += w * devpR[rIdx];
        atomicAdd(&devwR[shiftR + rIdx], w * devpT[threadIdx.x]);
        rIdx += 1;
    }

    __syncthreads();
   
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                result[threadIdx.x] += result[threadIdx.x + j];
            }
        
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                result[threadIdx.x] += result[threadIdx.x + j];
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        
        j /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {

        devwC[blockIdx.x * nK + kIdx] = result[0];
    }
}

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
                                int l)
{
    extern __shared__ RFLOAT result[];

    RFLOAT* resC = result;
    RFLOAT* resR = &resC[blockDim.x];
    RFLOAT* resD = &resR[blockDim.x];
    resC[threadIdx.x] = 0;
    resR[threadIdx.x] = 0;
    resD[threadIdx.x] = 0;
    
    for(int itr = threadIdx.x; itr < nT; itr += blockDim.x)
        devwT[itr] = 0;

    __syncthreads();
    
    RFLOAT w;
    int dvpIdx = threadIdx.x * nT;
    double rd = devR[threadIdx.x] * devD[0];
    double rc = devR[threadIdx.x] * oldC;
    double dc = devD[0] * oldC;
    double rcd = rc * devD[0];

    for (int itr = 0; itr < nT; itr++)
    {
#ifdef SINGLE_PRECISION
        w = expf(devDvp[dvpIdx + itr] - devBaseL[0]);
#else
        w = exp(devDvp[dvpIdx + itr] - devBaseL[0]);
#endif    
        resC[threadIdx.x] += w * rd * devT[itr];
        //if (threadIdx.x == 0)
        //    printf("itr:%d, w:%lf, dc:%lf, devT:%lf\n", itr, w, dc, devT[itr]);
        resR[threadIdx.x] += w * dc * devT[itr];
        //if (l == 4 && threadIdx.x == 0){
        //    printf("itr:%d, w:%lf, dc:%lf, devT:%lf\n", itr, w, dc, devT[itr]);
        //    if (itr == 2)
        //        printf("devDvp:%lf, baseL:%lf, dvpIdx:%d\n", devDvp[dvpIdx + itr], devBaseL[0], dvpIdx + itr);
        //    //printf("itr:%d, resR:%lf\n", itr, 10000 * w * dc * devT[itr]);
        //}
        resD[threadIdx.x] += w * rc * devT[itr];
        //if (l == 0 && threadIdx.x == 0)
        //   dpart[threadIdx.x] 
        atomicAdd(&devwT[itr], w * rcd);
    }

    devwR[threadIdx.x] = resR[threadIdx.x];
    __syncthreads();
   
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                resC[threadIdx.x] += resC[threadIdx.x + j];
                resD[threadIdx.x] += resD[threadIdx.x + j];
            }
        
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                resC[threadIdx.x] += resC[threadIdx.x + j];
                resD[threadIdx.x] += resD[threadIdx.x + j];
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        
        j /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        devwC[blockIdx.x] = resC[0];
        devwD[blockIdx.x] = resD[0];
    }
}

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
                                int nT)
{
    extern __shared__ RFLOAT result[];

    RFLOAT* resC = result;
    RFLOAT* resR = &resC[blockDim.x];
    RFLOAT* resD = &resR[blockDim.x];
    resC[threadIdx.x] = 0;
    resR[threadIdx.x] = 0;
    resD[threadIdx.x] = 0;
    
    for(int itr = threadIdx.x; itr < nT; itr += blockDim.x)
        devwT[itr] = 0;

    __syncthreads();
    
    RFLOAT w;
    int dvpIdx = threadIdx.x * nT;
    double rd = devR[threadIdx.x] * devD[0];
    double rc = devR[threadIdx.x] * oldC;
    double dc = devD[0] * oldC;
    double rcd = rc * devD[0];

    for (int itr = 0; itr < nT; itr++)
    {
#ifdef SINGLE_PRECISION
        w = expf(devDvp[dvpIdx + itr] - devBaseL[0]);
#else
        w = exp(devDvp[dvpIdx + itr] - devBaseL[0]);
#endif    
        resC[threadIdx.x] += w * rd * devT[itr];
        //if (threadIdx.x == 0)
        //    printf("itr:%d, w:%lf, dc:%lf, devT:%lf\n", itr, w, dc, devT[itr]);
        resR[threadIdx.x] += w * dc * devT[itr];
        resD[threadIdx.x] += w * rc * devT[itr];
        atomicAdd(&devwT[itr], w * rcd);
    }

    devwR[threadIdx.x] = resR[threadIdx.x];
    __syncthreads();
   
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                resC[threadIdx.x] += resC[threadIdx.x + j];
                resD[threadIdx.x] += resD[threadIdx.x + j];
            }
        
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                resC[threadIdx.x] += resC[threadIdx.x + j];
                resD[threadIdx.x] += resD[threadIdx.x + j];
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        
        j /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        devwC[blockIdx.x] = resC[0];
        devwD[blockIdx.x] = resD[0];
    }
}

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
                                 int nD)
{
    extern __shared__ RFLOAT result[];
    RFLOAT* resC = result;
    RFLOAT* resR = &resC[blockDim.x];
    resC[threadIdx.x] = 0;
    resR[threadIdx.x] = 0;
    
    __syncthreads();
    
    int itr;
    RFLOAT w;
    int dvpIdx = threadIdx.x * nT * nD;
    double rc = devR[threadIdx.x] * oldC;
    double td;
    int tIdx, dIdx;

    for (itr = 0; itr < nT * nD; itr++)
    {
#ifdef SINGLE_PRECISION
        w = expf(devDvp[dvpIdx + itr] - devBaseL[0]);
#else
        w = exp(devDvp[dvpIdx + itr] - devBaseL[0]);
#endif    
        tIdx = itr / nD;
        dIdx = itr % nD;
        td = devT[tIdx] * devD[dIdx];
        resC[threadIdx.x] += w * (td * devR[threadIdx.x]);
        resR[threadIdx.x] += w * (td * oldC);
        devtT[threadIdx.x + tIdx * blockDim.x] += w * (rc * devD[dIdx]);
        devtD[threadIdx.x + dIdx * blockDim.x] += w * (rc * devT[tIdx]);
    }

    devwR[threadIdx.x] = resR[threadIdx.x];
    __syncthreads();
   
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                resC[threadIdx.x] += resC[threadIdx.x + j];
            }
        
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                resC[threadIdx.x] += resC[threadIdx.x + j];
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        
        j /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        devwC[blockIdx.x] = resC[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_ReduceW(RFLOAT* devw,
                               RFLOAT* devt)
{
    extern __shared__ RFLOAT result[];
    
    result[threadIdx.x] = devt[blockIdx.x * blockDim.x + threadIdx.x];
    
    __syncthreads();
    
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                result[threadIdx.x] += result[threadIdx.x + j];
            }
        
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                result[threadIdx.x] += result[threadIdx.x + j];
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        
        j /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        devw[blockIdx.x] = result[0];
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
                                  double* dev_ramR,
                                  int* dev_nc)
{
    // blockIdx.x -> index of each image
    // threadIdx.x -> index of each insertation of each image
    
    if (threadIdx.x < dev_nc[blockIdx.x]) // true if this image should be inserted
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
                                 double* dev_tran,
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim)
{
    if (insertIdx < dev_nc[blockIdx.x])
    {
        int i, j;
        int off = (blockIdx.x * mReco + insertIdx) * 2;   
        
        Complex imgTemp(0.0, 0.0);
        RFLOAT phase, col, row;

        col = -(RFLOAT)(dev_tran[off]) / idim;
        row = -(RFLOAT)(dev_tran[off + 1]) / idim;
        //col = (RFLOAT)(PI_2 * (dev_tran[off] - dev_offS[blockIdx.x * 2]) / idim);
        //row = (RFLOAT)(PI_2 * (dev_tran[off + 1] - dev_offS[blockIdx.x * 2 + 1]) / idim);
        for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
        {
            i = deviCol[itr] / opf;
            j = deviRow[itr] / opf;
            
            phase = PI_2 * (i * col + j * row); 
#ifdef SINGLE_PRECISION
            imgTemp.set(cosf(-phase), sinf(-phase));
#else
            imgTemp.set(cos(-phase), sin(-phase));
#endif    
            devtranP[blockIdx.x * npxl + itr] = devdatP[blockIdx.x * npxl + itr] * imgTemp;
        }
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
                                 int* dev_nc,
                                 int* deviCol,
                                 int* deviRow,
                                 int insertIdx,
                                 int opf,
                                 int npxl,
                                 int mReco,
                                 int idim)
{
    if (insertIdx < dev_nc[blockIdx.x])
    {
        int i, j;
        int off = (blockIdx.x * mReco + insertIdx) * 2;   
        
        Complex imgTemp(0.0, 0.0);
        RFLOAT phase, col, row;

        col = -(RFLOAT)(dev_tran[off] - dev_offS[blockIdx.x * 2]) / idim;
        row = -(RFLOAT)(dev_tran[off + 1] - dev_offS[blockIdx.x * 2 + 1]) / idim;
        //col = (RFLOAT)(PI_2 * (dev_tran[off] - dev_offS[blockIdx.x * 2]) / idim);
        //row = (RFLOAT)(PI_2 * (dev_tran[off + 1] - dev_offS[blockIdx.x * 2 + 1]) / idim);
        for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
        {
            i = deviCol[itr] / opf;
            j = deviRow[itr] / opf;
            
            phase = PI_2 * (i * col + j * row); 
#ifdef SINGLE_PRECISION
            imgTemp.set(cosf(-phase), sinf(-phase));
#else
            imgTemp.set(cos(-phase), sin(-phase));
#endif    
            devtranP[blockIdx.x * npxl + itr] = devdatP[blockIdx.x * npxl + itr] * imgTemp;
        }
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

    col = -(RFLOAT)(dev_tran[off]) / idim;
    row = -(RFLOAT)(dev_tran[off + 1]) / idim;
    //col = (RFLOAT)(PI_2 * (dev_tran[off] - dev_offS[blockIdx.x * 2]) / idim);
    //row = (RFLOAT)(PI_2 * (dev_tran[off + 1] - dev_offS[blockIdx.x * 2 + 1]) / idim);
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = deviCol[itr] / opf;
        j = deviRow[itr] / opf;
        
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

    col = -(RFLOAT)(dev_tran[off] - dev_offS[blockIdx.x * 2]) / idim;
    row = -(RFLOAT)(dev_tran[off + 1] - dev_offS[blockIdx.x * 2 + 1]) / idim;
    //col = (RFLOAT)(PI_2 * (dev_tran[off] - dev_offS[blockIdx.x * 2]) / idim);
    //row = (RFLOAT)(PI_2 * (dev_tran[off + 1] - dev_offS[blockIdx.x * 2 + 1]) / idim);
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        i = deviCol[itr] / opf;
        j = deviRow[itr] / opf;
        
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
                                    int* dev_nc,
                                    int* deviCol,
                                    int* deviRow,
                                    RFLOAT pixel,
                                    int insertIdx,
                                    int opf,
                                    int npxl,
                                    int mReco)
{
    if (insertIdx < dev_nc[blockIdx.x])
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
                        //* cos(2 * angle)) / 2;
            
            ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[blockIdx.x].phaseShift;
            devctfP[blockIdx.x * npxl + itr] = -w1 * sinf(ki) + w2 * cosf(ki);
#else
            u = sqrt((i / pixel) * (i / pixel) + (j / pixel) * (j / pixel));
            angle = atan2((double)j, (double)i) - dev_ctfas[blockIdx.x].defocusTheta;
            
            defocus = -(dev_ctfas[blockIdx.x].defocusU 
                        + dev_ctfas[blockIdx.x].defocusV 
                        + (dev_ctfas[blockIdx.x].defocusU - dev_ctfas[blockIdx.x].defocusV) 
                        * cos(2 * angle)) * dev_ramD[quat] / 2;
                        //* cos(2 * angle)) / 2;
            
            ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[blockIdx.x].phaseShift;
            devctfP[blockIdx.x * npxl + itr] = -w1 * sin(ki) + w2 * cos(ki);
#endif    
        }
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
                    //* cos(2 * angle)) / 2;
        
        ki = k1 * defocus * u * u + k2 * u * u * u * u - dev_ctfas[blockIdx.x].phaseShift;
        devctfP[blockIdx.x * npxl + itr] = -w1 * sinf(ki) + w2 * cosf(ki);
#else
        u = sqrt((i / pixel) * (i / pixel) + (j / pixel) * (j / pixel));
        angle = atan2((double)j, (double)i) - dev_ctfas[blockIdx.x].defocusTheta;
        
        defocus = -(dev_ctfas[blockIdx.x].defocusU 
                    + dev_ctfas[blockIdx.x].defocusV 
                    + (dev_ctfas[blockIdx.x].defocusU - dev_ctfas[blockIdx.x].defocusV) 
                    * cos(2 * angle)) * dev_ramD[quat] / 2;
                    //* cos(2 * angle)) / 2;
        
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
                                 int smidx)
{
    RFLOAT ctfTemp; 
    double oldCor0, oldCor1;
    int ncIdx = blockIdx.x * mReco + insertIdx;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        oldCor0 = dev_nr[ncIdx * 2] * deviCol[itr] - dev_nr[ncIdx * 2 + 1] * deviRow[itr];
        oldCor1 = dev_nr[ncIdx * 2 + 1] * deviCol[itr] + dev_nr[ncIdx * 2] * deviRow[itr];

        //if (itr == 0 && blockIdx.x == 0)
        //    printf("old0:%.16lf, old1:%.16lf\n", oldCor0, 
        //                                         oldCor1);

        ctfTemp = devctfP[blockIdx.x * npxl + itr] 
                * devctfP[blockIdx.x * npxl + itr]
                * devsigRcpP[blockIdx.x * npxl + itr]
                * dev_ws_data[smidx][blockIdx.x];
        
        //if (itr == 0 && blockIdx.x == 0)
        //    printf("ctf:%.16lf, sig:%.16lf, ctfT:%.16lf\n", devctfP[blockIdx.x * npxl + itr], 
        //                                                    devsigRcpP[blockIdx.x *npxl + itr],
        //                                                    ctfTemp);

        addFTD2D(devDataT + dev_nc[ncIdx] * vdimSize,
                 ctfTemp,
                 (RFLOAT)oldCor0, 
                 (RFLOAT)oldCor1, 
                 vdim); 
    }
}

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
                                 int smidx)
{
    Complex tran(0.0, 0.0); 
    double oldCor0, oldCor1;
    int ncIdx = blockIdx.x * mReco + insertIdx;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        oldCor0 = dev_nr[ncIdx * 2] * deviCol[itr] - dev_nr[ncIdx * 2 + 1] * deviRow[itr];
        oldCor1 = dev_nr[ncIdx * 2 + 1] * deviCol[itr] + dev_nr[ncIdx * 2] * deviRow[itr];

        tran = devtranP[blockIdx.x * npxl + itr];
        tran *= (devctfP[blockIdx.x * npxl + itr] 
                * devsigRcpP[blockIdx.x * npxl + itr]
                * dev_ws_data[smidx][blockIdx.x]);
        
        addFTC2D(devDataF + dev_nc[ncIdx] * vdimSize,
                 tran,
                 (RFLOAT)oldCor0, 
                 (RFLOAT)oldCor1, 
                 vdim);
    }
    
}

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
                                 int smidx)
{
    RFLOAT ctfTemp; 
    double oldCor0, oldCor1;
    int ncIdx = blockIdx.x * mReco + insertIdx;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        oldCor0 = dev_nr[ncIdx * 2] * deviCol[itr] - dev_nr[ncIdx * 2 + 1] * deviRow[itr];
        oldCor1 = dev_nr[ncIdx * 2 + 1] * deviCol[itr] + dev_nr[ncIdx * 2] * deviRow[itr];

        ctfTemp = devctfP[blockIdx.x * npxl + itr] 
                * devctfP[blockIdx.x * npxl + itr]
                * dev_ws_data[smidx][blockIdx.x];
       
        addFTD2D(devDataT + dev_nc[ncIdx] * vdimSize,
                 ctfTemp,
                 (RFLOAT)oldCor0, 
                 (RFLOAT)oldCor1, 
                 vdim);
    }
    
}

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
                                 int smidx)
{
    Complex tran(0.0, 0.0); 
    double oldCor0, oldCor1;
    int ncIdx = blockIdx.x * mReco + insertIdx;
    for (int itr = threadIdx.x; itr < npxl; itr += blockDim.x)
    {
        oldCor0 = dev_nr[ncIdx * 2] * deviCol[itr] - dev_nr[ncIdx * 2 + 1] * deviRow[itr];
        oldCor1 = dev_nr[ncIdx * 2 + 1] * deviCol[itr] + dev_nr[ncIdx * 2] * deviRow[itr];

        tran = devtranP[blockIdx.x * npxl + itr];
        tran *= (devctfP[blockIdx.x * npxl + itr] 
                * dev_ws_data[smidx][blockIdx.x]);
        
        addFTC2D(devDataF + dev_nc[ncIdx] * vdimSize,
                 tran,
                 (RFLOAT)oldCor0, 
                 (RFLOAT)oldCor1, 
                 vdim);
    }
    
}

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
                                 int mReco)
{
    extern __shared__ double oCor[];

    double* resX = &oCor[blockDim.x];
    double* resY = &resX[blockDim.x];
    int* resC = (int*)&resY[blockDim.x];

    int ncIdx = threadIdx.x * mReco + insertIdx;
    resX[threadIdx.x] = -1 * dev_nr[ncIdx * 2] 
                           * (dev_nt[ncIdx * 2] 
                              - dev_offs[threadIdx.x * 2])
                           + dev_nr[ncIdx * 2 + 1] 
                           * (dev_nt[ncIdx * 2 + 1] 
                              - dev_offs[threadIdx.x * 2 + 1]);
    
    resY[threadIdx.x] = -1 * (dev_nr[ncIdx * 2 + 1] 
                              * (dev_nt[ncIdx * 2] 
                                 - dev_offs[threadIdx.x * 2])
                              + dev_nr[ncIdx * 2] 
                              * (dev_nt[ncIdx * 2 + 1] 
                                 - dev_offs[threadIdx.x * 2 + 1]));
    resC[threadIdx.x] = 1;

    __syncthreads();
   
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                resX[threadIdx.x] += resX[threadIdx.x + j];
                resY[threadIdx.x] += resY[threadIdx.x + j];
                resC[threadIdx.x] += resC[threadIdx.x + j];
            }
        
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                resX[threadIdx.x] += resX[threadIdx.x + j];
                resY[threadIdx.x] += resY[threadIdx.x + j];
                resC[threadIdx.x] += resC[threadIdx.x + j];
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        
        j /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        int oshift = dev_nc[ncIdx];
        atomicAdd(&devO[oshift * 2], resX[0]);
        atomicAdd(&devO[oshift * 2 + 1], resY[0]);
        atomicAdd(&devC[oshift], resC[0]);
    }
}

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
                                 int mReco)
{
    extern __shared__ double oCor[];

    double* resX = &oCor[blockDim.x];
    double* resY = &resX[blockDim.x];
    int* resC = (int*)&resY[blockDim.x];

    int ncIdx = threadIdx.x * mReco + insertIdx;
    resX[threadIdx.x] = -1 * dev_nr[ncIdx * 2] 
                           * dev_nt[ncIdx * 2] 
                           + dev_nr[ncIdx * 2 + 1] 
                           * dev_nt[ncIdx * 2 + 1]; 
    
    resY[threadIdx.x] = -1 * (dev_nr[ncIdx * 2 + 1] 
                              * dev_nt[ncIdx * 2] 
                              + dev_nr[ncIdx * 2] 
                              * dev_nt[ncIdx * 2 + 1]); 
    resC[threadIdx.x] = 1;

    __syncthreads();
   
    int j;
    bool flag;
    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                resX[threadIdx.x] += resX[threadIdx.x + j];
                resY[threadIdx.x] += resY[threadIdx.x + j];
                resC[threadIdx.x] += resC[threadIdx.x + j];
            }
        
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                resX[threadIdx.x] += resX[threadIdx.x + j];
                resY[threadIdx.x] += resY[threadIdx.x + j];
                resC[threadIdx.x] += resC[threadIdx.x + j];
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        
        j /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {

        int oshift = dev_nc[ncIdx];
        atomicAdd(&devO[oshift * 2], resX[0]);
        atomicAdd(&devO[oshift * 2 + 1], resY[0]);
        atomicAdd(&devC[oshift], resC[0]);
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
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    if (insertIdx < dev_nc[blockIdx.x])
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
}

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
                               int smidx)
{
    if (insertIdx < dev_nc[blockIdx.x])
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
                               int* dev_nc,
                               int* deviCol,
                               int* deviRow,
                               int insertIdx,
                               int npxl,
                               int mReco,
                               int vdim,
                               int smidx)
{
    if (insertIdx < dev_nc[blockIdx.x])
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
}

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
                               int smidx)
{
    if (insertIdx < dev_nc[blockIdx.x])
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
}

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
                                 int batchSize)
{
        extern __shared__ double oCor[];

        double* resX = oCor;
        double* resY = &resX[blockDim.x];
        double* resZ = &resY[blockDim.x];
        int* resC = (int*)&resZ[blockDim.x];

        int ncIdx;
        int nrIdx;
        int tran0;
        int tran1;

        resX[threadIdx.x] = 0;
        resY[threadIdx.x] = 0;
        resZ[threadIdx.x] = 0;
        resC[threadIdx.x] = 0;

        __syncthreads();
   
        for (int itr = threadIdx.x; itr < batchSize; itr += blockDim.x)
        {
            if (insertIdx < dev_nc[itr])
            {
                ncIdx = itr * mReco + insertIdx;
                nrIdx = ncIdx * 9;
                tran0 = dev_nt[ncIdx * 2] - dev_offs[itr * 2];
                tran1 = dev_nt[ncIdx * 2 + 1] - dev_offs[itr * 2 + 1];

                resX[threadIdx.x] += -1 * (dev_mat[nrIdx] * tran0 
                                        + dev_mat[nrIdx + 3] * tran1); 
                
                resY[threadIdx.x] += -1 * (dev_mat[nrIdx + 1] * tran0 
                                        + dev_mat[nrIdx + 4] * tran1); 
                
                resZ[threadIdx.x] += -1 * (dev_mat[nrIdx + 2] * tran0 
                                        + dev_mat[nrIdx + 5] * tran1); 
                
                resC[threadIdx.x] += 1;
            }

        }
        __syncthreads();
   
        int i = 64;
        while (i != 0) 
        {
            if (threadIdx.x < i)
            {
                resX[threadIdx.x] += resX[threadIdx.x + i];
                resY[threadIdx.x] += resY[threadIdx.x + i];
                resZ[threadIdx.x] += resZ[threadIdx.x + i];
                resC[threadIdx.x] += resC[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2; 
        }
        
        if (threadIdx.x == 0) 
        {
            atomicAdd(&devO[0], resX[0]);
            atomicAdd(&devO[1], resY[0]);
            atomicAdd(&devO[2], resZ[0]);
            atomicAdd(&devC[0], resC[0]);
        }  
}

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
                                 int batchSize)
{
        extern __shared__ double oCor[];

        double* resX = oCor;
        double* resY = &resX[blockDim.x];
        double* resZ = &resY[blockDim.x];
        int* resC = (int*)&resZ[blockDim.x];

        int ncIdx;
        int nrIdx;
        int tran0;
        int tran1;

        resX[threadIdx.x] = 0;
        resY[threadIdx.x] = 0;
        resZ[threadIdx.x] = 0;
        resC[threadIdx.x] = 0;

        __syncthreads();
   
        for (int itr = threadIdx.x; itr < batchSize; itr += blockDim.x)
        {
            if (insertIdx < dev_nc[itr])
            {
                ncIdx = itr * mReco + insertIdx;
                nrIdx = ncIdx * 9;
                tran0 = dev_nt[ncIdx * 2];
                tran1 = dev_nt[ncIdx * 2 + 1];

                resX[threadIdx.x] += -1 * (dev_mat[nrIdx] * tran0 
                                        + dev_mat[nrIdx + 3] * tran1); 
                
                resY[threadIdx.x] += -1 * (dev_mat[nrIdx + 1] * tran0 
                                        + dev_mat[nrIdx + 4] * tran1); 
                
                resZ[threadIdx.x] += -1 * (dev_mat[nrIdx + 2] * tran0 
                                        + dev_mat[nrIdx + 5] * tran1); 
                
                resC[threadIdx.x] += 1;
            }

        }
        __syncthreads();
   
        int i = 64;
        while (i != 0) 
        {
            if (threadIdx.x < i)
            {
                resX[threadIdx.x] += resX[threadIdx.x + i];
                resY[threadIdx.x] += resY[threadIdx.x + i];
                resZ[threadIdx.x] += resZ[threadIdx.x + i];
                resC[threadIdx.x] += resC[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2; 
        }
        
        if (threadIdx.x == 0) 
        {
            atomicAdd(&devO[0], resX[0]);
            atomicAdd(&devO[1], resY[0]);
            atomicAdd(&devO[2], resZ[0]);
            atomicAdd(&devC[0], resC[0]);
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
                                 int batchSize)
{
    extern __shared__ double oCor[];

    double* resX = oCor;
    double* resY = &resX[blockDim.x];
    double* resZ = &resY[blockDim.x];
    int* resC = (int*)&resZ[blockDim.x];

    int ncIdx;
    int nrIdx;
    int tran0;
    int tran1;

    resX[threadIdx.x] = 0;
    resY[threadIdx.x] = 0;
    resZ[threadIdx.x] = 0;
    resC[threadIdx.x] = 0;

    __syncthreads();
   
    for (int itr = threadIdx.x; itr < batchSize; itr += blockDim.x)
    {
        ncIdx = itr * mReco + insertIdx;
        nrIdx = ncIdx * 9;
        tran0 = dev_nt[ncIdx * 2] - dev_offs[itr * 2];
        tran1 = dev_nt[ncIdx * 2 + 1] - dev_offs[itr * 2 + 1];

        resX[threadIdx.x] += -1 * (dev_mat[nrIdx] * tran0 
                                + dev_mat[nrIdx + 3] * tran1); 
        
        resY[threadIdx.x] += -1 * (dev_mat[nrIdx + 1] * tran0 
                                + dev_mat[nrIdx + 4] * tran1); 
        
        resZ[threadIdx.x] += -1 * (dev_mat[nrIdx + 2] * tran0 
                                + dev_mat[nrIdx + 5] * tran1); 
        
        resC[threadIdx.x] += 1;

    }
    __syncthreads();
   
    int i = 64;
    while (i != 0) 
    {
        if (threadIdx.x < i)
        {
            resX[threadIdx.x] += resX[threadIdx.x + i];
            resY[threadIdx.x] += resY[threadIdx.x + i];
            resZ[threadIdx.x] += resZ[threadIdx.x + i];
            resC[threadIdx.x] += resC[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        atomicAdd(&devO[0], resX[0]);
        atomicAdd(&devO[1], resY[0]);
        atomicAdd(&devO[2], resZ[0]);
        atomicAdd(&devC[0], resC[0]);
    }  
}

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
                                 int batchSize)
{
    extern __shared__ double oCor[];

    double* resX = oCor;
    double* resY = &resX[blockDim.x];
    double* resZ = &resY[blockDim.x];
    int* resC = (int*)&resZ[blockDim.x];

    int ncIdx;
    int nrIdx;
    int tran0;
    int tran1;

    resX[threadIdx.x] = 0;
    resY[threadIdx.x] = 0;
    resZ[threadIdx.x] = 0;
    resC[threadIdx.x] = 0;

    __syncthreads();
   
    for (int itr = threadIdx.x; itr < batchSize; itr += blockDim.x)
    {
        ncIdx = itr * mReco + insertIdx;
        nrIdx = ncIdx * 9;
        tran0 = dev_nt[ncIdx * 2];
        tran1 = dev_nt[ncIdx * 2 + 1];

        resX[threadIdx.x] += -1 * (dev_mat[nrIdx] * tran0 
                                + dev_mat[nrIdx + 3] * tran1); 
        
        resY[threadIdx.x] += -1 * (dev_mat[nrIdx + 1] * tran0 
                                + dev_mat[nrIdx + 4] * tran1); 
        
        resZ[threadIdx.x] += -1 * (dev_mat[nrIdx + 2] * tran0 
                                + dev_mat[nrIdx + 5] * tran1); 
        
        resC[threadIdx.x] += 1;

    }
    __syncthreads();
   
    int i = 64;
    while (i != 0) 
    {
        if (threadIdx.x < i)
        {
            resX[threadIdx.x] += resX[threadIdx.x + i];
            resY[threadIdx.x] += resY[threadIdx.x + i];
            resZ[threadIdx.x] += resZ[threadIdx.x + i];
            resC[threadIdx.x] += resC[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {
        atomicAdd(&devO[0], resX[0]);
        atomicAdd(&devO[1], resY[0]);
        atomicAdd(&devO[2], resZ[0]);
        atomicAdd(&devC[0], resC[0]);
    }  
}

/**
 * @brief Normalize T: T = T * sf
 *
 * @param devDataT : the pointer of T3D
 * @param length : T3D's size
 * @param sf : the coefficient to Normalize T
 */
__global__ void kernel_SetSF2D(RFLOAT *devDataT, 
                               RFLOAT *sf,
                               int dimSize)
{
    sf[threadIdx.x] = 1.0 / devDataT[threadIdx.x * dimSize];
}

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
                                     int kIdx)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    devDataF[index] *= sf[kIdx];
    devDataT[index] *= sf[kIdx];
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
__global__ void kernel_ShellAverage2D(RFLOAT *devAvg2D, 
                                      int *devCount2D, 
                                      RFLOAT *devDataT,
                                      int dim, 
                                      int r)
{
    extern __shared__ RFLOAT sum[];

    RFLOAT *sumAvg = sum;
    int *sumCount = (int*)&sumAvg[r];

    for (int itr = threadIdx.x; itr < r; itr += blockDim.x)
    {
        sumAvg[itr] = 0;
        sumCount[itr] = 0;
    }

    __syncthreads();

    int j = blockIdx.x;
    if(j >= dim / 2) j = j - dim;
    
    int quad = threadIdx.x * threadIdx.x + j * j;
    
    if(quad < r * r)
    {
#ifdef SINGLE_PRECISION
        int u = (int)rintf(sqrtf((float)quad));
#else
        int u = (int)rint(sqrt((double)quad));
#endif    
        if (u < r)
        {
            atomicAdd(&sumAvg[u], devDataT[threadIdx.x + blockIdx.x * blockDim.x]);
            atomicAdd(&sumCount[u], 1);
        }
    }
    
    __syncthreads();

    for (int itr = threadIdx.x; itr < r; itr += blockDim.x)
    {
        devAvg2D[itr + blockIdx.x * r] = sumAvg[itr];
        devCount2D[itr + blockIdx.x * r] = sumCount[itr];
    }

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
__global__ void kernel_CalculateAvg2D(RFLOAT *devAvg2D,
                                      int *devCount2D,
                                      RFLOAT *devAvg,
                                      int *devCount,
                                      int dim,
                                      int r)
{
    devAvg[threadIdx.x] = 0;
    devCount[threadIdx.x] = 0;
    
    __syncthreads();

    for (int i = 0; i < dim; i++)
    {
        if (threadIdx.x < r)
        {
            devAvg[threadIdx.x] += devAvg2D[threadIdx.x + i * r];
            devCount[threadIdx.x] += devCount2D[threadIdx.x + i * r];
        }
    }
    
    if (threadIdx.x < r)
        devAvg[threadIdx.x] /= devCount[threadIdx.x];

    if(threadIdx.x == r - 1 ){
        devAvg[r] = devAvg[r - 1];
        devAvg[r + 1] = devAvg[r - 1];
    }
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
__global__ void kernel_CalculateFSC2D(RFLOAT *devDataT,
                                      RFLOAT *devFSC,
                                      RFLOAT *devAvg,
                                      bool joinHalf, 
                                      int fscMatsize,
                                      int wiener, 
                                      int dim,
                                      int pf,
                                      int r)
{
    int j = blockIdx.x;
    if(j >= dim / 2) j = j - dim;
    
    int quad = threadIdx.x * threadIdx.x + j * j;
    
    if(quad >= wiener && quad < r)
    {
#ifdef SINGLE_PRECISION
        int u = (int)rintf(sqrtf((float)quad));
        float FSC = (u / pf >= fscMatsize)
                  ? 0 
                  : devFSC[u / pf];
        
        FSC = fmaxf(1e-3, fminf(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
        FSC = sqrtf(2 * FSC / (1 + FSC));
#else
        if (joinHalf) 
            FSC = sqrtf(2 * FSC / (1 + FSC));
#endif

#else
        int u = (int)rint(sqrt((double)quad));
        double FSC = (u / pf >= fscMatsize)
                   ? 0 
                   : devFSC[u / pf];
        
        FSC = fmax(1e-3, fmin(1 - 1e-3, FSC));
#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
        FSC = sqrt(2 * FSC / (1 + FSC));
#else
        if (joinHalf) 
            FSC = sqrt(2 * FSC / (1 + FSC));
#endif

#endif    
            
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
        devDataT[threadIdx.x + blockIdx.x * blockDim.x] += (1 - FSC) / FSC * devAvg[u];
#else
        devDataT[threadIdx.x + blockIdx.x * blockDim.x] /= FSC;
#endif
    }
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
__global__ void kernel_CalculateW2D(RFLOAT *devDataW,  
                                    RFLOAT *devDataT,  
                                    const int dim,
                                    const int r)
{
    int j = blockIdx.x;
    if(j >= dim / 2) j = j - dim;
    
    int quad = threadIdx.x * threadIdx.x + j * j;

    if (quad < r)
    {
#ifdef SINGLE_PRECISION
        devDataW[threadIdx.x 
                 + blockDim.x 
                 * blockIdx.x] = 1.0 / fmaxf(fabsf(devDataT[threadIdx.x 
                                                            + blockDim.x 
                                                            * blockIdx.x]), 1e-6);
#else
        devDataW[threadIdx.x 
                 + blockDim.x 
                 * blockIdx.x] = 1.0 / fmax(fabs(devDataT[threadIdx.x 
                                                          + blockDim.x 
                                                          * blockIdx.x]), 1e-6);
#endif    
    }
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
__global__ void kernel_InitialW2D(RFLOAT *devDataW,  
                                  int initWR, 
                                  int dim)
{
    int j = blockIdx.x;
    if(j >= dim / 2) j = j - dim;
    
    int quad = threadIdx.x * threadIdx.x + j * j;

    if (quad < initWR)
    {
        devDataW[threadIdx.x + blockDim.x * blockIdx.x] = 1;
    }
    else
    {
        devDataW[threadIdx.x + blockDim.x * blockIdx.x] = 0;
    }
        
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
__global__ void kernel_DeterminingC2D(Complex *devDataC,
                                      RFLOAT *devDataT, 
                                      RFLOAT *devDataW)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    devDataC[index].set(devDataT[index] * devDataW[index], 0);
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
__global__ void kernel_convoluteC2D(RFLOAT *devDoubleC,
                                    TabFunction tabfunc,
                                    RFLOAT nf,
                                    int padSize,
                                    int dim,
                                    int dimSize)
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    if(i >= dim / 2) i = i - dim;
    if(j >= dim / 2) j = j - dim;
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    devDoubleC[index] = devDoubleC[index] 
                      / dimSize
                      * tabfunc((RFLOAT)(i * i + j * j) 
                                / (padSize * padSize))
                      / nf;
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
__global__ void kernel_RecalculateW2D(RFLOAT *devDataW,
                                      Complex *devDataC,  
                                      int initWR, 
                                      int dim)
{
    int j = blockIdx.x;
    if(j >= dim / 2) j = j - dim;
    
    RFLOAT mode = 0.0, u, x, y;
    int quad = threadIdx.x * threadIdx.x + j * j;
    int index = threadIdx.x + blockDim.x * blockIdx.x;

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
__global__ void kernel_CheckCAVG2D(RFLOAT *diff,
                                   int *counter,
                                   Complex *devDataC,  
                                   int r, 
                                   int dim)
{
    extern __shared__ RFLOAT sum[];

    RFLOAT *sumDiff = sum;
    int *sumCount = (int*)&sumDiff[blockDim.x];

    RFLOAT mode = 0, u, x, y;
    bool flag = true;
    
    int j = blockIdx.x;
    if(j >= dim / 2) j = j - dim;
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int quad = threadIdx.x * threadIdx.x + j * j;
    
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

        sumDiff[threadIdx.x] = fabsf(mode - 1);
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

        sumDiff[threadIdx.x] = fabs(mode - 1);
#endif    
        sumCount[threadIdx.x] = 1;
    }

    __syncthreads();

    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                sumDiff[threadIdx.x] += sumDiff[threadIdx.x + j];
                sumCount[threadIdx.x] += sumCount[threadIdx.x + j];
            }
        
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                sumDiff[threadIdx.x] += sumDiff[threadIdx.x + j];
                sumCount[threadIdx.x] += sumCount[threadIdx.x + j];
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        
        j /= 2; 
    }
    
    if (threadIdx.x == 0) 
    {

        diff[blockIdx.x] = sumDiff[0];
        counter[blockIdx.x] = sumCount[0];
    }
}

/**
 * @brief ...
 *
 * @param ...
 * @param ...
 */
__global__ void kernel_CheckCAVG(RFLOAT *diff,
                                 int *counter,
                                 Complex *devDataC,  
                                 int r, 
                                 int dim,
                                 int dimSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ RFLOAT sum[];

    RFLOAT *sumDiff = sum;
    int *sumCount = (int*)&sumDiff[dim];

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
__global__ void kernel_CheckCMAX2D(RFLOAT *devMax,
                                   Complex *devDataC,  
                                   int r, 
                                   int dim)
{
    extern __shared__ RFLOAT singleMax[];
    
    singleMax[threadIdx.x] = 0;
    
    __syncthreads();

    RFLOAT mode = 0.0, u, x, y;
    bool flag = true;
    
    int j = blockIdx.x;
    if(j >= dim / 2) j = j - dim;
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int quad = threadIdx.x * threadIdx.x + j * j;
    
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

        singleMax[threadIdx.x] = fabsf(mode - 1);
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

        singleMax[threadIdx.x] = fabs(mode - 1);
#endif    
    }
    
    __syncthreads();

    if (blockDim.x % 2 == 0)
    {
        j = blockDim.x / 2;
        flag = true;
    }
    else
    {
        j = blockDim.x / 2 + 1;
        flag = false;
    }
    
    while (j != 0) 
    {
        if (flag)
        {
            if (threadIdx.x < j)
            {
                if (singleMax[threadIdx.x] < singleMax[threadIdx.x + j])
                {
                    singleMax[threadIdx.x] = singleMax[threadIdx.x + j]; 
                }
            }
                
        }
        else
        {
            if (threadIdx.x < j - 1)
            {
                if (singleMax[threadIdx.x] < singleMax[threadIdx.x + j])
                {
                    singleMax[threadIdx.x] = singleMax[threadIdx.x + j]; 
                }
            }
        
        }
        
        __syncthreads();
        
        if(j % 2 != 0 && j != 1)
        {
            j++;
            flag = false;
        }
        else
            flag = true;
        j /= 2;
    }
    
    if (threadIdx.x == 0) 
    {
        devMax[blockIdx.x] = singleMax[0];
    }
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
__global__ void kernel_NormalizeFW2D(Complex *devDst,
                                     Complex *devDataF, 
                                     RFLOAT *devDataW,
                                     const int r,
                                     const int pdim,
                                     const int fdim)
{
    int pj;
    int dIdx;

    int j = blockIdx.x;
    
    if(j >= fdim / 2) 
    {
        j = j - fdim;
        pj = j + pdim;
    }
    else
        pj = j;
    
    dIdx = threadIdx.x + pj * (pdim / 2 + 1);
        
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int quad = threadIdx.x * threadIdx.x + j * j;

    if (quad < r)
    {
        devDst[dIdx] = devDataF[index] 
                     * devDataW[index];
    }
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
 * @brief 
 *
 * @param 
 * @param 
 * @param 
 */
__global__ void kernel_NormalizeP2D(RFLOAT *devDstR, 
                                    int dimSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    devDstR[index] /= dimSize;
}

/**
 * @brief 
 *
 * @param 
 * @param 
 * @param 
 */
__global__ void kernel_NormalizeP(RFLOAT *devDstR, 
                                  int length,
                                  int shift,
                                  int dimSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    while (index < length)
    {
        devDstR[index + shift] /= dimSize;
        index += blockDim.x * gridDim.x;
    }
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
__global__ void kernel_CorrectF2D(RFLOAT *devDstI, 
                                  RFLOAT *devMkb,
                                  RFLOAT nf,
                                  const int dim) 
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    if(i >= dim / 2) i = dim - i;
    if(j >= dim / 2) j = dim - j;
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int mkbIndex = mkbIndex = j * (dim / 2 + 1) + i;

    devDstI[index] = devDstI[index] 
                   / devMkb[mkbIndex]
                   * nf;
#ifdef RECONSTRUCTOR_REMOVE_NEG
        if (devDstI[index] < 0)
            devDstI[index] = 0; 
#endif
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
__global__ void kernel_CorrectF2D(RFLOAT *devDstI, 
                                  RFLOAT *devTik,
                                  const int dim) 
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    if(i >= dim / 2) i = dim - i;
    if(j >= dim / 2) j = dim - j;
    
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int mkbIndex = j * (dim / 2 + 1) + i;

    devDstI[index] = devDstI[index] / devTik[mkbIndex];
#ifdef RECONSTRUCTOR_REMOVE_NEG
        if (devDstI[index] < 0)
            devDstI[index] = 0; 
#endif
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
__global__ void kernel_TranslateI2D(Complex* devSrc, 
                                    RFLOAT ox,
                                    RFLOAT oy,
                                    int r,
                                    int shift,
                                    int dim) 
{
    int i = (threadIdx.x + shift) % (dim / 2 + 1);
    int j = (threadIdx.x + shift) / (dim / 2 + 1);
    if(j >= dim / 2) j = j - dim;
   
    RFLOAT rCol = ox / dim;
    RFLOAT rRow = oy / dim;
    RFLOAT phase;

    Complex imgTemp(0.0, 0.0);
    int quad = i * i + j * j;
    if (quad < r * r)
    {
        phase = PI_2 * (i * rCol + j * rRow);
#ifdef SINGLE_PRECISION
        imgTemp.set(cosf(-phase), sinf(-phase));
#else
        imgTemp.set(cos(-phase), sin(-phase));
#endif    
        devSrc[threadIdx.x + shift] *= imgTemp;
    } 
}

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
                                  int shift,
                                  int dim,
                                  int batch) 
{
    int i, j, k, quad;
    RFLOAT phase;
    Complex imgTemp(0.0, 0.0);

    RFLOAT rCol = ox / dim;
    RFLOAT rRow = oy / dim;
    RFLOAT rSlc = oz / dim;
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    while(index < batch)
    {
        i = (index + shift) % (dim / 2 + 1);
        j = ((index + shift) / (dim / 2 + 1)) % dim;
        k = ((index + shift) / (dim / 2 + 1)) / dim;
        if(j >= dim / 2) j = j - dim;
        if(k >= dim / 2) k = k - dim;

        quad = i * i + j * j + k * k;
        if (quad < r * r)
        {
            phase = PI_2 * (i * rCol + j * rRow + k * rSlc);
#ifdef SINGLE_PRECISION
            imgTemp.set(cosf(-phase), sinf(-phase));
#else
            imgTemp.set(cos(-phase), sin(-phase));
#endif    
            devRef[index + shift] *= imgTemp;
        } 
        
        index += blockDim.x * gridDim.x;
    }
}

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
                                int imgSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int i, j;
    RFLOAT u;

    while(index < imgSize)
    {
        i = index % dim;
        j = index / dim;
        if(i >= dim / 2) i = i - dim;
        if(j >= dim / 2) j = j - dim;

#ifdef SINGLE_PRECISION
        u = sqrtf(i * i + j * j);
#else
        u = sqrt(i * i + j * j);
#endif    
        if (u > r + ew)
        {
            devMask[index] = 0;
        }
        else
        { 
            if (u >= r)
            {
#ifdef SINGLE_PRECISION
               devMask[index] = 0.5 + 0.5 * cosf((u - r) / ew * PI);
#else
               devMask[index] = 0.5 + 0.5 * cos((u - r) / ew * PI);
#endif    
            }
            else
               devMask[index] = 1; 
        }

        index += blockDim.x * gridDim.x;
    }
}

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
                               int imgSize)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int shift = index + imgIdx * imgSize;
    
    while(index < imgSize)
    {
        dev_image[shift] = dev_image[shift] 
                         / imgSize
                         * devMask[index];
        index += blockDim.x * gridDim.x;
        shift += blockDim.x * gridDim.x;
    }
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
