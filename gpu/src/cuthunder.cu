/*
 * FileName: cuthunder.cu
 * Author  : Kunpeng WANG，Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 ***************************************************************/
#include "cuthunder.h"

#include "cufft.h"
#include "nccl.h"
#include "Config.cuh"
#include "Device.cuh"
#include "Complex.cuh"
#include "Image.cuh"
#include "Volume.cuh"
#include "TabFunction.cuh"
#include "Constructor.cuh"

#include "Kernel.cuh"

#include <cuda.h>
#include <cfloat>
#include <cuda_profiler_api.h>

/* Index for two stream buffer. */
#define NUM_STREAM 3
#define A 0
#define B 1

#define DIFF_C_THRES 1e-2
#define DIFF_C_DECREASE_THRES 0.95
#define N_DIFF_C_NO_DECREASE 2

/* Perf */
//#define PERF_SYNC_STREAM
//#define PERF_CALLBACK

namespace cuthunder {

////////////////////////////////////////////////////////////////

/**
 * Test rountines.
 *
 * ...
 */
void allocDeviceVolume(Volume& vol, int nCol, int nRow, int nSlc)
{
    vol.init(nCol, nRow, nSlc);

    Complex *dat;
    cudaMalloc((void**)&dat, vol.nSize() * sizeof(Complex));
    //cudaCheckErrors("Allocate device volume data.");

    vol.devPtr(dat);
}

__global__ void adder(Volume v1, Volume v2, Volume v3)
{
    int i = threadIdx.x - 4;
    int j = threadIdx.y - 4;
    int k = threadIdx.z - 4;

    Volume v(v2);

    Complex c = v1.getFT(i, j, k) + v2.getFT(i, j, k);
    v3.setFT(c, i, j, k);
}

void addTest()
{
    Volume v1, v2, v3;

    allocDeviceVolume(v1, 8, 8, 8);
    allocDeviceVolume(v2, 8, 8, 8);
    allocDeviceVolume(v3, 8, 8, 8);

    cudaSetDevice(0);
    //cudaCheckErrors("Set device error.");
    
    dim3 block(8, 8, 8);
    adder<<<1, block>>>(v1, v2, v3);
    //cudaCheckErrors("Lanch kernel adder.");

    cudaFree(v1.devPtr());
    //cudaCheckErrors("Free device volume memory 1.");
    cudaFree(v2.devPtr());
    //cudaCheckErrors("Free device volume memory 2.");
    cudaFree(v3.devPtr());
    //cudaCheckErrors("Free device volume memory 3.");
}


////////////////////////////////////////////////////////////////
//                     COMMON SUBROUTINES
//
//   The following routines are used by interface rountines to
// manipulate the data allocation, synchronization or transfer
// between host and device.
//

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void getAviDevice(vector<int>& gpus)
{
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    //cudaCheckErrors("get devices num.");

    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus.push_back(n);
        }
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void createVolume(Volume& vol,
                  int ndim,
                  VolCrtKind type,
                  const Complex *data = NULL)
{
    Complex *ptr = NULL;
    vol.init(ndim, ndim, ndim);

    if (type & HOST_ONLY){
        cudaHostAlloc((void**)&ptr,
                      vol.nSize() * sizeof(Complex),
                      cudaHostAllocPortable|cudaHostAllocWriteCombined);
        //cudaCheckErrors("Allocate page-lock volume data.");

        vol.hostPtr(ptr);

        if (data)
            memcpy(ptr, data, vol.nSize() * sizeof(Complex));
    }

    if (type & DEVICE_ONLY) {
        cudaMalloc((void**)&ptr, vol.nSize() * sizeof(Complex));
        //cudaCheckErrors("Allocate device volume data.");

        vol.devPtr(ptr);
    }

    if ((type & HD_SYNC) && (type & DEVICE_ONLY)) {
        if (data == NULL) return;

        cudaMemcpy((void*)vol.devPtr(),
                   (void*)data,
                   vol.nSize() * sizeof(Complex),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("Copy src volume data to device.");
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void freeVolume(Volume& vol)
{
    Complex *ptr;

    if (ptr = vol.hostPtr()) {
        cudaFreeHost(ptr);
        //cudaCheckErrors("Free host page-lock memory.");
    }
    
    if (ptr = vol.devPtr()) {
        cudaFree(ptr);
        //cudaCheckErrors("Free device memory.");
    }

    vol.clear();
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void setupSurfaceF(cudaArray *symArray,
                   cudaResourceDesc& resDesc,
                   cudaSurfaceObject_t& surfObject,
                   Complex *volume,
                   int dim)
{
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)volume, (dim / 2 + 1) * sizeof(int4), dim / 2 + 1, dim);
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    //cudaCheckErrors("copy F3D to device.");
    
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = symArray;

    cudaCreateSurfaceObject(&surfObject, &resDesc);
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void setupSurfaceT(cudaArray *symArray,
                   cudaResourceDesc& resDesc,
                   cudaSurfaceObject_t& surfObject,
                   double *volume,
                   int dim)
{
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)volume, (dim / 2 + 1) * sizeof(int2), dim / 2 + 1, dim);
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    //cudaCheckErrors("copy T3D to device.");
    
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = symArray;

    cudaCreateSurfaceObject(&surfObject, &resDesc);
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void reduceT3D(cudaArray* symArrayT[],
               cudaStream_t* stream,
               ncclComm_t* comm,
               double* T3D,
               int dimSize,
               int aviDevs,
               int nranks,
               int dim)
{
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);
    
    double *devDataT[aviDevs];

    for (int i = 0; i < aviDevs; i++)
    {
        cudaSetDevice(i);
        cudaMalloc((void**)&devDataT[i], dimSize * sizeof(double));
        //cudaCheckErrors("malloc normal devDataT.");
        
        cudaMemcpy3DParms copyParamsT = {0};
        copyParamsT.dstPtr   = make_cudaPitchedPtr((void*)devDataT[i], (dim / 2 + 1) * sizeof(int2), dim / 2 + 1, dim);
        copyParamsT.srcArray = symArrayT[i];
        copyParamsT.extent   = extent;
        copyParamsT.kind     = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParamsT);
        //cudaCheckErrors("copy T3D from array to normal.");
    }
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(i); 
        NCCLCHECK(ncclReduce((const void*)devDataT[i], 
                             (void*)devDataT[0], 
                             dimSize, 
                             ncclDouble, 
                             ncclSum,
                             0, 
                             comm[i], 
                             stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        int baseS = n * NUM_STREAM;
        cudaSetDevice(n);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
   
    if (nranks == 0)
    {
        cudaSetDevice(0);
        cudaMemcpy(T3D,
                   devDataT[0],
                   dimSize * sizeof(double),
                   cudaMemcpyDeviceToHost);
        //cudaCheckErrors("copy F3D from device to host.");
    }
    
    //free device buffers 
    for (int i = 0; i < aviDevs; ++i) 
    { 
        cudaSetDevice(i);
        cudaFree(devDataT[i]); 
    } 
    
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void reduceF3D(cudaArray* symArrayF[],
               cudaStream_t* stream,
               ncclComm_t* comm,
               Complex* F3D,
               int dimSize,
               int aviDevs,
               int nranks,
               int dim)
{
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);
    
    Complex *devDataF[aviDevs];

    for (int i = 0; i < aviDevs; i++)
    {
        cudaSetDevice(i);
        cudaMalloc((void**)&devDataF[i], dimSize * sizeof(Complex));
        //cudaCheckErrors("malloc normal devDataF.");
        
        cudaMemcpy3DParms copyParamsF = {0};
        copyParamsF.dstPtr   = make_cudaPitchedPtr((void*)devDataF[i], (dim / 2 + 1) * sizeof(int4), dim / 2 + 1, dim);
        copyParamsF.srcArray = symArrayF[i];
        copyParamsF.extent   = extent;
        copyParamsF.kind     = cudaMemcpyDeviceToDevice;
        cudaMemcpy3D(&copyParamsF);
        //cudaCheckErrors("copy T3D from array to normal.");
    }
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(i); 
        NCCLCHECK(ncclReduce((const void*)devDataF[i], 
                             (void*)devDataF[0], 
                             dimSize * 2, 
                             ncclDouble, 
                             ncclSum,
                             0, 
                             comm[i], 
                             stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        int baseS = n * NUM_STREAM;
        cudaSetDevice(n);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
   
    if (nranks == 0)
    {
        cudaSetDevice(0);
        cudaMemcpy(F3D,
                   devDataF[0],
                   dimSize * sizeof(double),
                   cudaMemcpyDeviceToHost);
        //cudaCheckErrors("copy F3D from device to host.");
    }
    
    //free device buffers 
    for (int i = 0; i < aviDevs; ++i) 
    { 
        cudaSetDevice(i);
        cudaFree(devDataF[i]); 
    } 
    
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocPGLKImagesBuffer(Complex **pglkptr, int ndim, int length)
{
    size_t size = length * (ndim / 2 + 1) * ndim;

    cudaHostAlloc((void**)pglkptr,
                  size * sizeof(Complex),
                  cudaHostAllocPortable|cudaHostAllocWriteCombined);
    //cudaCheckErrors("Alloc page-lock memory of batch size images.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocPGLKCTFAttrBuffer(CTFAttr **pglkptr, int length)
{
    cudaHostAlloc((void**)pglkptr,
                  length * sizeof(CTFAttr),
                  cudaHostAllocPortable|cudaHostAllocWriteCombined);
    //cudaCheckErrors("Alloc page-lock memory of batch CTFAttr.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void updatePGLKImagesBuffer(Complex *pglkptr,
                            const vector<Complex*>& images,
                            int ndim,
                            int basePos,
                            int batchSize)
{
    size_t imageSize = (ndim / 2 + 1) * ndim;

    for (int i = 0; i < batchSize; i++) {
        memcpy((void*)(pglkptr + i * imageSize),
               (void*)(images[basePos + i]),
               imageSize * sizeof(Complex));
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void updatePGLKCTFAttrsBuffer(CTFAttr *pglkptr,
                              const vector<CTFAttr*>& ctfas,
                              int basePos,
                              int batchSize)
{
    for (int i = 0; i < batchSize; i++) {
        memcpy((void*)(pglkptr + i),
               (void*)(ctfas[basePos + i]),
               sizeof(CTFAttr));
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
//Complex *cb_pglkptr = NULL;
//vector<Complex*> *cb_images = NULL;
//int cb_ndim = 0;
//int cb_batchSize = 0;
typedef struct {
    Complex *pglkptr;
    vector<Complex*> *images;
    int imageSize;
    int batchSize;
    int basePos;
} CB_UPIB_t;

void CUDART_CB cbUpdatePGLKImagesBuffer(cudaStream_t stream,
                                        cudaError_t status,
                                        void *data)
{
    CB_UPIB_t *args = (CB_UPIB_t *)data;
    long long shift = 0;

    for (int i = 0; i < args->batchSize; i++) {
        shift = i * args->imageSize;
        memcpy((void*)(args->pglkptr + shift),
               (void*)(*args->images)[args->basePos + i],
               args->imageSize * sizeof(Complex));
    }
}

void CUDART_CB cbUpdateImagesBuffer(cudaStream_t stream,
                                    cudaError_t status,
                                    void *data)
{
    CB_UPIB_t *args = (CB_UPIB_t *)data;
    long long shift = 0;

    for (int i = 0; i < args->batchSize; i++) {
        shift = i * args->imageSize;
        memcpy((void*)(*args->images)[args->basePos + i],
               (void*)(args->pglkptr + shift),
               args->imageSize * sizeof(Complex));
    }
}

typedef struct {
    CTFAttr *pglkptr;
    vector<CTFAttr*> *ctfa;
    int batchSize;
    int basePos;
} CB_UPIB_ta;

void CUDART_CB cbUpdatePGLKCTFABuffer(cudaStream_t stream,
                                      cudaError_t status,
                                      void *data)
{
    CB_UPIB_ta *args = (CB_UPIB_ta *)data;

    for (int i = 0; i < args->batchSize; i++) {
        memcpy((void*)(args->pglkptr + i),
               (void*)(*args->ctfa)[args->basePos + i],
               sizeof(CTFAttr));
    }
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceCTFAttrBuffer(CTFAttr **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(CTFAttr));
    //cudaCheckErrors("Alloc device memory of batch CTF.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceComplexBuffer(Complex **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(Complex));
    //cudaCheckErrors("Alloc device memory of batch param.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceParamBuffer(RFLOAT **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(RFLOAT));
    //cudaCheckErrors("Alloc device memory of batch param.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceParamBufferD(double **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(double));
    //cudaCheckErrors("Alloc device memory of batch param.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceParamBufferI(int **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(int));
    //cudaCheckErrors("Alloc device memory of batch param.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void allocDeviceRandomBuffer(int **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(int));
    //cudaCheckErrors("Alloc device memory of batch random num.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void uploadTabFunction(TabFunction& tabfunc, const RFLOAT *tab)
{
    RFLOAT *devptr;

    cudaMalloc((void**)&devptr, tabfunc.size() * sizeof(RFLOAT));
    //cudaCheckErrors("Alloc device memory for tabfunction.");

    cudaMemcpy((void*)devptr,
               (void*)tab,
               tabfunc.size() * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    //cudaCheckErrors("Copy tabfunction to device.");

    tabfunc.devPtr(devptr);
}

////////////////////////////////////////////////////////////////
//                    RECONSTRUCTION ROUTINES
//
//   Below are interface rountines implemented to accelerate the
// reconstruction process.
//

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
                  int imgNum)
{
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    //cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    int streamNum = aviDevs * NUM_STREAM;
    CTFAttr *pglk_ctfas_buf[streamNum];
    
    CTFAttr *dev_ctfas_buf[streamNum];
    RFLOAT *dev_def_buf[streamNum];
    RFLOAT *dev_k1_buf[streamNum];
    RFLOAT *dev_k2_buf[streamNum];

    CB_UPIB_ta cbArgsA[streamNum];
    
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];
    
    LOG(INFO) << "Step1: alloc Memory.";
    
    cudaStream_t stream[streamNum];

    int baseS;
    const int BATCH_SIZE = BUFF_SIZE;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        
        cudaSetDevice(gpus[n]); 
        
        cudaMalloc((void**)&deviCol[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iCol data.");
        
        cudaMalloc((void**)&deviRow[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iRow data.");
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            allocPGLKCTFAttrBuffer(&pglk_ctfas_buf[i + baseS], BATCH_SIZE);
            allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&dev_def_buf[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&dev_k1_buf[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&dev_k2_buf[i + baseS], BATCH_SIZE);
            
            cudaStreamCreate(&stream[i + baseS]);
            
        }       
    }

    LOG(INFO) << "alloc memory done, begin to cpy...";
    
    for (int n = 0; n < aviDevs; ++n) 
    {     
        cudaSetDevice(gpus[n]); 
        
        cudaMemcpy(deviCol[n],
                   iCol,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iCol.");
        
        cudaMemcpy(deviRow[n],
                   iRow,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iRow.");
    }        
        
    LOG(INFO) << "Volume memcpy done...";
        
    int batchSize = 0, smidx = 0;
    
    for (int i = 0; i < imgNum;) 
    {    
        for (int n = 0; n < aviDevs; ++n) 
        {     
            if (i >= imgNum)
                break;
           
            baseS = n * NUM_STREAM;
            batchSize = (i + BATCH_SIZE < imgNum) ? BATCH_SIZE : (imgNum - i);
            printf("batch:%d, smidx:%d, baseS:%d\n", batchSize, smidx, baseS);
            
            cudaSetDevice(gpus[n]); 

            cbArgsA[smidx + baseS].pglkptr = pglk_ctfas_buf[smidx + baseS];
            cbArgsA[smidx + baseS].ctfa = &ctfaData;
            cbArgsA[smidx + baseS].batchSize = batchSize;
            cbArgsA[smidx + baseS].basePos = i;
            cudaStreamAddCallback(stream[smidx + baseS], cbUpdatePGLKCTFABuffer, (void*)&cbArgsA[smidx + baseS], 0);
           
            cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                            pglk_ctfas_buf[smidx + baseS],
                            batchSize * sizeof(CTFAttr),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy CTFAttr to device.");
            
            kernel_ExpectPrectf<<<batchSize, 
                                  512, 
                                  0, 
                                  stream[smidx + baseS]>>>(dev_ctfas_buf[smidx + baseS],
                                                           dev_def_buf[smidx + baseS],
                                                           dev_k1_buf[smidx + baseS],
                                                           dev_k2_buf[smidx + baseS],
                                                           deviCol[n],
                                                           deviRow[n],
                                                           npxl);
            //cudaCheckErrors("kernel expectPrectf error.");
            
            cudaMemcpyAsync(def + i * npxl,
                            dev_def_buf[smidx + baseS],
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy def to host.");
            
            cudaMemcpyAsync(k1 + i,
                            dev_k1_buf[smidx + baseS],
                            batchSize * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy k1 to host.");
            
            cudaMemcpyAsync(k2 + i,
                            dev_k2_buf[smidx + baseS],
                            batchSize * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy k2 to host.");

            i += batchSize;
        }
        
        smidx = (++smidx) % NUM_STREAM;
    }

    //synchronizing on CUDA streams 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        //cudaDeviceSynchronize();
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamSynchronize(stream[i + baseS]); 
            //cudaCheckErrors("Stream synchronize.");

            cudaFreeHost(pglk_ctfas_buf[i + baseS]);
            cudaFree(dev_ctfas_buf[i + baseS]);
            cudaFree(dev_def_buf[i + baseS]);
            cudaFree(dev_k1_buf[i + baseS]);
            cudaFree(dev_k2_buf[i + baseS]);
        }
    } 
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(deviCol[n]); 
        cudaFree(deviRow[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamDestroy(stream[i + baseS]);
    } 
  
    delete[] gpus; 
    LOG(INFO) << "expectationPre done.";
}

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
                    int imgNum)
{
    LOG(INFO) << "expectation Global begin.";
    
    int dimSize;
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    //cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    int streamNum = aviDevs * NUM_STREAM;
    cudaStream_t stream[streamNum];
    
    Complex* devtraP[aviDevs];
    double* dev_trans[aviDevs]; 
    double* devpR[aviDevs]; 
    double* devpT[aviDevs]; 
    double* devnR[aviDevs]; 
    double* devRotm[aviDevs]; 
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];

#ifdef SINGLE_PRECISION
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0,cudaChannelFormatKindFloat);
#else
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindSigned);
#endif    
    cudaArray *symArray[aviDevs * nK]; 
    struct cudaResourceDesc resDesc[aviDevs * nK];
    cudaTextureObject_t texObject[aviDevs * nK];
    
    dimSize = (vdim / 2 + 1) * vdim;
    
    cudaHostRegister(volume, nK * dimSize * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register volume data.");
    
    cudaHostRegister(rot, nR * 2 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register rot data.");
    
    int baseS;
    const int BATCH_SIZE = BUFF_SIZE;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]); 
        
        for (int k = 0; k < nK; k++)
        {
            cudaMallocArray(&symArray[k + n * nK], &channelDesc, vdim / 2 + 1, vdim);
            //cudaCheckErrors("Allocate symArray data.");
        }
        
        cudaMalloc((void**)&devtraP[n], nT * npxl * sizeof(Complex));
        //cudaCheckErrors("Allocate traP data.");
        
        cudaMalloc((void**)&deviCol[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iCol data.");
        
        cudaMalloc((void**)&deviRow[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iRow data.");
        
        cudaMalloc((void**)&dev_trans[n], nT * 2 * sizeof(double));
        //cudaCheckErrors("Allocate trans data.");
        
        cudaMalloc((void**)&devnR[n], nR * 2 * sizeof(double));
        //cudaCheckErrors("Allocate nR data.");
        
        cudaMalloc((void**)&devpR[n], nR * sizeof(double));
        //cudaCheckErrors("Allocate pR data.");
        
        cudaMalloc((void**)&devpT[n], nT * sizeof(double));
        //cudaCheckErrors("Allocate pT data.");
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamCreate(&stream[i + baseS]);
            //cudaCheckErrors("stream create.");
        }       
    }

    for (int n = 0; n < aviDevs; ++n) 
    {     
        cudaSetDevice(gpus[n]); 
        
        cudaMemcpy(deviCol[n],
                   iCol,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iCol.");
        
        cudaMemcpy(deviRow[n],
                   iRow,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iRow.");
        
        cudaMemcpy(dev_trans[n],
                   trans,
                   nT * 2 * sizeof(double),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy trans.");
    
        cudaMemcpy(devpR[n],
                   pR,
                   nR * sizeof(double),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy pR.");
    
        cudaMemcpy(devpT[n],
                   pT,
                   nT * sizeof(double),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy pT.");
    
    }        
    
    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]); 
        
        cudaMemcpyAsync(devnR[n],
                        rot,
                        nR * 2 * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[0 + baseS]);
        //cudaCheckErrors("memcpy rot to device.");
        
        kernel_Translate<<<nT, 
                           512, 
                           0, 
                           stream[1 + baseS]>>>(devtraP[n],
                                                dev_trans[n],
                                                deviCol[n],
                                                deviRow[n],
                                                idim,
                                                npxl);
        //cudaCheckErrors("kernel trans.");
        
        for (int k = 0; k < nK; k++)
        {
            cudaMemcpyToArrayAsync(symArray[k + n * nK], 
                                   0, 
                                   0, 
                                   (void*)(volume + k * dimSize), 
                                   sizeof(Complex) * dimSize, 
                                   cudaMemcpyHostToDevice,
                                   stream[2 + baseS]);
            //cudaCheckErrors("memcpy array error");
        }
    }        
    
    cudaHostRegister(datP, imgNum * npxl * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register datP data.");
    
    cudaHostRegister(ctfP, imgNum * npxl * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register ctfP data.");
    
    cudaHostRegister(sigRcpP, imgNum * npxl * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register sigRcpP data.");
    
    cudaHostRegister(wC, imgNum * nK * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register wC data.");
    
    cudaHostRegister(wR, imgNum * nK * nR * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register wR data.");
    
    cudaHostRegister(wT, imgNum * nK * nT * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register wT data.");
    
    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        for (int k = 0; k < nK; k++)
        {
            memset(&resDesc[k + n * nK], 0, sizeof(resDesc[0]));
            resDesc[k + n * nK].resType = cudaResourceTypeArray;
            resDesc[k + n * nK].res.array.array = symArray[k + n * nK];
            
            cudaSetDevice(gpus[n]);
            cudaCreateTextureObject(&texObject[k + n * nK], &resDesc[k + n * nK], &td, NULL);
            //cudaCheckErrors("create TexObject.");
        }
    }        
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamSynchronize(stream[i + baseS]); 
            //cudaCheckErrors("device synchronize.");
        }
        
        cudaFree(dev_trans[n]);
        //cudaCheckErrors("Free tran.");
    }

    cudaHostUnregister(volume);
    //cudaCheckErrors("Unregister vol.");
    cudaHostUnregister(rot);
    //cudaCheckErrors("Unregister rot.");
    
    Complex* devdatP[streamNum];
    Complex* priRotP[streamNum];
    RFLOAT* devctfP[streamNum];
    RFLOAT* devsigP[streamNum];
    RFLOAT* devDvp[streamNum];
    RFLOAT* devbaseL[streamNum];
    RFLOAT* devwC[streamNum];
    RFLOAT* devwR[streamNum];
    RFLOAT* devwT[streamNum];
    RFLOAT* devcomP[streamNum];

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            allocDeviceComplexBuffer(&priRotP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceComplexBuffer(&devdatP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devsigP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devDvp[i + baseS], BATCH_SIZE * nR * nT);
            allocDeviceParamBuffer(&devbaseL[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&devwC[i + baseS], BATCH_SIZE * nK);
            allocDeviceParamBuffer(&devwR[i + baseS], BATCH_SIZE * nK * nR);
            allocDeviceParamBuffer(&devwT[i + baseS], BATCH_SIZE * nK * nT);
            
            if (nK != 1)
            {
                allocDeviceParamBuffer(&devcomP[i + baseS], BATCH_SIZE);
            }
        }       
    }

    int batchSize = 0, rbatch = 0, smidx = 0;
   
    for (int i = 0; i < imgNum;) 
    {    
        for (int n = 0; n < aviDevs; ++n) 
        {     
            if (i >= imgNum)
                break;
           
            baseS = n * NUM_STREAM;
            batchSize = (i + BATCH_SIZE < imgNum) ? BATCH_SIZE : (imgNum - i);
            
            cudaSetDevice(gpus[n]); 

            cudaMemcpyAsync(devdatP[smidx + baseS],
                            datP + i * npxl,
                            batchSize * npxl * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy datP to device.");
            
            cudaMemcpyAsync(devctfP[smidx + baseS],
                            ctfP + i * npxl,
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy ctfP to device.");
            
            cudaMemcpyAsync(devsigP[smidx + baseS],
                            sigRcpP + i * npxl,
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy sigP to device.");
                
            cudaMemsetAsync(devwC[smidx + baseS],
                            0.0,
                            batchSize * nK * sizeof(RFLOAT),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset wC.");
        
            cudaMemsetAsync(devwR[smidx + baseS],
                            0.0,
                            batchSize * nK * nR * sizeof(RFLOAT),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset wR.");
        
            cudaMemsetAsync(devwT[smidx + baseS],
                            0.0,
                            batchSize * nK * nT * sizeof(RFLOAT),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset wT.");
        
            for (int k = 0; k < nK; k++)
            {
                for (int r = 0; r < nR;)
                {
                    rbatch = (r + BATCH_SIZE < nR) ? BATCH_SIZE : (nR - r);
                
                    kernel_Project2D<<<rbatch,
                                       512,
                                       2 * sizeof(double),
                                       stream[smidx + baseS]>>>(priRotP[smidx + baseS],
                                                                devnR[n],
                                                                deviCol[n],
                                                                deviRow[n],
                                                                r,
                                                                pf,
                                                                vdim,
                                                                npxl,
                                                                interp,
                                                                texObject[k + n * nK]);

                    kernel_logDataVS<<<rbatch * batchSize * nT, 
                                       64, 
                                       64 * sizeof(RFLOAT), 
                                       stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                                priRotP[smidx + baseS],
                                                                devtraP[n],
                                                                devctfP[smidx + baseS],
                                                                devsigP[smidx + baseS],
                                                                devDvp[smidx + baseS],
                                                                r,
                                                                nR,
                                                                nT,
                                                                rbatch,
                                                                npxl);
                    
                    r += rbatch; 
                }

                if (k == 0)
                {
                    kernel_getMaxBase<<<batchSize, 
                                        512, 
                                        512 * sizeof(RFLOAT), 
                                        stream[smidx + baseS]>>>(devbaseL[smidx + baseS],
                                                                 devDvp[smidx + baseS],
                                                                 nR * nT);
                }
                else
                {
                    kernel_getMaxBase<<<batchSize, 
                                        512, 
                                        512 * sizeof(RFLOAT), 
                                        stream[smidx + baseS]>>>(devcomP[smidx + baseS],
                                                                 devDvp[smidx + baseS],
                                                                 nR * nT);

                    kernel_setBaseLine<<<batchSize, 
                                         512, 
                                         0, 
                                         stream[smidx + baseS]>>>(devcomP[smidx + baseS],
                                                                  devbaseL[smidx + baseS],
                                                                  devwC[smidx + baseS],
                                                                  devwR[smidx + baseS],
                                                                  devwT[smidx + baseS], 
                                                                  nK,
                                                                  nR,
                                                                  nT);
                }

                kernel_UpdateW<<<batchSize,
                                 nT,
                                 nT * sizeof(RFLOAT),
                                 stream[smidx + baseS]>>>(devDvp[smidx + baseS],
                                                          devbaseL[smidx + baseS],
                                                          devwC[smidx + baseS],
                                                          devwR[smidx + baseS],
                                                          devwT[smidx + baseS],
                                                          devpR[n], 
                                                          devpT[n], 
                                                          k,
                                                          nK,
                                                          nR,
                                                          nR * nT);
                    
            }

            cudaMemcpyAsync(wC + i * nK,
                            devwC[smidx + baseS],
                            batchSize * nK * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy wC to host.");
                
            cudaMemcpyAsync(wR + i * nR * nK,
                            devwR[smidx + baseS],
                            batchSize * nR  * nK * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy wR to host.");
            
            cudaMemcpyAsync(wT + i * nT * nK,
                            devwT[smidx + baseS],
                            batchSize * nT * nK * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy wT to host.");
                
            i += batchSize;
        }
        
        smidx = (++smidx) % NUM_STREAM;
    }

    //synchronizing on CUDA streams 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamSynchronize(stream[i + baseS]); 
            //cudaCheckErrors("Stream synchronize after.");
            
            cudaFree(devdatP[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(devsigP[i + baseS]);
            cudaFree(priRotP[i + baseS]);
            cudaFree(devDvp[i + baseS]);
            cudaFree(devbaseL[i + baseS]);
            cudaFree(devwC[i + baseS]);
            cudaFree(devwR[i + baseS]);
            cudaFree(devwT[i + baseS]);
            if (nK != 1)
            {
                cudaFree(devcomP[i + baseS]);
            }
        }
    } 
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(devnR[n]);
        cudaFree(devpR[n]);
        cudaFree(devpT[n]);
        cudaFree(deviCol[n]); 
        cudaFree(deviRow[n]); 
        cudaFree(devtraP[n]); 
        cudaFree(devRotm[n]); 
        
        for (int k = 0; k < nK; k++)
        {
            cudaFreeArray(symArray[k + n * nK]);
            cudaDestroyTextureObject(texObject[k + n * nK]);
        }
        
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamDestroy(stream[i + baseS]);
    } 
  
    //unregister pglk_memory
    cudaHostUnregister(datP);
    cudaHostUnregister(ctfP);
    cudaHostUnregister(sigRcpP);
    cudaHostUnregister(wC);
    cudaHostUnregister(wR);
    cudaHostUnregister(wT);
   
    delete[] gpus; 
    LOG(INFO) << "expectation Global done.";
}

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
                  int npxl)
{
    LOG(INFO) << "expectation Rotation and Translate begin.";
    
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    cudaCheckErrors("get devices num.");

    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            aviDevs = n;
        }
    }

    cudaSetDevice(aviDevs); 
    
    Complex* devtraP;
    double* dev_trans; 
    double* devnR; 
    double* devRotm; 
    int *deviCol;
    int *deviRow;

    cudaStream_t stream[2];
    
    cudaMalloc((void**)&devtraP,  (long long)nT * npxl * sizeof(Complex));
    cudaCheckErrors("Allocate traP data.");
    
    cudaMalloc((void**)&dev_trans, nT * 2 * sizeof(double));
    cudaCheckErrors("Allocate trans data.");
    
    cudaMalloc((void**)&devnR, nR * 4 * sizeof(double));
    cudaCheckErrors("Allocate nR data.");
    
    cudaMalloc((void**)&devRotm, nR * 9 * sizeof(double));
    cudaCheckErrors("Allocate rot data.");
        
    cudaMalloc((void**)&deviCol, npxl * sizeof(int));
    cudaCheckErrors("Allocate iCol data.");
    
    cudaMalloc((void**)&deviRow, npxl * sizeof(int));
    cudaCheckErrors("Allocate iRow data.");
    
    for (int i = 0; i < 2; i++)
    {
        cudaStreamCreate(&stream[i]);
        cudaCheckErrors("stream create.");
    }       

    cudaMemcpyAsync(deviCol,
                    iCol,
                    npxl * sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy iCol.");
    
    cudaMemcpyAsync(deviRow,
                    iRow,
                    npxl * sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy iRow.");
    
    cudaMemcpyAsync(dev_trans,
                    trans,
                    nT * 2 * sizeof(double),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy trans.");
    
    kernel_Translate<<<nT, 
                       512, 
                       0, 
                       stream[0]>>>(devtraP,
                                    dev_trans,
                                    deviCol,
                                    deviRow,
                                    idim,
                                    npxl);
    cudaCheckErrors("kernel trans.");
        
    cudaMemcpyAsync(traP,
                    devtraP,
                    nT * npxl * sizeof(Complex),
                    cudaMemcpyDeviceToHost,
                    stream[0]);
    cudaCheckErrors("memcpy rot to device.");
    
    int rblock;
    
    if (nR % 200 != 0)
        rblock = nR / 200 + 1;
    else
        rblock = nR / 200;

    cudaMemcpyAsync(devnR,
                    rot,
                    nR * 4 * sizeof(double),
                    cudaMemcpyHostToDevice,
                    stream[1]);
    cudaCheckErrors("memcpy rot to device.");
    
    kernel_getRotMat<<<rblock, 
                       200, 
                       200 * 18 * sizeof(double), 
                       stream[1]>>>(devRotm,
                                    devnR,
                                    nR);
    cudaCheckErrors("getRotMat3D kernel.");
    
    cudaMemcpyAsync(rotMat,
                    devRotm,
                    nR * 9 * sizeof(double),
                    cudaMemcpyDeviceToHost,
                    stream[1]);
    cudaCheckErrors("memcpy rot to device.");
    
    for (int i = 0; i < 2; i++)
    {
        cudaStreamSynchronize(stream[i]); 
        cudaCheckErrors("device synchronize.");
    }
        
    //double* dd = new double[nR * 9];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("mat.dat", "rb");
    //if (pfile == NULL)
    //    printf("open w3d error!\n");
    //if (fread(dd, sizeof(double), nR * 9, pfile) != nR * 9)
    //    printf("read w3d error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%.16lf,gw:%.16lf\n",0,dd[0],rotMat[0]);
    //for (t = 0; t < nR * 9; t++){
    //    if (fabs(rotMat[t] - dd[t]) >= 1e-15){
    //        printf("i:%d,cw:%.16lf,gw:%.16lf\n",t,dd[t],rotMat[t]);
    //        break;
    //    }
    //}
    //if (t == nR * 9)
    //    printf("successw:%d\n", nR * 9);

    //Complex* dd = new Complex[nT * npxl];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("traP.dat", "rb");
    //if (pfile == NULL)
    //    printf("open w3d error!\n");
    //if (fread(dd, sizeof(Complex), nT * npxl, pfile) != nT * npxl)
    //    printf("read w3d error!\n");
    //fclose(pfile);
    //printf("i:%d,ct:%.16lf,gt:%.16lf\n",0,dd[0].real(),traP[0].real());
    //for (t = 0; t < nT * npxl; t++){
    //    if (fabs(traP[t].real() - dd[t].real()) >= 1e-14){
    //        printf("i:%d,ct:%.16lf,gt:%.16lf\n",t,dd[t].real(),traP[t].real());
    //        break;
    //    }
    //}
    //if (t == nT * npxl)
    //    printf("successT:%d\n", nT * npxl);

    for (int i = 0; i < 2; i++)
        cudaStreamDestroy(stream[i]);
  
    cudaFree(devnR);
    cudaFree(dev_trans); 
    cudaFree(deviCol); 
    cudaFree(deviRow); 
    cudaFree(devtraP); 
    cudaFree(devRotm); 
        
    LOG(INFO) << "expect Rotation and Translate done.";
}

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
                   int npxl)
{
    LOG(INFO) << "expectation Projection begin.";
    
    const int BATCH_SIZE = BUFF_SIZE;

    int numDevs;
    cudaGetDeviceCount(&numDevs);
    cudaCheckErrors("get devices num.");

    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            aviDevs = n;
        }
    }

    cudaSetDevice(aviDevs); 
    
    cudaHostRegister(rotP, (long long)nR * npxl * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register volume data.");
    
    cudaHostRegister(rotMat, nR * 9 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register rot data.");
    
#ifdef SINGLE_PRECISION
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
#else
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
#endif    
    cudaArray *symArray; 
    cudaExtent extent = make_cudaExtent(vdim / 2 + 1, vdim, vdim);
    cudaMalloc3DArray(&symArray, &channelDesc, extent);
    cudaCheckErrors("Allocate symArray data.");
    
    cudaMemcpy3DParms copyParams;
    copyParams = {0};
#ifdef SINGLE_PRECISION
    copyParams.srcPtr = make_cudaPitchedPtr((void*)volume, 
                                            (vdim / 2 + 1) * sizeof(float2), 
                                            vdim / 2 + 1, 
                                            vdim);
#else
    copyParams.srcPtr = make_cudaPitchedPtr((void*)volume, 
                                            (vdim / 2 + 1) * sizeof(int4), 
                                            vdim / 2 + 1, 
                                            vdim);
#endif    
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    cudaCheckErrors("memcpy array error");
    
    LOG(INFO) << "sym array memcpy done.";
    
    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = symArray;
    
    cudaTextureObject_t texObject;
    cudaCreateTextureObject(&texObject, &resDesc, &td, NULL);
    cudaCheckErrors("create TexObject.");
    
    Complex* devrotP[NUM_STREAM];
    double* devRotm[NUM_STREAM]; 
    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaMalloc((void**)&devrotP[i], BATCH_SIZE * npxl * sizeof(Complex));
        cudaCheckErrors("Allocate traP data.");
        
        cudaMalloc((void**)&devRotm[i], BATCH_SIZE * 9 * sizeof(double));
        cudaCheckErrors("Allocate rot data.");
    }

    int *deviCol;
    int *deviRow;

    cudaMalloc((void**)&deviCol, npxl * sizeof(int));
    cudaCheckErrors("Allocate iCol data.");
    
    cudaMalloc((void**)&deviRow, npxl * sizeof(int));
    cudaCheckErrors("Allocate iRow data.");
    
    cudaMemcpy(deviCol,
               iCol,
               npxl * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy iCol.");
    
    cudaMemcpy(deviRow,
               iRow,
               npxl * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy iRow.");
    
    int streamNum = NUM_STREAM;
    cudaStream_t stream[streamNum];
    
    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamCreate(&stream[i]);
        cudaCheckErrors("stream create.");
    }       

    LOG(INFO) << "Projection begin.";
    
    int smidx = 0, rbatch;

    for (int r = 0; r < nR;)
    {
        rbatch = (r + BATCH_SIZE < nR) ? BATCH_SIZE : (nR - r);
    
        cudaMemcpyAsync(devRotm[smidx],
                        rotMat + r * 9,
                        rbatch * 9 * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("memcpy wT to host.");
        
        kernel_Project3D<<<rbatch,
                           512,
                           0,
                           stream[smidx]>>>(devrotP[smidx],
                                            devRotm[smidx],
                                            deviCol,
                                            deviRow,
                                            pf,
                                            vdim,
                                            npxl,
                                            interp,
                                            texObject);

        cudaMemcpyAsync(rotP + (long long)r * npxl,
                        devrotP[smidx],
                        rbatch * npxl * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        cudaCheckErrors("memcpy wT to host.");
        
        r += rbatch;
        smidx = (++smidx) % NUM_STREAM;
    }
            
    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaStreamSynchronize(stream[i]); 
        cudaCheckErrors("device synchronize.");
    }
    
    cudaDestroyTextureObject(texObject);
    
    LOG(INFO) << "Projection done.";
    
    //Complex* dd = new Complex[nR * npxl];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("rotP.dat", "rb");
    //if (pfile == NULL)
    //    printf("open w3d error!\n");
    //if (fread(dd, sizeof(Complex), nR * npxl, pfile) != nR * npxl)
    //    printf("read w3d error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%.16lf,gw:%.16lf\n",0,dd[0].imag(),rotP[0].imag());
    //for (t = 0; t < nR * npxl; t++){
    //    if (fabs(rotP[t].imag() - dd[t].imag()) >= 1e-14){
    //        printf("i:%d,cw:%.16lf,gw:%.16lf\n",t,dd[t].imag(),rotP[t].imag());
    //        break;
    //    }
    //}
    //if (t == nR * npxl)
    //    printf("successw:%d\n", nR * npxl);

    //unregister pglk_memory
    cudaHostUnregister(rotP);
    cudaHostUnregister(rotMat);
    cudaCheckErrors("unregister rotP.");
   
    cudaFree(deviCol); 
    cudaFree(deviRow); 
    cudaFreeArray(symArray);
    cudaCheckErrors("device memory free.");
        
    for (int i = 0; i < NUM_STREAM; i++)
    {
        cudaFree(devrotP[i]); 
        cudaFree(devRotm[i]); 
        cudaStreamDestroy(stream[i]);
        cudaCheckErrors("device stream destory.");
    }

    LOG(INFO) << "expect Projection done.";
}

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
                    int imgNum)
{
    LOG(INFO) << "expectation Global begin.";
 
    const int BATCH_SIZE = BUFF_SIZE;
 
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    int streamNum = aviDevs * NUM_STREAM;
    cudaStream_t stream[streamNum];
    
    Complex* devtraP[aviDevs];
    double* devpR[aviDevs];
    double* devpT[aviDevs];
    int baseS;
    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamCreate(&stream[i + baseS]);
            cudaCheckErrors("stream create.");
        }       
        
        cudaMalloc((void**)&devtraP[n], (long long)nT * npxl * sizeof(Complex));
        cudaCheckErrors("Allocate traP data.");
    
        cudaMalloc((void**)&devpR[n], nR * sizeof(double));
        cudaCheckErrors("Allocate pR data.");
    
        cudaMalloc((void**)&devpT[n], nR * sizeof(double));
        cudaCheckErrors("Allocate pT data.");
    
        cudaMemcpy(devtraP[n],
                   traP,
                   nT * npxl * sizeof(Complex),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy trans.");
    
        cudaMemcpy(devpR[n],
                   pR,
                   nR * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy pR.");
    
        cudaMemcpy(devpT[n],
                   pT,
                   nT * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy pT.");
    }        
   
    long long datSize = (long long)imgNum * npxl;

    cudaHostRegister(rotP, (long long)nR * npxl * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register datP data.");
    
    cudaHostRegister(datP, datSize * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register datP data.");
    
    cudaHostRegister(ctfP, datSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register ctfP data.");
    
    cudaHostRegister(sigRcpP, datSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register sigRcpP data.");
    
    cudaHostRegister(baseL, imgNum * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register wC data.");
    
    cudaHostRegister(wC, imgNum * nK * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register wC data.");
    
    cudaHostRegister(wR, (long long)imgNum * nK * nR * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register wR data.");
    
    cudaHostRegister(wT, (long long)imgNum * nK * nT * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register wT data.");
    
    Complex* priRotP[streamNum];
    Complex* devdatP[streamNum];
    RFLOAT* devctfP[streamNum];
    RFLOAT* devsigP[streamNum];
    RFLOAT* devDvp[streamNum];
    RFLOAT* devbaseL[streamNum];
    RFLOAT* devwC[streamNum];
    RFLOAT* devwR[streamNum];
    RFLOAT* devwT[streamNum];
    RFLOAT* devcomP[streamNum];

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            allocDeviceComplexBuffer(&priRotP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceComplexBuffer(&devdatP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devsigP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devDvp[i + baseS], BATCH_SIZE * nR * nT);
            allocDeviceParamBuffer(&devbaseL[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&devwC[i + baseS], BATCH_SIZE * nK);
            allocDeviceParamBuffer(&devwR[i + baseS], BATCH_SIZE * nK * nR);
            allocDeviceParamBuffer(&devwT[i + baseS], BATCH_SIZE * nK * nT);
            
            if (kIdx != 0)
            {
                allocDeviceParamBuffer(&devcomP[i + baseS], BATCH_SIZE);
            }
        }       
    }

    //synchronizing on CUDA streams 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamSynchronize(stream[i + baseS]); 
            cudaCheckErrors("Stream synchronize before expect.");
        }
    } 
    
    int batchSize = 0, rbatch = 0, smidx = 0;

    for (int i = 0; i < imgNum;) 
    {    
        for (int n = 0; n < aviDevs; ++n) 
        {     
            if (i >= imgNum)
                break;
           
            baseS = n * NUM_STREAM;
            batchSize = (i + BATCH_SIZE < imgNum) ? BATCH_SIZE : (imgNum - i);
            
            cudaSetDevice(gpus[n]); 

            long long imgShift = (long long)i * npxl;
            
            cudaMemcpyAsync(devdatP[smidx + baseS],
                            datP + imgShift,
                            batchSize * npxl * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy datP to device.");
            
            cudaMemcpyAsync(devctfP[smidx + baseS],
                            ctfP + imgShift,
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy ctfP to device.");
            
            cudaMemcpyAsync(devsigP[smidx + baseS],
                            sigRcpP + imgShift,
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy sigP to device.");
                
            if (kIdx == 0)
            {

                cudaMemsetAsync(devwC[smidx + baseS],
                                0.0,
                                batchSize * nK * sizeof(RFLOAT),
                                stream[smidx + baseS]);
                cudaCheckErrors("for memset wC.");
        
                cudaMemsetAsync(devwR[smidx + baseS],
                                0.0,
                                batchSize * nK * nR * sizeof(RFLOAT),
                                stream[smidx + baseS]);
                cudaCheckErrors("for memset wR.");
        
                cudaMemsetAsync(devwT[smidx + baseS],
                                0.0,
                                batchSize * nK * nT * sizeof(RFLOAT),
                                stream[smidx + baseS]);
                cudaCheckErrors("for memset wT.");
            }
            else
            {
                cudaMemcpyAsync(devbaseL[smidx + baseS],
                                baseL + i,
                                batchSize * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("for memset baseL.");
                
                cudaMemcpyAsync(devwC[smidx + baseS],
                                wC + i * nK,
                                batchSize * nK * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("for memset wC.");
        
                cudaMemcpyAsync(devwR[smidx + baseS],
                                wR + (long long)i * nR,
                                batchSize * nK * nR * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("for memset wR.");
        
                cudaMemcpyAsync(devwT[smidx + baseS],
                                wT + (long long)i * nT,
                                batchSize * nK * nT * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("for memset wT.");
            } 

            for (int r = 0; r < nR;)
            {
                rbatch = (r + BATCH_SIZE < nR) ? BATCH_SIZE : (nR - r);
            
                cudaMemcpyAsync(priRotP[smidx + baseS],
                                rotP + (long long)r * npxl,
                                rbatch * npxl * sizeof(Complex),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("memcpy datP to device.");
            
                //printf("rotP re:%.16lf, im:%.16lf\n", rotP[26 * npxl + 1064].real(), rotP[26 * npxl + 1064].imag());
                
                kernel_logDataVS<<<rbatch * batchSize * nT, 
                                   64, 
                                   64 * sizeof(RFLOAT), 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            priRotP[smidx + baseS],
                                                            devtraP[n],
                                                            devctfP[smidx + baseS],
                                                            devsigP[smidx + baseS],
                                                            devDvp[smidx + baseS],
                                                            r,
                                                            nR,
                                                            nT,
                                                            rbatch,
                                                            npxl);

                r += rbatch; 
            }
            
            if (kIdx == 0)
            {
                kernel_getMaxBase<<<batchSize, 
                                    512, 
                                    512 * sizeof(RFLOAT), 
                                    stream[smidx + baseS]>>>(devbaseL[smidx + baseS],
                                                             devDvp[smidx + baseS],
                                                             nR * nT);
            }
            else
            {
                kernel_getMaxBase<<<batchSize, 
                                    512, 
                                    512 * sizeof(RFLOAT), 
                                    stream[smidx + baseS]>>>(devcomP[smidx + baseS],
                                                             devDvp[smidx + baseS],
                                                             nR * nT);

                kernel_setBaseLine<<<batchSize, 
                                     512, 
                                     0, 
                                     stream[smidx + baseS]>>>(devcomP[smidx + baseS],
                                                              devbaseL[smidx + baseS],
                                                              devwC[smidx + baseS],
                                                              devwR[smidx + baseS],
                                                              devwT[smidx + baseS],
                                                              nK,
                                                              nR,
                                                              nT);
            }

            kernel_UpdateW<<<batchSize,
                             nT,
                             nT * sizeof(RFLOAT),
                             stream[smidx + baseS]>>>(devDvp[smidx + baseS],
                                                      devbaseL[smidx + baseS],
                                                      devwC[smidx + baseS],
                                                      devwR[smidx + baseS],
                                                      devwT[smidx + baseS],
                                                      devpR[n],
                                                      devpT[n],
                                                      kIdx,
                                                      nK,
                                                      nR,
                                                      nR * nT);
                    
            cudaMemcpyAsync(baseL + i,
                            devbaseL[smidx + baseS],
                            batchSize * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy baseL to host.");
                
            cudaMemcpyAsync(wC + i * nK,
                            devwC[smidx + baseS],
                            batchSize * nK * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy wC to host.");
                
            cudaMemcpyAsync(wR + (long long)i * nK * nR,
                            devwR[smidx + baseS],
                            batchSize * nK * nR * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy wR to host.");
            
            cudaMemcpyAsync(wT + (long long)i * nK * nT,
                            devwT[smidx + baseS],
                            batchSize * nK * nT * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy wT to host.");
                
            i += batchSize;
        }
        
        smidx = (++smidx) % NUM_STREAM;
    }

    //synchronizing on CUDA streams 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamSynchronize(stream[i + baseS]); 
            cudaCheckErrors("Stream synchronize after.");
            
            if (kIdx != 0)
            {
                cudaFree(devcomP[i + baseS]);
            }
            cudaFree(devdatP[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(devsigP[i + baseS]);
            cudaFree(priRotP[i + baseS]);
            cudaFree(devDvp[i + baseS]);
            cudaFree(devbaseL[i + baseS]);
            cudaFree(devwC[i + baseS]);
            cudaFree(devwR[i + baseS]);
            cudaFree(devwT[i + baseS]);
        }
    } 
    
    //RFLOAT* dd = new RFLOAT[imgNum];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("baseL.dat", "rb");
    //if (pfile == NULL)
    //    printf("open base error!\n");
    //if (fread(dd, sizeof(RFLOAT), imgNum, pfile) != imgNum)
    //    printf("read base error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%lf,gw:%lf\n",0,dd[0],baseL[0]);
    //for (t = 0; t < imgNum; t++){
    //    if (fabs(baseL[t] - dd[t]) >= 1e-4){
    //        printf("i:%d,cw:%lf,gw:%lf\n",t,dd[t],baseL[t]);
    //        break;
    //    }
    //}
    //if (t == imgNum)
    //    printf("successw:%d\n", imgNum);

    //cudaSetDevice(0);
    //RFLOAT* dvp = new RFLOAT[imgNum * nR * nT];
    //cudaMemcpy(dvp,
    //           devDvp[0],
    //           imgNum * nR * nT * sizeof(RFLOAT),
    //           cudaMemcpyDeviceToHost);
    //cudaCheckErrors("memcpy dvp to host.");
    //            
    //RFLOAT* dd = new RFLOAT[imgNum * nR * nT];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("dvp.dat", "rb");
    //if (pfile == NULL)
    //    printf("open dvp error!\n");
    //if (fread(dd, sizeof(RFLOAT), imgNum * nR * nT, pfile) != imgNum * nR * nT)
    //    printf("read dvp error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%lf,gw:%lf\n",0,dd[0],dvp[0]);
    //for (t = 0; t < imgNum * nR * nT; t++){
    //    if (fabs(dvp[t] - dd[t]) >= 1e-4){
    //        printf("i:%d,cw:%lf,gw:%lf\n",t,dd[t],dvp[t]);
    //        break;
    //    }
    //}
    //if (t == imgNum * nR * nT)
    //    printf("successw:%d\n", imgNum * nR * nT);

    //cudaSetDevice(0);
    //cudaMemcpy(re,
    //           devRe,
    //           npxl * sizeof(RFLOAT),
    //           cudaMemcpyDeviceToHost);
    //cudaCheckErrors("memcpy dvp to host.");
    //            
    //RFLOAT* dd = new RFLOAT[npxl];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("re.dat", "rb");
    //if (pfile == NULL)
    //    printf("open re error!\n");
    //if (fread(dd, sizeof(RFLOAT), npxl, pfile) != npxl)
    //    printf("read re error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%lf,gw:%lf\n",0,dd[0],re[0]);
    //for (t = 0; t < npxl; t++){
    //    if (fabs(re[t] - dd[t]) >= 1e-6){
    //        printf("i:%d,cw:%lf,gw:%lf\n",t,dd[t],re[t]);
    //        break;
    //    }
    //}
    //if (t == npxl)
    //    printf("successw:%d\n", npxl);
    //
    //RFLOAT ret = 0;
    //RFLOAT ddt = 0;
    //for (int i = 0; i < npxl; i++)
    //{
    //    ret += re[i];
    //    ddt += dd[i];
    //}
    //printf("re:%lf, dd:%lf\n", ret, ddt);

    //double* dd = new double[200];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("wc.dat", "rb");
    //if (pfile == NULL)
    //    printf("open wc error!\n");
    //if (fread(dd, sizeof(double), 200, pfile) != 200)
    //    printf("read wc error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%.16lf,gw:%.16lf\n",0,dd[0],wC[0]);
    //for (t = 0; t < 200; t++){
    //    if (fabs(wC[t] - dd[t]) >= 1e-12){
    //        printf("i:%d,cw:%.16lf,gw:%.16lf\n",t,dd[t],wC[t]);
    //        break;
    //    }
    //}
    //if (t == 200)
    //    printf("successw:%d\n", 200);

    //double* dd = new double[200 * nR];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("wr.dat", "rb");
    //if (pfile == NULL)
    //    printf("open wc error!\n");
    //if (fread(dd, sizeof(double), 200 * nR, pfile) != 200 * nR)
    //    printf("read wc error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%.16lf,gw:%.16lf\n",0,dd[0],wR[0]);
    //for (t = 0; t < 200 * nR; t++){
    //    if (fabs(wR[t] - dd[t]) >= 1e-14){
    //        printf("i:%d,cw:%.16lf,gw:%.16lf\n",t,dd[t],wR[t]);
    //        break;
    //    }
    //}
    //if (t == 200 * nR)
    //    printf("successw:%d\n", 200 * nR);

    //double* dd = new double[200];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("wt.dat", "rb");
    //if (pfile == NULL)
    //    printf("open wc error!\n");
    //if (fread(dd, sizeof(double), 200, pfile) != 200)
    //    printf("read wc error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%.16lf,gw:%.16lf\n",0,dd[0],wT[0]);
    //for (t = 0; t < 200; t++){
    //    if (fabs(wT[t] - dd[t]) >= 1e-14){
    //        printf("i:%d,cw:%.16lf,gw:%.16lf\n",t,dd[t],wT[t]);
    //        break;
    //    }
    //}
    //if (t == 200)
    //    printf("successw:%d\n", 200);

    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        cudaCheckErrors("set device.");
        
        cudaFree(devtraP[n]); 
        cudaFree(devpR[n]); 
        cudaFree(devpT[n]); 
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamDestroy(stream[i + baseS]);
            cudaCheckErrors("Stream destory.");
        }
    } 
  
    //unregister pglk_memory
    cudaHostUnregister(rotP);
    cudaHostUnregister(datP);
    cudaHostUnregister(ctfP);
    cudaHostUnregister(sigRcpP);
    cudaHostUnregister(baseL);
    cudaHostUnregister(wC);
    cudaHostUnregister(wR);
    cudaHostUnregister(wT);
    cudaCheckErrors("unregister rot.");
   
    delete[] gpus; 
    LOG(INFO) << "expectation Global done.";
}

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
                  int npxl)
{
    cudaSetDevice(gpuIdx); 
    
    cudaMalloc((void**)deviCol, npxl * sizeof(int));
    cudaCheckErrors("Allocate iCol data.");
    
    cudaMalloc((void**)deviRow, npxl * sizeof(int));
    cudaCheckErrors("Allocate iRow data.");
    
    cudaMemcpy(*deviCol,
               iCol,
               npxl * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy iCol.");
    
    cudaMemcpy(*deviRow,
               iRow,
               npxl * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy iRow.");
}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectPrefre(int gpuIdx,
                  RFLOAT** devfreQ,
                  RFLOAT* freQ,
                  int npxl)
{
    cudaSetDevice(gpuIdx); 
    
    cudaMalloc((void**)devfreQ, npxl * sizeof(RFLOAT));
    cudaCheckErrors("Allocate freQ data.");
    
    cudaMemcpy(*devfreQ,
               freQ,
               npxl * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy freQ.");
}

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
                   int cSearch)
{
    cudaSetDevice(gpuIdx); 

    cudaMalloc((void**)devdatP, cpyNum * npxl * sizeof(Complex));
    cudaCheckErrors("Allocate datP data.");
    
    if (cSearch != 2)
    {
        cudaMalloc((void**)devctfP, cpyNum * npxl * sizeof(RFLOAT));
        cudaCheckErrors("Allocate ctfP data.");
    }
    else
    {
        cudaMalloc((void**)devdefO, cpyNum * npxl * sizeof(RFLOAT));
        cudaCheckErrors("Allocate defocus data.");
    } 
    
    cudaMalloc((void**)devsigP, cpyNum * npxl * sizeof(RFLOAT));
    cudaCheckErrors("Allocate sigP data.");
}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalV2D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int dimSize)
{
    cudaSetDevice(gpuIdx); 
    
    cudaArray* symArray = static_cast<cudaArray*>(mgr->GetArray());

    cudaMemcpyToArray(symArray, 
                      0, 
                      0, 
                      (void*)(volume), 
                      sizeof(Complex) * dimSize, 
                      cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy array error");
}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectLocalV3D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int vdim)
{
    
    cudaSetDevice((mgr->getDeviceId())); 
    cudaCheckErrors("set Device error");
    
    cudaExtent extent = make_cudaExtent(vdim / 2 + 1, vdim, vdim);
    cudaArray* symArray = static_cast<cudaArray*>(mgr->GetArray());

    cudaMemcpy3DParms copyParams;
    copyParams = {0};
#ifdef SINGLE_PRECISION
    copyParams.srcPtr = make_cudaPitchedPtr((void*)volume, 
                                            (vdim / 2 + 1) * sizeof(float2), 
                                            vdim / 2 + 1, 
                                            vdim);
#else
    copyParams.srcPtr = make_cudaPitchedPtr((void*)volume, 
                                            (vdim / 2 + 1) * sizeof(int4), 
                                            vdim / 2 + 1, 
                                            vdim);
#endif    
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);
    cudaCheckErrors("memcpy array error");
}

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
                      int cSearch)
{
    cudaSetDevice(gpuIdx); 
    
    cudaHostAlloc((void**)wC, sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Host Alloc wC data.");
    
    cudaHostAlloc((void**)wR, mR * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldR data.");
    
    cudaHostAlloc((void**)wT, mT * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldT data.");
    
    cudaHostAlloc((void**)wD, mD * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldD data.");
    
    cudaHostAlloc((void**)oldR, mR * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldR data.");
    
    cudaHostAlloc((void**)oldT, mT * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register oldT data.");
    
    cudaHostAlloc((void**)trans, mT * 2 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register trans data.");
    
    cudaHostAlloc((void**)rot, mR * 4 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register rot data.");
    
    if (cSearch == 2)
    {
        cudaHostAlloc((void**)dpara, mD * sizeof(double), cudaHostRegisterDefault);
        cudaCheckErrors("Register dpara data.");
        
        cudaHostAlloc((void**)oldD, mD * sizeof(double), cudaHostRegisterDefault);
        cudaCheckErrors("Register oldD data.");
    }
    else
    {
        cudaHostAlloc((void**)oldD, sizeof(double), cudaHostRegisterDefault);
        cudaCheckErrors("Register oldD data.");
    }
}

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
                  int cSearch)
{
    cudaSetDevice(gpuIdx); 
 
    cudaMemcpy(devdatP + threadId * npxl,
               datP + imgId * npxl,
               npxl * sizeof(Complex),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy datP.");
    
    if (cSearch != 2)
    {
        cudaMemcpy(devctfP + threadId * npxl,
                   ctfP + imgId * npxl,
                   npxl * sizeof(RFLOAT),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy ctfP.");
    }
    else
    {
        cudaMemcpy(devdefO + threadId * npxl,
                   defO + imgId * npxl,
                   npxl * sizeof(RFLOAT),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy defO.");
    } 
    
    cudaMemcpy(devsigP + threadId * npxl,
               sigRcpP + imgId * npxl,
               npxl * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy sigP.");
}

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
                    double* dpara)
{
    cudaSetDevice(gpuIdx); 
    
    cudaMemcpyAsync(mcp->getDevR(),
                    oldR,
                    mcp->getNR() * sizeof(double),
                    cudaMemcpyHostToDevice,
                    *((cudaStream_t*)mcp->getStream()));
    cudaCheckErrors("memcpy oldR to device memory.");
           
    cudaMemcpyAsync(mcp->getDevT(),
                    oldT,
                    mcp->getNT() * sizeof(double),
                    cudaMemcpyHostToDevice,
                    *((cudaStream_t*)mcp->getStream()));
    cudaCheckErrors("memcpy oldT to device memory.");
           
    cudaMemcpyAsync(mcp->getDevnR(),
                    rot,
                    mcp->getNR() * 4 * sizeof(double),
                    cudaMemcpyHostToDevice,
                    *((cudaStream_t*)mcp->getStream()));
    cudaCheckErrors("memcpy trans to device memory.");
           
    cudaMemcpyAsync(mcp->getDevnT(),
                    trans,
                    mcp->getNT() * 2 * sizeof(double),
                    cudaMemcpyHostToDevice,
                    *((cudaStream_t*)mcp->getStream()));
    cudaCheckErrors("memcpy trans to device memory.");
           
    if (mcp->getCSearch() == 2)
    {
        cudaMemcpyAsync(mcp->getDevdP(),
                        dpara,
                        mcp->getMD() * sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("for memcpy iRow.");
           
        cudaMemcpyAsync(mcp->getDevD(),
                        oldD,
                        mcp->getMD() * sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy oldD to device memory.");
    }
    else
    {
        cudaMemcpyAsync(mcp->getDevD(),
                        oldD,
                        sizeof(double),
                        cudaMemcpyHostToDevice,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy oldD to device memory.");
    }
    //cudaStreamSynchronize(*((cudaStream_t*)mcp->getStream())); 
    //cudaCheckErrors("stream synchronize rtd.");
        
}

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
                       int interp)
{
    cudaSetDevice(gpuIdx);

    Complex* traP = reinterpret_cast<cuthunder::Complex*>(mcp->getDevtraP()); 
    Complex* rotP = reinterpret_cast<cuthunder::Complex*>(mcp->getPriRotP()); 
    
    kernel_TranslateL<<<mcp->getNT(), 
                        512, 
                        0, 
                        *((cudaStream_t*)mcp->getStream())>>>(traP,
                                                              mcp->getDevnT(),
                                                              deviCol,
                                                              deviRow,
                                                              idim,
                                                              npxl);
    cudaCheckErrors("kernel trans.");
        
    if (mcp->getCSearch() == 2)
    {
        kernel_CalCTFL<<<mcp->getMD(), 
                         512,
                         0,
                         *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevctfD(),
                                                               devdefO + datShift * npxl,
                                                               devfreQ,
                                                               mcp->getDevdP(),
                                                               phaseShift,
                                                               conT,
                                                               k1,
                                                               k2,
                                                               npxl);
        cudaCheckErrors("kernel ctf.");
    }

    kernel_Project2DL<<<mcp->getNR(), 
                        512,
                        0,
                        *((cudaStream_t*)mcp->getStream())>>>(rotP,
                                                              mcp->getDevnR(),
                                                              deviCol,
                                                              deviRow,
                                                              pf,
                                                              vdim,
                                                              npxl,
                                                              interp,
                                                              *static_cast<cudaTextureObject_t*>(mgr->GetTextureObject()));
    cudaCheckErrors("kernel Project.");
}

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
                       int interp)
{
    cudaSetDevice(gpuIdx); 
    
    Complex* traP = reinterpret_cast<cuthunder::Complex*>(mcp->getDevtraP()); 
    Complex* rotP = reinterpret_cast<cuthunder::Complex*>(mcp->getPriRotP()); 
    
    kernel_TranslateL<<<mcp->getNT(), 
                        512, 
                        0, 
                        *((cudaStream_t*)mcp->getStream())>>>(traP,
                                                              mcp->getDevnT(),
                                                              deviCol,
                                                              deviRow,
                                                              idim,
                                                              npxl);
    cudaCheckErrors("kernel trans.");
        
    if (mcp->getCSearch() == 2)
    {
        kernel_CalCTFL<<<mcp->getMD(), 
                         512,
                         0,
                         *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevctfD(),
                                                               devdefO + datShift * npxl,
                                                               devfreQ,
                                                               mcp->getDevdP(),
                                                               phaseShift,
                                                               conT,
                                                               k1,
                                                               k2,
                                                               npxl);
        cudaCheckErrors("kernel ctf.");
    }

    kernel_getRotMatL<<<1, 
                        mcp->getNR(), 
                        mcp->getNR() * 18 * sizeof(double), 
                        *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevRotm(),
                                                              mcp->getDevnR(),
                                                              mcp->getNR());
    cudaCheckErrors("getRotMat3D kernel.");
    
    kernel_Project3DL<<<mcp->getNR(), 
                        512,
                        0,
                        *((cudaStream_t*)mcp->getStream())>>>(rotP,
                                                              mcp->getDevRotm(),
                                                              deviCol,
                                                              deviRow,
                                                              pf,
                                                              vdim,
                                                              npxl,
                                                              interp,
                                                              *static_cast<cudaTextureObject_t*>(mgr->GetTextureObject()));
    cudaCheckErrors("kernel Project.");
        
}

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
                  int npxl)
{
    cudaSetDevice(gpuIdx); 
    
    Complex* traP = reinterpret_cast<cuthunder::Complex*>(mcp->getDevtraP()); 
    Complex* rotP = reinterpret_cast<cuthunder::Complex*>(mcp->getPriRotP()); 

    //RFLOAT* re = new RFLOAT[npxl];
    //RFLOAT* devre;
    //cudaMalloc((void**)&devre, npxl * sizeof(RFLOAT));
    //cudaCheckErrors("Allocate datP data.");

    if (mcp->getCSearch() != 2)
    {
        //if (l == 4)
        kernel_logDataVSL<<<mcp->getNR() * mcp->getNT(), 
                            64, 
                            64 * sizeof(RFLOAT), 
                            *((cudaStream_t*)mcp->getStream())>>>(rotP,
                                                                  traP,
                                                                  devdatP + datShift * npxl,
                                                                  devctfP + datShift * npxl,
                                                                  devsigP + datShift * npxl,
                                                                  mcp->getDevDvp(),
                                                                  //devre,
                                                                  mcp->getNT(),
                                                                  npxl);
        cudaCheckErrors("logDataVSL kernel.");

        kernel_getMaxBaseL<<<1, 
                             1024, 
                             1024 * sizeof(RFLOAT), 
                             *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevBaseL(),
                                                                   mcp->getDevDvp(),
                                                                   mcp->getNR() * mcp->getNT());
        cudaCheckErrors("getMaxBaseL kernel.");
        
        kernel_UpdateWL<<<1,
                          mcp->getNR(),
                          mcp->getNR() * 3 * sizeof(RFLOAT),
                          *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevDvp(),
                                                                mcp->getDevBaseL(),
                                                                mcp->getDevwC(),
                                                                mcp->getDevwR(),
                                                                mcp->getDevwT(),
                                                                mcp->getDevwD(),
                                                                mcp->getDevR(),
                                                                mcp->getDevT(),
                                                                mcp->getDevD(),
                                                                oldC,
                                                                mcp->getNT());
        cudaCheckErrors("UpdateWL kernel.");
        
        cudaMemcpyAsync(wC,
                        mcp->getDevwC(),
                        sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wC to host.");
        
        cudaMemcpyAsync(wR,
                        mcp->getDevwR(),
                        mcp->getNR() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wR to host.");
        
        cudaMemcpyAsync(wT,
                        mcp->getDevwT(),
                        mcp->getNT() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wT to host.");
        
        cudaMemcpyAsync(wD,
                        mcp->getDevwD(),
                        mcp->getMD() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wD to host.");
        
        //cudaMemcpyAsync(dvpA + l * mcp->getNR() * mcp->getNT(),
        //                mcp->getDevDvp(),
        //                mcp->getNR() * mcp->getNT() * sizeof(RFLOAT),
        //                cudaMemcpyDeviceToHost,
        //                *((cudaStream_t*)mcp->getStream()));
        //cudaCheckErrors("memcpy wD to host.");
        //
        //cudaMemcpyAsync(baseL + l,
        //                mcp->getDevBaseL(),
        //                sizeof(RFLOAT),
        //                cudaMemcpyDeviceToHost,
        //                *((cudaStream_t*)mcp->getStream()));
        //cudaCheckErrors("memcpy wD to host.");
        //
        //cudaMemcpyAsync(re,
        //                devre,
        //                npxl * sizeof(RFLOAT),
        //                cudaMemcpyDeviceToHost,
        //                *((cudaStream_t*)mcp->getStream()));
        //cudaCheckErrors("memcpy re to host.");
    }
    else
    {
        kernel_logDataVSLC<<<mcp->getNR() * mcp->getNT() * mcp->getMD(), 
                             512, 
                             512 * sizeof(RFLOAT), 
                             *((cudaStream_t*)mcp->getStream())>>>(rotP,
                                                                   traP,
                                                                   devdatP + datShift * npxl,
                                                                   mcp->getDevctfD(),
                                                                   devsigP + datShift * npxl,
                                                                   mcp->getDevDvp(),
                                                                   mcp->getNT(),
                                                                   mcp->getMD(),
                                                                   npxl);
        cudaCheckErrors("logDataVSLC kernel.");

        //cudaStreamSynchronize(*((cudaStream_t*)mcp->getStream())); 
        //cudaCheckErrors("stream synchronize log.");
        
        kernel_getMaxBaseL<<<1, 
                             1024, 
                             1024 * sizeof(RFLOAT), 
                             *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevBaseL(),
                                                                   mcp->getDevDvp(),
                                                                   mcp->getNR() * mcp->getNT() 
                                                                                * mcp->getMD());
        cudaCheckErrors("getMaxBaseLC kernel.");
        
        //cudaStreamSynchronize(*((cudaStream_t*)mcp->getStream())); 
        //cudaCheckErrors("stream synchronize0.");
        
        cudaMemsetAsync(mcp->getDevtT(),
                        0.0,
                        mcp->getNR() * mcp->getNT() * sizeof(RFLOAT),
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("for memset tT.");
        
        cudaMemsetAsync(mcp->getDevtD(),
                        0.0,
                        mcp->getNR() * mcp->getMD() * sizeof(RFLOAT),
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("for memset tD.");
        
        //cudaStreamSynchronize(*((cudaStream_t*)mcp->getStream())); 
        //cudaCheckErrors("stream synchronize1.");
        
        kernel_UpdateWLC<<<1,
                           mcp->getNR(),
                           2 * mcp->getNR() * sizeof(RFLOAT),
                           *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevDvp(),
                                                                 mcp->getDevBaseL(),
                                                                 mcp->getDevwC(),
                                                                 mcp->getDevwR(),
                                                                 mcp->getDevtT(),
                                                                 mcp->getDevtD(),
                                                                 mcp->getDevR(),
                                                                 mcp->getDevT(),
                                                                 mcp->getDevD(),
                                                                 oldC,
                                                                 mcp->getNT(),
                                                                 mcp->getMD());
        cudaCheckErrors("UpdateWL kernel.");
        
        //cudaStreamSynchronize(*((cudaStream_t*)mcp->getStream())); 
        //cudaCheckErrors("stream synchronize2.");
        
        kernel_ReduceW<<<mcp->getNT(),
                         mcp->getNR(),
                         mcp->getNR() * sizeof(RFLOAT),
                         *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevwT(),
                                                               mcp->getDevtT());
        cudaCheckErrors("ReduceWT kernel.");
        
        kernel_ReduceW<<<mcp->getMD(),
                         mcp->getNR(),
                         mcp->getNR() * sizeof(RFLOAT),
                         *((cudaStream_t*)mcp->getStream())>>>(mcp->getDevwD(),
                                                               mcp->getDevtD());
        cudaCheckErrors("ReduceWD kernel.");
        
        cudaMemcpyAsync(wC,
                        mcp->getDevwC(),
                        sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wC to host.");
        
        cudaMemcpyAsync(wR,
                        mcp->getDevwR(),
                        mcp->getNR() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wR to host.");
        
        cudaMemcpyAsync(wT,
                        mcp->getDevwT(),
                        mcp->getNT() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wT to host.");
        
        cudaMemcpyAsync(wD,
                        mcp->getDevwD(),
                        mcp->getMD() * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        *((cudaStream_t*)mcp->getStream()));
        cudaCheckErrors("memcpy wD to host.");
    } 
        
    cudaStreamSynchronize(*((cudaStream_t*)mcp->getStream())); 
    cudaCheckErrors("stream synchronize.");

    //if (l == 4)
    //{
    //    RFLOAT* dd = new RFLOAT[npxl];
    //    FILE* pfile;
    //    int t = 0;
    //    pfile = fopen("re.dat", "rb");
    //    if (pfile == NULL)
    //        printf("open dvp error!\n");
    //    if (fread(dd, sizeof(RFLOAT), npxl, pfile) != npxl)
    //        printf("read dvp error!\n");
    //    fclose(pfile);
    //    printf("i:%d,cdvp:%lf,gdvp:%lf\n",0,dd[0],re[0]);
    //    for (t = 0; t < npxl; t++){
    //        if (fabs(re[t] - dd[t]) >= 1e-6){
    //            printf("i:%d,cre:%lf,gre:%lf\n",t,dd[t],re[t]);
    //            break;
    //        }
    //    }
    //    RFLOAT tempG = 0;
    //    for (int m = 0; m < npxl; m++)
    //        tempG += re[m];
    //    printf("tempG:%lf\n", tempG);
    //    RFLOAT tempC = 0;
    //    for (int m = 0; m < npxl; m++)
    //        tempC += dd[m];
    //    printf("tempC:%lf\n", tempC);
    //    if (t == npxl)
    //        printf("successDVP:%d\n", npxl);
    //}
}

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
                      int cSearch)
{
    cudaSetDevice(gpuIdx); 
    
    cudaFreeHost(*wC);
    cudaFreeHost(*wR);
    cudaFreeHost(*wT);
    cudaFreeHost(*wD);
    cudaFreeHost(*oldR);
    cudaFreeHost(*oldT);
    cudaFreeHost(*oldD);
    cudaFreeHost(*trans);
    cudaFreeHost(*rot);
    if (cSearch == 2)
    {
        cudaFreeHost(*dpara);
    }
    cudaCheckErrors("Free host page-lock memory.");
}

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
                    int cSearch)
{
    cudaSetDevice(gpuIdx); 
    
    cudaFree(*devdatP);
    cudaFree(*devsigP);
    
    if (cSearch != 2)
    {
        cudaFree(*devctfP);
    }
    else
    {
        cudaFree(*devdefO);
        cudaFree(*devfreQ);
    }
    cudaCheckErrors("Free host Image data memory.");
}

/**
 * @brief  Expectation GLobal.
 *
 * @param
 * @param
 */
void expectFreeIdx(int gpuIdx,
                   int** deviCol,
                   int** deviRow)
{
    cudaSetDevice(gpuIdx); 
    
    cudaFree(*deviCol);
    cudaFree(*deviRow);
    cudaCheckErrors("Free host Pre iCol & iRow memory.");
}

/**
 * @brief Insert reales into volume.
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
               int imgNum)
{
    RFLOAT pixel = pixelSize * idim;
    int dimSize = (vdim / 2 + 1) * vdim;
    
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    //cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    ncclUniqueId commIdF, commIdO; 
    ncclComm_t commF[aviDevs];
    ncclComm_t commO[aviDevs];
    
    int nranksF, nsizeF;
    int nranksO, nsizeO;
    MPI_Comm_size(hemi, &nsizeF);
    MPI_Comm_rank(hemi, &nranksF);
    MPI_Comm_size(slav, &nsizeO);
    MPI_Comm_rank(slav, &nranksO);
   
    //NCCLCHECK(ncclCommInitAll(comm, aviDevs, gpus));

    // NCCL Communicator creation
    if (nranksF == 0)
        NCCLCHECK(ncclGetUniqueId(&commIdF));
    MPI_Bcast(&commIdF, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, hemi);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclCommInitRank(commF + i, 
                                   nsizeF * aviDevs, 
                                   commIdF, 
                                   nranksF * aviDevs + i)); 
    } 
    NCCLCHECK(ncclGroupEnd());
    
    if (nranksO == 0)
        NCCLCHECK(ncclGetUniqueId(&commIdO));
    MPI_Bcast(&commIdO, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, slav);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclCommInitRank(commO + i, 
                                   nsizeO * aviDevs, 
                                   commIdO, 
                                   nranksO * aviDevs + i)); 
    } 
    NCCLCHECK(ncclGroupEnd());
    
    int streamNum = aviDevs * NUM_STREAM;

    Complex *devdatP[streamNum];
    Complex *devtranP[streamNum];
    RFLOAT *devctfP[streamNum];
    double *dev_nr_buf[streamNum];
    double *dev_nt_buf[streamNum];
    double *dev_nd_buf[streamNum];
    double *dev_offs_buf[streamNum];
    int *dev_nc_buf[streamNum];

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    RFLOAT *devsigRcpP[streamNum];
#endif
    CTFAttr *dev_ctfas_buf[streamNum];

    LOG(INFO) << "rank" << nranksO << ": Step1: Insert Image.";
    //printf("rank%d: Step1: Insert Image.\n", nranks);
    
    Complex *devDataF[aviDevs];
    RFLOAT *devDataT[aviDevs];
    double *devO[aviDevs];
    int *devC[aviDevs];
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];
    long long imgShift = (long long)imgNum * npxl;

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    cudaHostRegister(sigRcpP, imgShift * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register sigRcpP data.");
#endif
    //register pglk_memory
    cudaHostRegister(F2D, nk * dimSize * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register F2D data.");
    
    cudaHostRegister(T2D, nk * dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register T2D data.");
    
    cudaHostRegister(O2D, nk * 2 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register O2D data.");
    
    cudaHostRegister(counter, nk * sizeof(int), cudaHostRegisterDefault);
    //cudaCheckErrors("Register O2D data.");
    
    cudaHostRegister(datP, imgShift * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register datP data.");

    cudaHostRegister(ctfP, imgShift * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register ctfP data.");

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostRegister(offS, imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register offset data.");
#endif

    cudaHostRegister(w, imgNum * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register w data.");
    
    cudaHostRegister(nC, mReco * imgNum * sizeof(int), cudaHostRegisterDefault);
    //cudaCheckErrors("Register nR data.");
    
    cudaHostRegister(nR, mReco * imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register nR data.");
    
    cudaHostRegister(nT, mReco * imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register nT data.");
    
    if (cSearch)
    {
        cudaHostRegister(nD, mReco * imgNum * sizeof(double), cudaHostRegisterDefault);
        //cudaCheckErrors("Register nT data.");
        
        cudaHostRegister(ctfaData, imgNum * sizeof(CTFAttr), cudaHostRegisterDefault);
        //cudaCheckErrors("Register ctfAdata data.");
    }

    /* Create and setup cuda stream */
    cudaStream_t stream[streamNum];

    //cudaEvent_t start[streamNum], stop[streamNum];

    int baseS;
    const int BATCH_SIZE = BUFF_SIZE;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        
        cudaSetDevice(gpus[n]); 
        cudaMalloc((void**)&devDataF[n], nk * dimSize * sizeof(Complex));
        //cudaCheckErrors("Allocate devDataF data.");

        cudaMalloc((void**)&devDataT[n], nk * dimSize * sizeof(RFLOAT));
        //cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&devO[n], nk * 2 * sizeof(double));
        //cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&devC[n], nk * sizeof(int));
        //cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&deviCol[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iCol data.");
        
        cudaMalloc((void**)&deviRow[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iRow data.");
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            if (cSearch)
            {
                allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
                allocDeviceParamBufferD(&dev_nd_buf[i + baseS], BATCH_SIZE * mReco);
            }
            
            allocDeviceComplexBuffer(&devdatP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceComplexBuffer(&devtranP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBufferI(&dev_nc_buf[i + baseS], BATCH_SIZE * mReco);
            allocDeviceParamBufferD(&dev_nr_buf[i + baseS], BATCH_SIZE * mReco * 2);
            allocDeviceParamBufferD(&dev_nt_buf[i + baseS], BATCH_SIZE * mReco * 2);
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            allocDeviceParamBufferD(&dev_offs_buf[i + baseS], BATCH_SIZE * 2);
#endif

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            allocDeviceParamBuffer(&devsigRcpP[i + baseS], BATCH_SIZE * npxl);
            //cudaCheckErrors("Allocate sigRcp data.");
#endif        
            
            cudaStreamCreate(&stream[i + baseS]);
            
            //cudaEventCreate(&start[i + baseS]); 
            //cudaEventCreate(&stop[i + baseS]);
            //cudaCheckErrors("CUDA event init.");
        }       
    }

    LOG(INFO) << "alloc memory done, begin to cpy...";
    //printf("alloc memory done, begin to cpy...\n");
    
    for (int n = 0; n < aviDevs; ++n) 
    {     
        cudaSetDevice(gpus[n]); 
        
        cudaMemcpy(deviCol[n],
                   iCol,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iCol.");
        
        cudaMemcpy(deviRow[n],
                   iRow,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iRow.");
        
    }        
        
    cudaSetDevice(gpus[0]); 
    
    cudaMemcpyAsync(devDataF[0],
                    F2D,
                    nk * dimSize * sizeof(Complex),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    //cudaCheckErrors("for memcpy F2D.");
    
    cudaMemcpyAsync(devDataT[0],
                    T2D,
                    nk * dimSize * sizeof(RFLOAT),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    //cudaCheckErrors("for memcpy T2D.");

    cudaMemcpyAsync(devO[0],
                    O2D,
                    nk * 2 * sizeof(double),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    //cudaCheckErrors("for memcpy O2D.");

    cudaMemcpyAsync(devC[0],
                    counter,
                    nk * sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    //cudaCheckErrors("for memcpy O2D.");

    for (int n = 1; n < aviDevs; ++n)
    {
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaMemsetAsync(devDataF[n],
                        0.0,
                        nk * dimSize * sizeof(Complex),
                        stream[0 + baseS]);
        //cudaCheckErrors("for memset F2D.");
        
        cudaMemsetAsync(devDataT[n],
                        0.0,
                        nk * dimSize * sizeof(RFLOAT),
                        stream[0 + baseS]);
        //cudaCheckErrors("for memset T2D.");
        
        cudaMemsetAsync(devO[n],
                        0.0,
                        nk * 2 * sizeof(double),
                        stream[0 + baseS]);
        //cudaCheckErrors("for memset O2D.");
        
        cudaMemsetAsync(devC[n],
                        0.0,
                        nk * sizeof(int),
                        stream[0 + baseS]);
        //cudaCheckErrors("for memset O2D.");
    }

    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
   
    LOG(INFO) << "Volume memcpy done...";
    //printf("device%d:Volume memcpy done...\n", n);
  
    Complex* tranP = new Complex[npxl];

    int batchSize = 0, smidx = 0;
    
    for (int i = 0; i < imgNum;) 
    {    
        for (int n = 0; n < aviDevs; ++n) 
        {     
            if (i >= imgNum)
                break;
           
            baseS = n * NUM_STREAM;
            batchSize = (i + BATCH_SIZE < imgNum) ? BATCH_SIZE : (imgNum - i);
            //printf("batch:%d, smidx:%d, baseS:%d\n", batchSize, smidx, baseS);
            
            cudaSetDevice(gpus[n]); 
            long long imgS = (long long)i * npxl;

            cudaMemcpyAsync(dev_nr_buf[smidx + baseS],
                            nR + i * mReco * 2,
                            batchSize * mReco * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy nr to device.");
            
            cudaMemcpyAsync(dev_nt_buf[smidx + baseS],
                            nT + i * mReco * 2,
                            batchSize * mReco * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy nt to device.");
            
            cudaMemcpyAsync(devdatP[smidx + baseS],
                            datP + imgS,
                            batchSize * npxl * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy reale to device.");
            
            if (cSearch)
            {
                cudaMemcpyAsync(dev_nd_buf[smidx + baseS],
                                nD + i * mReco,
                                batchSize * mReco * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                //cudaCheckErrors("memcpy nt to device.");
            
                cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                                ctfaData + i,
                                batchSize * sizeof(CTFAttr),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                //cudaCheckErrors("memcpy CTFAttr to device.");
            }
            else
            {
                cudaMemcpyAsync(devctfP[smidx + baseS],
                                ctfP + imgS,
                                batchSize * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                //cudaCheckErrors("memcpy ctf to device.");
            } 
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaMemcpyAsync(devsigRcpP[smidx + baseS],
                            sigRcpP + imgS,
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memcpy sigRcp.");
#endif            

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaMemcpyAsync(dev_offs_buf[smidx + baseS],
                            offS + 2 * i,
                            batchSize * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy offset to device.");
#endif
            
            cudaMemcpyToSymbolAsync(dev_ws_data,
                                    w + i,
                                    batchSize * sizeof(RFLOAT),
                                    smidx * batchSize * sizeof(RFLOAT),
                                    cudaMemcpyHostToDevice,
                                    stream[smidx + baseS]);
            //cudaCheckErrors("memcpy w to device constant memory.");
           
            cudaMemcpyAsync(dev_nc_buf[smidx + baseS],
                            nC + i * mReco,
                            batchSize * mReco * sizeof(int),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy nr to device.");
            
            //cudaEventRecord(start[smidx + baseS], stream[smidx + baseS]);
    
            for (int m = 0; m < mReco; m++)
            {
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                kernel_Translate<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            devtranP[smidx + baseS],
                                                            dev_offs_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            opf,
                                                            npxl,
                                                            mReco,
                                                            idim);
                
                //cudaCheckErrors("translate kernel.");
#else
                kernel_Translate<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            devtranP[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            opf,
                                                            npxl,
                                                            mReco,
                                                            idim);
                
                //cudaCheckErrors("translate kernel.");
#endif
            
                if (cSearch)
                {
                    kernel_CalculateCTF<<<batchSize, 
                                          512, 
                                          0, 
                                          stream[smidx + baseS]>>>(devctfP[smidx + baseS],
                                                                   dev_ctfas_buf[smidx + baseS],
                                                                   dev_nd_buf[smidx + baseS],
                                                                   deviCol[n],
                                                                   deviRow[n],
                                                                   pixel,
                                                                   m,
                                                                   opf,
                                                                   npxl,
                                                                   mReco);
                    
                    //cudaCheckErrors("calculateCTF kernel.");
                }
                
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                kernel_InsertT2D<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devDataT[n],
                                                            devctfP[smidx + baseS],
                                                            devsigRcpP[smidx + baseS],
                                                            dev_nr_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            npxl,
                                                            mReco,
                                                            vdim,
                                                            dimSize,
                                                            smidx);
                //cudaCheckErrors("InsertT error.");
                
                kernel_InsertF2D<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devDataF[n],
                                                            devtranP[smidx + baseS],
                                                            devctfP[smidx + baseS],
                                                            devsigRcpP[smidx + baseS],
                                                            dev_nr_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            npxl,
                                                            mReco,
                                                            vdim,
                                                            dimSize,
                                                            smidx);
                //cudaCheckErrors("InsertF error.");
#else
                kernel_InsertT2D<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devDataT[n],
                                                            devctfP[smidx + baseS],
                                                            dev_nr_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            npxl,
                                                            mReco,
                                                            vdim,
                                                            dimSize,
                                                            smidx);
                //cudaCheckErrors("InsertT error.");
                
                kernel_InsertF2D<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devDataF[n],
                                                            devtranP[smidx + baseS],
                                                            devctfP[smidx + baseS],
                                                            dev_nr_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            npxl,
                                                            mReco,
                                                            vdim,
                                                            dimSize,
                                                            smidx);
                //cudaCheckErrors("InsertF error.");

#endif            
            
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                kernel_InsertO2D<<<1, 
                                   batchSize, 
                                   2 * batchSize * sizeof(double)
                                     + batchSize * sizeof(int), 
                                   stream[smidx + baseS]>>>(devO[n],
                                                            devC[n],
                                                            dev_nr_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            dev_offs_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            m,
                                                            mReco);
                
                //cudaCheckErrors("InsertO kernel.");
#else
                kernel_InsertO2D<<<1, 
                                   batchSize, 
                                   2 * batchSize * sizeof(double) 
                                     + batchSize * sizeof(int), 
                                   stream[smidx + baseS]>>>(devO[n],
                                                            devC[n],
                                                            dev_nr_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            m,
                                                            mReco);
                
                //cudaCheckErrors("InsertO kernel.");
#endif
            }

            //cudaEventRecord(stop[smidx + baseS], stream[smidx + baseS]);

            i += batchSize;
        }
        smidx = (++smidx) % NUM_STREAM;
    }

    //synchronizing on CUDA streams to wait for start of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            
            cudaStreamSynchronize(stream[i + baseS]); 
            //cudaCheckErrors("Stream synchronize.");
            //cudaEventSynchronize(stop[i + baseS]);
            //float elapsed_time;
            //cudaEventElapsedTime(&elapsed_time, start[i + baseS], stop[i + baseS]);
            //if (n == 0 && i == 0)
            //{
            //    printf("insertF:%f\n", elapsed_time);
            //}

            if (cSearch)
            {
                cudaFree(dev_ctfas_buf[i + baseS]);
                cudaFree(dev_nd_buf[i + baseS]);
            }
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaFree(devsigRcpP[i + baseS]);
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaFree(dev_offs_buf[i + baseS]);
#endif
            cudaFree(devdatP[i + baseS]);
            cudaFree(devtranP[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(dev_nc_buf[i + baseS]);
            cudaFree(dev_nr_buf[i + baseS]);
            cudaFree(dev_nt_buf[i + baseS]);
           /* 
            cudaEventDestroy(start[i + baseS]); 
            cudaEventDestroy(stop[i + baseS]);
            //cudaCheckErrors("Event destory.");
            */          
        }
    } 
    
    //unregister pglk_memory
    cudaHostUnregister(datP);
    cudaHostUnregister(ctfP);
    cudaHostUnregister(w);
    cudaHostUnregister(nC);
    cudaHostUnregister(nR);
    cudaHostUnregister(nT);
    if (cSearch)
    {
        cudaHostUnregister(nD);
        cudaHostUnregister(ctfaData);
    }

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostUnregister(offS);
#endif
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    cudaHostUnregister(sigRcpP);
#endif

    LOG(INFO) << "Insert done.";
    //printf("Insert done.\n");

    MPI_Barrier(hemi);

    LOG(INFO) << "rank" << nranksO << ": Step2: Reduce Volume.";
    //printf("rank%d: Step2: Reduce Volume.\n", nranks);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devDataF[i], 
                                (void*)devDataF[i], 
                                nk * dimSize * 2, 
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commF[i], 
                                stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devDataT[i], 
                                (void*)devDataT[i], 
                                nk * dimSize, 
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commF[i], 
                                stream[1 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
    cudaSetDevice(gpus[0]);
    RFLOAT* sf;
    cudaMalloc((void**)&sf, nk * sizeof(RFLOAT));
    //cudaCheckErrors("Allocate sf data.");
    
    kernel_SetSF2D<<<1, nk>>>(devDataT[0],
                              sf,
                              dimSize);
    
    smidx = 0;
    for(int i = 0; i < nk; i++)
    {
        kernel_NormalizeTF2D<<<vdim, 
                               vdim / 2 + 1, 
                               0, 
                               stream[smidx]>>>(devDataF[0] + i * dimSize,
                                                devDataT[0] + i * dimSize,
                                                sf,
                                                i);

        cudaMemcpyAsync(T2D + i * dimSize,
                        devDataT[0] + i * dimSize,
                        dimSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        //cudaCheckErrors("copy T3D from device to host.");
        
        cudaMemcpyAsync(F2D + i * dimSize,
                        devDataF[0] + i * dimSize,
                        dimSize * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        //cudaCheckErrors("copy F3D from device to host.");
        
        smidx = (++smidx) % NUM_STREAM;
    }
    
    for (int i = 0; i < NUM_STREAM; i++)
        cudaStreamSynchronize(stream[i]); 
    
    cudaFree(sf); 
    
    MPI_Barrier(hemi);
    MPI_Barrier(slav);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devO[i], 
                                (void*)devO[i], 
                                nk * 2, 
                                ncclDouble,
                                ncclSum,
                                commO[i], 
                                stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devC[i], 
                                (void*)devC[i], 
                                nk, 
                                ncclInt64,
                                ncclSum,
                                commO[i], 
                                stream[1 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
    cudaSetDevice(gpus[0]);
    cudaMemcpy(O2D,
               devO[0],
               nk * 2 * sizeof(double),
               cudaMemcpyDeviceToHost);
    //cudaCheckErrors("copy O2D from device to host.");
        
    cudaMemcpy(counter,
               devC[0],
               nk * sizeof(int),
               cudaMemcpyDeviceToHost);
    //cudaCheckErrors("copy O2D from device to host.");
        
    MPI_Barrier(slav);
    
    LOG(INFO) << "rank" << nranksO << ":Step3: Copy done, free Volume and Nccl object.";
    //printf("rank%d:Step4: Copy done, free Volume and Nccl object.\n", nranks);
    
    cudaHostUnregister(F2D);
    cudaHostUnregister(T2D);
    cudaHostUnregister(O2D);
    cudaHostUnregister(counter);
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(devDataF[n]); 
        cudaFree(devDataT[n]); 
        cudaFree(devO[n]); 
        cudaFree(devC[n]); 
        cudaFree(deviCol[n]); 
        cudaFree(deviRow[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamDestroy(stream[i + baseS]);
    } 
   
    //finalizing NCCL 
    for (int i = 0; i < aviDevs; i++) 
    { 
        ncclCommDestroy(commF[i]); 
        ncclCommDestroy(commO[i]); 
    }
    
    delete[] gpus; 
}

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
              int vdim)
{
    RFLOAT pixel = pixelSize * idim;
    int dimSize = (vdim / 2 + 1) * vdim * vdim;
    
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    ncclUniqueId commIdF, commIdO; 
    ncclComm_t commF[aviDevs];
    ncclComm_t commO[aviDevs];
    
    int nranksF, nsizeF;
    int nranksO, nsizeO;
    MPI_Comm_size(hemi, &nsizeF);
    MPI_Comm_rank(hemi, &nranksF);
    MPI_Comm_size(slav, &nsizeO);
    MPI_Comm_rank(slav, &nranksO);
   
    //NCCLCHECK(ncclCommInitAll(comm, aviDevs, gpus));

    // NCCL Communicator creation
    if (nranksF == 0)
        NCCLCHECK(ncclGetUniqueId(&commIdF));
    MPI_Bcast(&commIdF, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, hemi);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclCommInitRank(commF + i, 
                                   nsizeF * aviDevs, 
                                   commIdF, 
                                   nranksF * aviDevs + i)); 
    } 
    NCCLCHECK(ncclGroupEnd());
    
    if (nranksO == 0)
        NCCLCHECK(ncclGetUniqueId(&commIdO));
    MPI_Bcast(&commIdO, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, slav);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclCommInitRank(commO + i, 
                                   nsizeO * aviDevs, 
                                   commIdO, 
                                   nranksO * aviDevs + i)); 
    } 
    NCCLCHECK(ncclGroupEnd());
    
    int streamNum = aviDevs * NUM_STREAM;

    Complex *devdatP[streamNum];
    Complex *devtranP[streamNum];
    RFLOAT *devctfP[streamNum];
    double *dev_nr_buf[streamNum];
    double *dev_nt_buf[streamNum];
    double *dev_nd_buf[streamNum];
    double *dev_offs_buf[streamNum];
    int *dev_nc_buf[streamNum];

    CTFAttr *dev_ctfas_buf[streamNum];
    double *dev_mat_buf[streamNum];

    LOG(INFO) << "rank" << nranksO << ": Step1: Insert Image.";
    //printf("rank%d: Step1: Insert Image.\n", nranks);
    
    Complex *devDataF[aviDevs];
    RFLOAT *devDataT[aviDevs];
    double *devO[aviDevs];
    int *devC[aviDevs];
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];
    long long imgShift = (long long)imgNum * npxl;

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    RFLOAT *devsigRcpP[streamNum];
    cudaHostRegister(sigRcpP, imgShift * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register sigRcpP data.");
#endif
    //register pglk_memory
    cudaHostRegister(F3D, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register F3D data.");
    
    cudaHostRegister(T3D, dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");
    
    cudaHostRegister(O3D, 3 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register O3D data.");
    
    cudaHostRegister(counter, sizeof(int), cudaHostRegisterDefault);
    cudaCheckErrors("Register O3D data.");
    
    cudaHostRegister(datP, imgShift * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register datP data.");

    cudaHostRegister(ctfP, imgShift * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register ctfP data.");

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostRegister(offS, imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register offset data.");
#endif

    cudaHostRegister(w, imgNum * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register w data.");
    
    cudaHostRegister(nC, imgNum * sizeof(int), cudaHostRegisterDefault);
    cudaCheckErrors("Register nR data.");
    
    cudaHostRegister(nR, mReco * imgNum * 4 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register nR data.");
    
    cudaHostRegister(nT, mReco * imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register nT data.");
    
    if (cSearch)
    {
        cudaHostRegister(nD, mReco * imgNum * sizeof(double), cudaHostRegisterDefault);
        cudaCheckErrors("Register nT data.");
        
        cudaHostRegister(ctfaData, imgNum * sizeof(CTFAttr), cudaHostRegisterDefault);
        cudaCheckErrors("Register ctfAdata data.");
    }

    /* Create and setup cuda stream */
    cudaStream_t stream[streamNum];

    //cudaEvent_t start[streamNum], stop[streamNum];

    int baseS;
    const int BATCH_SIZE = BUFF_SIZE;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        
        cudaSetDevice(gpus[n]); 
        cudaMalloc((void**)&devDataF[n], dimSize * sizeof(Complex));
        cudaCheckErrors("Allocate devDataF data.");

        cudaMalloc((void**)&devDataT[n], dimSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&devO[n], 3 * sizeof(double));
        cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&devC[n], sizeof(int));
        cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&deviCol[n], npxl * sizeof(int));
        cudaCheckErrors("Allocate iCol data.");
        
        cudaMalloc((void**)&deviRow[n], npxl * sizeof(int));
        cudaCheckErrors("Allocate iRow data.");
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            if (cSearch)
            {
                allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
                allocDeviceParamBufferD(&dev_nd_buf[i + baseS], BATCH_SIZE * mReco);
            }
            
            allocDeviceComplexBuffer(&devdatP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceComplexBuffer(&devtranP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBufferI(&dev_nc_buf[i + baseS], BATCH_SIZE);
            allocDeviceParamBufferD(&dev_nr_buf[i + baseS], BATCH_SIZE * mReco * 4);
            allocDeviceParamBufferD(&dev_nt_buf[i + baseS], BATCH_SIZE * mReco * 2);
            allocDeviceParamBufferD(&dev_mat_buf[i + baseS], BATCH_SIZE * mReco * 9);
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            allocDeviceParamBufferD(&dev_offs_buf[i + baseS], BATCH_SIZE * 2);
#endif
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            allocDeviceParamBuffer(&devsigRcpP[i + baseS], BATCH_SIZE * npxl);
            cudaCheckErrors("Allocate sigRcp data.");
#endif        
            
            cudaStreamCreate(&stream[i + baseS]);
            
            //cudaEventCreate(&start[i + baseS]); 
            //cudaEventCreate(&stop[i + baseS]);
            cudaCheckErrors("CUDA event init.");
        }       
    }

    LOG(INFO) << "alloc memory done, begin to cpy...";
    //printf("alloc memory done, begin to cpy...\n");
    
    for (int n = 0; n < aviDevs; ++n) 
    {     
        cudaSetDevice(gpus[n]); 
        
        cudaMemcpy(deviCol[n],
                   iCol,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy iCol.");
        
        cudaMemcpy(deviRow[n],
                   iRow,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy iRow.");
        
    }        
        
    cudaSetDevice(gpus[0]); 
    
    cudaMemcpyAsync(devDataF[0],
                    F3D,
                    dimSize * sizeof(Complex),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy F3D.");
    
    cudaMemcpyAsync(devDataT[0],
                    T3D,
                    dimSize * sizeof(RFLOAT),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy T3D.");

    cudaMemcpyAsync(devO[0],
                    O3D,
                    3 * sizeof(double),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy O3D.");

    cudaMemcpyAsync(devC[0],
                    counter,
                    sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy O3D.");

    for (int n = 1; n < aviDevs; ++n)
    {
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaMemsetAsync(devDataF[n],
                        0.0,
                        dimSize * sizeof(Complex),
                        stream[0 + baseS]);
        cudaCheckErrors("for memset F3D.");
        
        cudaMemsetAsync(devDataT[n],
                        0.0,
                        dimSize * sizeof(RFLOAT),
                        stream[0 + baseS]);
        cudaCheckErrors("for memset T3D.");
        
        cudaMemsetAsync(devO[n],
                        0.0,
                        3 * sizeof(double),
                        stream[0 + baseS]);
        cudaCheckErrors("for memset O3D.");
        
        cudaMemsetAsync(devC[n],
                        0.0,
                        sizeof(int),
                        stream[0 + baseS]);
        cudaCheckErrors("for memset O3D.");
    }

    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
   
    LOG(INFO) << "Volume memcpy done...";
    //printf("device%d:Volume memcpy done...\n", n);
        
    int batchSize = 0, smidx = 0;
    
    for (int i = 0; i < imgNum;) 
    {    
        for (int n = 0; n < aviDevs; ++n) 
        {     
            if (i >= imgNum)
                break;
           
            baseS = n * NUM_STREAM;
            batchSize = (i + BATCH_SIZE < imgNum) ? BATCH_SIZE : (imgNum - i);
            //printf("batch:%d, smidx:%d, baseS:%d\n", batchSize, smidx, baseS);
            
            cudaSetDevice(gpus[n]); 
            long long imgS = (long long)i * npxl;

            cudaMemcpyAsync(dev_nc_buf[smidx + baseS],
                            nC + i,
                            batchSize * sizeof(int),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy nr to device.");
            
            cudaMemcpyAsync(dev_nr_buf[smidx + baseS],
                            nR + i * mReco * 4,
                            batchSize * mReco * 4 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy nr to device.");
            
            kernel_getRandomR<<<batchSize, 
                                mReco, 
                                mReco * 18 * sizeof(double), 
                                stream[smidx + baseS]>>>(dev_mat_buf[smidx + baseS],
                                                         dev_nr_buf[smidx + baseS],
                                                         dev_nc_buf[smidx + baseS]);
            cudaCheckErrors("getrandomR kernel.");
            
            cudaMemcpyAsync(dev_nt_buf[smidx + baseS],
                            nT + i * mReco * 2,
                            batchSize * mReco * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy nt to device.");
            
            cudaMemcpyAsync(devdatP[smidx + baseS],
                            datP + imgS,
                            batchSize * npxl * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy image to device.");
            
            if (cSearch)
            {
                cudaMemcpyAsync(dev_nd_buf[smidx + baseS],
                                nD + i * mReco,
                                batchSize * mReco * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("memcpy nt to device.");
            
                cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                                ctfaData + i,
                                batchSize * sizeof(CTFAttr),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("memcpy CTFAttr to device.");
            }
            else
            {
                cudaMemcpyAsync(devctfP[smidx + baseS],
                                ctfP + imgS,
                                batchSize * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("memcpy ctf to device.");
            } 
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaMemcpyAsync(devsigRcpP[smidx + baseS],
                            sigRcpP + imgS,
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("for memcpy sigRcp.");
#endif            
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaMemcpyAsync(dev_offs_buf[smidx + baseS],
                            offS + 2 * i,
                            batchSize * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy offset to device.");
#endif
            
            cudaMemcpyToSymbolAsync(dev_ws_data,
                                    w + i,
                                    batchSize * sizeof(RFLOAT),
                                    smidx * batchSize * sizeof(RFLOAT),
                                    cudaMemcpyHostToDevice,
                                    stream[smidx + baseS]);
            cudaCheckErrors("memcpy w to device constant memory.");
           
            //cudaEventRecord(start[smidx + baseS], stream[smidx + baseS]);
    
            for (int m = 0; m < mReco; m++)
            {
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                kernel_Translate<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            devtranP[smidx + baseS],
                                                            dev_offs_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            opf,
                                                            npxl,
                                                            mReco,
                                                            idim);
                
                cudaCheckErrors("translate kernel.");
                
                kernel_InsertO3D<<<1, 
                                   128, 
                                   3 * 128 * sizeof(double)
                                     + 128 * sizeof(int), 
                                   stream[smidx + baseS]>>>(devO[n],
                                                            devC[n],
                                                            dev_mat_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            dev_offs_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            m,
                                                            mReco,
                                                            batchSize);
                
                cudaCheckErrors("InsertO kernel.");
#else
                kernel_Translate<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            devtranP[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            opf,
                                                            npxl,
                                                            mReco,
                                                            idim);
                
                cudaCheckErrors("translate kernel.");
                
                kernel_InsertO3D<<<1, 
                                   128, 
                                   3 * 128 * sizeof(double)
                                     + 128 * sizeof(int), 
                                   stream[smidx + baseS]>>>(devO[n],
                                                            devC[n],
                                                            dev_mat_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            dev_nc_buf[smidx + baseS],
                                                            m,
                                                            mReco,
                                                            batchSize);
                
                cudaCheckErrors("InsertO kernel.");
#endif
                
                if (cSearch)
                {
                    kernel_CalculateCTF<<<batchSize, 
                                          512, 
                                          0, 
                                          stream[smidx + baseS]>>>(devctfP[smidx + baseS],
                                                                   dev_ctfas_buf[smidx + baseS],
                                                                   dev_nd_buf[smidx + baseS],
                                                                   dev_nc_buf[smidx + baseS],
                                                                   deviCol[n],
                                                                   deviRow[n],
                                                                   pixel,
                                                                   m,
                                                                   opf,
                                                                   npxl,
                                                                   mReco);
                    
                    cudaCheckErrors("calculateCTF kernel.");
                }
                
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                kernel_InsertT<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataT[n],
                                                          devctfP[smidx + baseS],
                                                          devsigRcpP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          dev_nc_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                cudaCheckErrors("InsertT error.");
                
                kernel_InsertF<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataF[n],
                                                          devtranP[smidx + baseS],
                                                          //devdatP[smidx + baseS],
                                                          devctfP[smidx + baseS],
                                                          devsigRcpP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          dev_nc_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                cudaCheckErrors("InsertF error.");
#else
                kernel_InsertT<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataT[n],
                                                          devctfP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          dev_nc_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                cudaCheckErrors("InsertT error.");
                
                kernel_InsertF<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataF[n],
                                                          devtranP[smidx + baseS],
                                                          //devdatP[smidx + baseS],
                                                          devctfP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          dev_nc_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                cudaCheckErrors("InsertF error.");

#endif            
            }

            //cudaEventRecord(stop[smidx + baseS], stream[smidx + baseS]);

            i += batchSize;
        }
        smidx = (++smidx) % NUM_STREAM;
    }

    //synchronizing on CUDA streams to wait for start of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            
            cudaStreamSynchronize(stream[i + baseS]); 
            cudaCheckErrors("Stream synchronize.");
            //cudaEventSynchronize(stop[i + baseS]);
            //float elapsed_time;
            //cudaEventElapsedTime(&elapsed_time, start[i + baseS], stop[i + baseS]);
            //if (n == 0 && i == 0)
            //{
            //    printf("insertF:%f\n", elapsed_time);
            //}

            if (cSearch)
            {
                cudaFree(dev_ctfas_buf[i + baseS]);
                cudaFree(dev_nd_buf[i + baseS]);
            }
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaFree(devsigRcpP[i + baseS]);
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaFree(dev_offs_buf[i + baseS]);
#endif
            cudaFree(devdatP[i + baseS]);
            cudaFree(devtranP[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(dev_nc_buf[i + baseS]);
            cudaFree(dev_nr_buf[i + baseS]);
            cudaFree(dev_nt_buf[i + baseS]);
            cudaFree(dev_mat_buf[i + baseS]);
           /* 
            cudaEventDestroy(start[i + baseS]); 
            cudaEventDestroy(stop[i + baseS]);
            cudaCheckErrors("Event destory.");
            */          
        }
    } 
    
    //unregister pglk_memory
    cudaHostUnregister(datP);
    cudaHostUnregister(ctfP);
    cudaHostUnregister(w);
    cudaHostUnregister(nC);
    cudaHostUnregister(nR);
    cudaHostUnregister(nT);
    if (cSearch)
    {
        cudaHostUnregister(nD);
        cudaHostUnregister(ctfaData);
    }

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostUnregister(offS);
#endif
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    cudaHostUnregister(sigRcpP);
#endif

    LOG(INFO) << "Insert done.";
    //printf("Insert done.\n");

    MPI_Barrier(hemi);
    //MPI_Barrier(MPI_COMM_WORLD);

    LOG(INFO) << "rank" << nranksO << ": Step2: Reduce Volume.";
    //printf("rank%d: Step2: Reduce Volume.\n", nranks);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devDataF[i], 
                                (void*)devDataF[i], 
                                dimSize * 2, 
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commF[i], 
                                stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devDataT[i], 
                                (void*)devDataT[i], 
                                dimSize, 
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commF[i], 
                                stream[1 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(hemi);
    
    cudaSetDevice(gpus[0]);
    cudaMemcpy(F3D,
               devDataF[0],
               dimSize * sizeof(Complex),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy F3D from device to host.");
        
    cudaMemcpy(T3D,
               devDataT[0],
               dimSize * sizeof(RFLOAT),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy T3D from device to host.");
    
    MPI_Barrier(hemi);
    MPI_Barrier(slav);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devO[i], 
                                (void*)devO[i], 
                                3, 
                                ncclDouble,
                                ncclSum,
                                commO[i], 
                                stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devC[i], 
                                (void*)devC[i], 
                                1, 
                                ncclInt64,
                                ncclSum,
                                commO[i], 
                                stream[1 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
    cudaSetDevice(gpus[0]);
    cudaMemcpy(O3D,
               devO[0],
               3 * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy O2D from device to host.");
        
    cudaMemcpy(counter,
               devC[0],
               sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy O2D from device to host.");
        
    MPI_Barrier(slav);
    
    //MPI_Comm_rank(MPI_COMM_WORLD, &nranks);
    //if (nranks == 2){
        //cudaMemcpy(ctfP,
        //           devctfP[0],
        //           200 * npxl * sizeof(RFLOAT),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
    
        //Complex* tranP = new Complex[200 * npxl];
        //cudaMemcpy(tranP,
        //           devtranP[0],
        //           200 * npxl * sizeof(Complex),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
        //
        //double* matt = new double[200 * 9];
        //cudaMemcpy(matt,
        //           dev_mat_buf[0],
        //           200 * 9 * sizeof(RFLOAT),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
    
    
    //double* ctfc = new double[200 * npxl];
    //Complex* tranc = new Complex[200 * npxl];
    //double* matc = new double[200 * 9];

    //FILE* pfile;
    //pfile = fopen("ctfP.dat", "rb");
    //if (pfile == NULL)
    //    printf("open ctf error!\n");
    //if (fread(ctfc, sizeof(double), 200 * npxl, pfile) != 200 * npxl)
    //    printf("read ctf error!\n");
    //fclose(pfile);
    //printf("i:%d,ctfc:%.16lf,ctfg:%.16lf\n",0,ctfc[0],ctfP[0]);
    //int t = 0;
    //for (t = 0; t < 200 * npxl; t++){
    //    if (fabs(ctfP[t] - ctfc[t]) >= 1e-13){
    //        printf("i:%d,ctfc:%.16lf,ctfg:%.16lf\n",t,ctfc[t],ctfP[t]);
    //        break;
    //    }
    //}
    //if (t == 200 * npxl)
    //    printf("successCTF:%d\n", 200 * npxl);
    //
    //pfile = fopen("tranP.dat", "rb");
    //if (pfile == NULL)
    //    printf("open tran error!\n");
    //if (fread(tranc, sizeof(Complex), 200 * npxl, pfile) != 200 * npxl)
    //    printf("read tran error!\n");
    //fclose(pfile);
    //printf("i:%d,tranc:%.16lf,trang:%.16lf\n",0,tranc[0].imag(),tranP[0].imag());
    //for (t = 0; t < 200 * npxl; t++){
    //    if (fabs(tranP[t].imag() - tranc[t].imag()) >= 1e-12){
    //        printf("i:%d,tranc:%.16lf,trang:%.16lf\n",t,tranc[t].imag(),tranP[t].imag());
    //        break;
    //    }
    //}
    //if (t == 200 * npxl)
    //    printf("successTran:%d\n", 200 * 9);

    //pfile = fopen("mat.dat", "rb");
    //if (pfile == NULL)
    //    printf("open mat error!\n");
    //if (fread(matc, sizeof(double), 200 * 9, pfile) != 200 * 9)
    //    printf("read mat error!\n");
    //fclose(pfile);
    //printf("i:%d,matc:%.16lf,matg:%.16lf\n",0,matc[0],matt[0]);
    //for (t = 0; t < 200 * 9; t++){
    //    if (fabs(matt[t] - matc[t]) >= 1e-10){
    //        printf("i:%d,matc:%.16lf,matg:%.16lf\n",t,matc[t],matt[t]);
    //        break;
    //    }
    //}
    //if (t == 200 * 9)
    //    printf("successmat:%d\n", 200 * 9);

    //Complex* f3d = new Complex[dimSize];
    //Complex* t3d = new Complex[dimSize];

    //FILE* pfile;
    //pfile = fopen("f3db.dat", "rb");
    //if (pfile == NULL)
    //    printf("open f3d error!\n");
    //if (fread(f3d, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read f3d error!\n");
    //fclose(pfile);
    //printf("i:%d,f3d:%.16lf,F3D:%.16lf\n",0,f3d[0].real(),F3D[0].real());
    //int t = 0;
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(F3D[t].real() - f3d[t].real()) >= 1e-9){
    //        printf("i:%d,f3d:%.16lf,F3D:%.16lf\n",t,f3d[t].real(),F3D[t].real());
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successF:%d\n", dimSize);

    //pfile = fopen("t3db.dat", "rb");
    //if (pfile == NULL)
    //    printf("open t3d error!\n");
    //if (fread(t3d, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read t3d error!\n");
    //fclose(pfile);
    //printf("i:%d,t3d:%.16lf,T3D:%.16lf\n",0,t3d[0].real(),T3D[0]);
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(T3D[t] - t3d[t].real()) >= 1e-9){
    //        printf("i:%d,t3d:%.16lf,T3D:%.16lf\n",t,t3d[t].real(),T3D[t]);
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successT:%d\n", dimSize); 

    LOG(INFO) << "rank" << nranksO << ":Step3: Copy done, free Volume and Nccl object.";
    //printf("rank%d:Step4: Copy done, free Volume and Nccl object.\n", nranks);
    
    cudaHostUnregister(F3D);
    cudaHostUnregister(T3D);
    cudaHostUnregister(O3D);
    cudaHostUnregister(counter);
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(devDataF[n]); 
        cudaFree(devDataT[n]); 
        cudaFree(devO[n]); 
        cudaFree(devC[n]); 
        cudaFree(deviCol[n]); 
        cudaFree(deviRow[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamDestroy(stream[i + baseS]);
    } 
   
    //finalizing NCCL 
    for (int i = 0; i < aviDevs; i++) 
    { 
        ncclCommDestroy(commF[i]); 
        ncclCommDestroy(commO[i]); 
    }
    
    delete[] gpus; 
}

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
              int vdim)
{
    RFLOAT pixel = pixelSize * idim;
    int dimSize = (vdim / 2 + 1) * vdim * vdim;
    
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    ncclUniqueId commIdF, commIdO; 
    ncclComm_t commF[aviDevs];
    ncclComm_t commO[aviDevs];
    
    int nranksF, nsizeF;
    int nranksO, nsizeO;
    MPI_Comm_size(hemi, &nsizeF);
    MPI_Comm_rank(hemi, &nranksF);
    MPI_Comm_size(slav, &nsizeO);
    MPI_Comm_rank(slav, &nranksO);
   
    //NCCLCHECK(ncclCommInitAll(comm, aviDevs, gpus));

    // NCCL Communicator creation
    if (nranksF == 0)
        NCCLCHECK(ncclGetUniqueId(&commIdF));
    MPI_Bcast(&commIdF, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, hemi);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclCommInitRank(commF + i, 
                                   nsizeF * aviDevs, 
                                   commIdF, 
                                   nranksF * aviDevs + i)); 
    } 
    NCCLCHECK(ncclGroupEnd());
    
    if (nranksO == 0)
        NCCLCHECK(ncclGetUniqueId(&commIdO));
    MPI_Bcast(&commIdO, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, slav);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclCommInitRank(commO + i, 
                                   nsizeO * aviDevs, 
                                   commIdO, 
                                   nranksO * aviDevs + i)); 
    } 
    NCCLCHECK(ncclGroupEnd());
    
    int streamNum = aviDevs * NUM_STREAM;

    Complex *devdatP[streamNum];
    Complex *devtranP[streamNum];
    RFLOAT *devctfP[streamNum];
    double *dev_nr_buf[streamNum];
    double *dev_nt_buf[streamNum];
    double *dev_nd_buf[streamNum];
    double *dev_offs_buf[streamNum];

    CTFAttr *dev_ctfas_buf[streamNum];
    double *dev_mat_buf[streamNum];

    LOG(INFO) << "rank" << nranksO << ": Step1: Insert Image.";
    //printf("rank%d: Step1: Insert Image.\n", nranks);
    
    Complex *devDataF[aviDevs];
    RFLOAT *devDataT[aviDevs];
    double *devO[aviDevs];
    int *devC[aviDevs];
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];
    long long imgSize = (long long)imgNum * npxl;

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    RFLOAT *devsigRcpP[streamNum];
    cudaHostRegister(sigRcpP, imgSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register sigRcpP data.");
#endif
    //register pglk_memory
    cudaHostRegister(F3D, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register F3D data.");
    
    cudaHostRegister(T3D, dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");
    
    cudaHostRegister(O3D, 3 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register O3D data.");
    
    cudaHostRegister(counter, sizeof(int), cudaHostRegisterDefault);
    cudaCheckErrors("Register O3D data.");
    
    cudaHostRegister(datP, imgSize * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register datP data.");

    cudaHostRegister(ctfP, imgSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register ctfP data.");

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostRegister(offS, imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register offset data.");
#endif

    cudaHostRegister(w, imgNum * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register w data.");
    
    cudaHostRegister(nR, mReco * imgNum * 4 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register nR data.");
    
    cudaHostRegister(nT, mReco * imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    cudaCheckErrors("Register nT data.");
    
    if (cSearch)
    {
        cudaHostRegister(nD, mReco * imgNum * sizeof(double), cudaHostRegisterDefault);
        cudaCheckErrors("Register nT data.");
        
        cudaHostRegister(ctfaData, imgNum * sizeof(CTFAttr), cudaHostRegisterDefault);
        cudaCheckErrors("Register ctfAdata data.");
    }

    /* Create and setup cuda stream */
    cudaStream_t stream[streamNum];

    //cudaEvent_t start[streamNum], stop[streamNum];

    int baseS;
    const int BATCH_SIZE = BUFF_SIZE;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        
        cudaSetDevice(gpus[n]); 
        cudaMalloc((void**)&devDataF[n], dimSize * sizeof(Complex));
        cudaCheckErrors("Allocate devDataF data.");

        cudaMalloc((void**)&devDataT[n], dimSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&devO[n], 3 * sizeof(double));
        cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&devC[n], sizeof(int));
        cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&deviCol[n], npxl * sizeof(int));
        cudaCheckErrors("Allocate iCol data.");
        
        cudaMalloc((void**)&deviRow[n], npxl * sizeof(int));
        cudaCheckErrors("Allocate iRow data.");
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            if (cSearch)
            {
                allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
                allocDeviceParamBufferD(&dev_nd_buf[i + baseS], BATCH_SIZE * mReco);
            }
            
            allocDeviceComplexBuffer(&devdatP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceComplexBuffer(&devtranP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBufferD(&dev_nr_buf[i + baseS], BATCH_SIZE * mReco * 4);
            allocDeviceParamBufferD(&dev_nt_buf[i + baseS], BATCH_SIZE * mReco * 2);
            allocDeviceParamBufferD(&dev_mat_buf[i + baseS], BATCH_SIZE * mReco * 9);
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            allocDeviceParamBufferD(&dev_offs_buf[i + baseS], BATCH_SIZE * 2);
#endif

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            allocDeviceParamBuffer(&devsigRcpP[i + baseS], BATCH_SIZE * npxl);
            cudaCheckErrors("Allocate sigRcp data.");
#endif        
            
            cudaStreamCreate(&stream[i + baseS]);
            
            //cudaEventCreate(&start[i + baseS]); 
            //cudaEventCreate(&stop[i + baseS]);
            //cudaCheckErrors("CUDA event init.");
        }       
    }

    LOG(INFO) << "alloc memory done, begin to cpy...";
    //printf("alloc memory done, begin to cpy...\n");
    
    for (int n = 0; n < aviDevs; ++n) 
    {     
        cudaSetDevice(gpus[n]); 
        
        cudaMemcpy(deviCol[n],
                   iCol,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy iCol.");
        
        cudaMemcpy(deviRow[n],
                   iRow,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaCheckErrors("for memcpy iRow.");
        
    }        
        
    cudaSetDevice(gpus[0]); 
    
    cudaMemcpyAsync(devDataF[0],
                    F3D,
                    dimSize * sizeof(Complex),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy F3D.");
    
    cudaMemcpyAsync(devDataT[0],
                    T3D,
                    dimSize * sizeof(RFLOAT),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy T3D.");

    cudaMemcpyAsync(devO[0],
                    O3D,
                    3 * sizeof(double),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy O3D.");

    cudaMemcpyAsync(devC[0],
                    counter,
                    sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    cudaCheckErrors("for memcpy O3D.");

    for (int n = 1; n < aviDevs; ++n)
    {
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaMemsetAsync(devDataF[n],
                        0.0,
                        dimSize * sizeof(Complex),
                        stream[0 + baseS]);
        cudaCheckErrors("for memset F3D.");
        
        cudaMemsetAsync(devDataT[n],
                        0.0,
                        dimSize * sizeof(RFLOAT),
                        stream[0 + baseS]);
        cudaCheckErrors("for memset T3D.");
        
        cudaMemsetAsync(devO[n],
                        0.0,
                        3 * sizeof(double),
                        stream[0 + baseS]);
        cudaCheckErrors("for memset O3D.");
        
        cudaMemsetAsync(devC[n],
                        0.0,
                        sizeof(int),
                        stream[0 + baseS]);
        cudaCheckErrors("for memset O3D.");
    }

    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
   
    LOG(INFO) << "Volume memcpy done...";
    //printf("device%d:Volume memcpy done...\n", n);
        
    int batchSize = 0, smidx = 0;
    
    for (int i = 0; i < imgNum;) 
    {    
        for (int n = 0; n < aviDevs; ++n) 
        {     
            if (i >= imgNum)
                break;
           
            baseS = n * NUM_STREAM;
            batchSize = (i + BATCH_SIZE < imgNum) ? BATCH_SIZE : (imgNum - i);
            //printf("batch:%d, smidx:%d, baseS:%d\n", batchSize, smidx, baseS);
            
            cudaSetDevice(gpus[n]); 

            long long imgShift = (long long)i * npxl;
            long long mrShift  = (long long)i * mReco;
            
            cudaMemcpyAsync(dev_nr_buf[smidx + baseS],
                            nR + mrShift * 4,
                            batchSize * mReco * 4 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy nr to device.");
            
            kernel_getRandomR<<<batchSize, 
                                mReco, 
                                mReco * 18 * sizeof(double), 
                                stream[smidx + baseS]>>>(dev_mat_buf[smidx + baseS],
                                                         dev_nr_buf[smidx + baseS]);
            cudaCheckErrors("getrandomR kernel.");
            
            cudaMemcpyAsync(dev_nt_buf[smidx + baseS],
                            nT + mrShift * 2,
                            batchSize * mReco * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy nt to device.");
            
            cudaMemcpyAsync(devdatP[smidx + baseS],
                            datP + imgShift,
                            batchSize * npxl * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy image to device.");
            
            if (cSearch)
            {
                cudaMemcpyAsync(dev_nd_buf[smidx + baseS],
                                nD + mrShift,
                                batchSize * mReco * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("memcpy nt to device.");
            
                cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                                ctfaData + i,
                                batchSize * sizeof(CTFAttr),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("memcpy CTFAttr to device.");
            }
            else
            {
                cudaMemcpyAsync(devctfP[smidx + baseS],
                                ctfP + imgShift,
                                batchSize * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                cudaCheckErrors("memcpy ctf to device.");
            } 
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaMemcpyAsync(devsigRcpP[smidx + baseS],
                            sigRcpP + imgShift,
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("for memcpy sigRcp.");
#endif            
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaMemcpyAsync(dev_offs_buf[smidx + baseS],
                            offS + 2 * i,
                            batchSize * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy offset to device.");
#endif
            
            cudaMemcpyToSymbolAsync(dev_ws_data,
                                    w + i,
                                    batchSize * sizeof(RFLOAT),
                                    smidx * batchSize * sizeof(RFLOAT),
                                    cudaMemcpyHostToDevice,
                                    stream[smidx + baseS]);
            cudaCheckErrors("memcpy w to device constant memory.");
           
            //cudaEventRecord(start[smidx + baseS], stream[smidx + baseS]);
    
            for (int m = 0; m < mReco; m++)
            {
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
                kernel_Translate<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            devtranP[smidx + baseS],
                                                            dev_offs_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            opf,
                                                            npxl,
                                                            mReco,
                                                            idim);
                
                cudaCheckErrors("translate kernel.");

                kernel_InsertO3D<<<1, 
                                   128, 
                                   3 * 128 * sizeof(double) 
                                     + 128 * sizeof(int),
                                   stream[smidx + baseS]>>>(devO[n],
                                                            devC[n],
                                                            dev_mat_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            dev_offs_buf[smidx + baseS],
                                                            m,
                                                            mReco,
                                                            batchSize);
                
                cudaCheckErrors("InsertO kernel.");
#else
                kernel_Translate<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            devtranP[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            opf,
                                                            npxl,
                                                            mReco,
                                                            idim);
                
                cudaCheckErrors("translate kernel.");

                kernel_InsertO3D<<<1, 
                                   128, 
                                   3 * 128 * sizeof(double)
                                     + 128 * sizeof(int),
                                   stream[smidx + baseS]>>>(devO[n],
                                                            devC[n],
                                                            dev_mat_buf[smidx + baseS],
                                                            dev_nt_buf[smidx + baseS],
                                                            m,
                                                            mReco,
                                                            batchSize);
                
                cudaCheckErrors("InsertO kernel.");
#endif
                if (cSearch)
                {
                    kernel_CalculateCTF<<<batchSize, 
                                          512, 
                                          0, 
                                          stream[smidx + baseS]>>>(devctfP[smidx + baseS],
                                                                   dev_ctfas_buf[smidx + baseS],
                                                                   dev_nd_buf[smidx + baseS],
                                                                   deviCol[n],
                                                                   deviRow[n],
                                                                   pixel,
                                                                   m,
                                                                   opf,
                                                                   npxl,
                                                                   mReco);
                    
                    cudaCheckErrors("calculateCTF kernel.");
                }
                
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                kernel_InsertT<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataT[n],
                                                          devctfP[smidx + baseS],
                                                          devsigRcpP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                cudaCheckErrors("InsertT error.");
                
                kernel_InsertF<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataF[n],
                                                          devtranP[smidx + baseS],
                                                          //devdatP[smidx + baseS],
                                                          devctfP[smidx + baseS],
                                                          devsigRcpP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                cudaCheckErrors("InsertF error.");
#else
                kernel_InsertT<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataT[n],
                                                          devctfP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                cudaCheckErrors("InsertT error.");
                
                kernel_InsertF<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataF[n],
                                                          devtranP[smidx + baseS],
                                                          //devdatP[smidx + baseS],
                                                          devctfP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                cudaCheckErrors("InsertF error.");

#endif            
            }

            //cudaEventRecord(stop[smidx + baseS], stream[smidx + baseS]);

            i += batchSize;
        }
        smidx = (++smidx) % NUM_STREAM;
    }

    //synchronizing on CUDA streams to wait for start of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            
            cudaStreamSynchronize(stream[i + baseS]); 
            cudaCheckErrors("Stream synchronize.");
            //cudaEventSynchronize(stop[i + baseS]);
            //float elapsed_time;
            //cudaEventElapsedTime(&elapsed_time, start[i + baseS], stop[i + baseS]);
            //if (n == 0 && i == 0)
            //{
            //    printf("insertF:%f\n", elapsed_time);
            //}

            if (cSearch)
            {
                cudaFree(dev_ctfas_buf[i + baseS]);
                cudaFree(dev_nd_buf[i + baseS]);
            }
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaFree(devsigRcpP[i + baseS]);
#endif
#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
            cudaFree(dev_offs_buf[i + baseS]);
#endif
            cudaFree(devdatP[i + baseS]);
            cudaFree(devtranP[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(dev_nr_buf[i + baseS]);
            cudaFree(dev_nt_buf[i + baseS]);
            cudaFree(dev_mat_buf[i + baseS]);
           /* 
            cudaEventDestroy(start[i + baseS]); 
            cudaEventDestroy(stop[i + baseS]);
            cudaCheckErrors("Event destory.");
            */          
        }
    } 
    
    //unregister pglk_memory
    cudaHostUnregister(datP);
    cudaHostUnregister(ctfP);
    cudaHostUnregister(w);
    cudaHostUnregister(nR);
    cudaHostUnregister(nT);
    if (cSearch)
    {
        cudaHostUnregister(nD);
        cudaHostUnregister(ctfaData);
    }

#ifdef OPTIMISER_RECENTRE_IMAGE_EACH_ITERATION
    cudaHostUnregister(offS);
#endif
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    cudaHostUnregister(sigRcpP);
#endif

    cudaCheckErrors("host memory unregister.");
    
    LOG(INFO) << "Insert done.";
    //printf("Insert done.\n");

    MPI_Barrier(hemi);
    //MPI_Barrier(MPI_COMM_WORLD);

    LOG(INFO) << "rank" << nranksO << ": Step2: Reduce Volume.";
    //printf("rank%d: Step2: Reduce Volume.\n", nranks);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devDataF[i], 
                                (void*)devDataF[i], 
                                dimSize * 2, 
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commF[i], 
                                stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devDataT[i], 
                                (void*)devDataT[i], 
                                dimSize, 
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                commF[i], 
                                stream[1 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(hemi);
    
    cudaSetDevice(gpus[0]);
    cudaMemcpy(F3D,
               devDataF[0],
               dimSize * sizeof(Complex),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy F3D from device to host.");
        
    cudaMemcpy(T3D,
               devDataT[0],
               dimSize * sizeof(RFLOAT),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy T3D from device to host.");
    
    MPI_Barrier(hemi);
    MPI_Barrier(slav);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devO[i], 
                                (void*)devO[i], 
                                3, 
                                ncclDouble,
                                ncclSum,
                                commO[i], 
                                stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devC[i], 
                                (void*)devC[i], 
                                1, 
                                ncclInt64,
                                ncclSum,
                                commO[i], 
                                stream[1 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
    cudaSetDevice(gpus[0]);
    cudaMemcpy(O3D,
               devO[0],
               3 * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy O2D from device to host.");
        
    cudaMemcpy(counter,
               devC[0],
               sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("copy O2D from device to host.");
        
    //MPI_Barrier(slav);
    
    //MPI_Comm_rank(MPI_COMM_WORLD, &nranks);
    //if (nranks == 2){
        //cudaMemcpy(ctfP,
        //           devctfP[0],
        //           200 * npxl * sizeof(RFLOAT),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
    
        //Complex* tranP = new Complex[200 * npxl];
        //cudaMemcpy(tranP,
        //           devtranP[0],
        //           200 * npxl * sizeof(Complex),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
        //
        //double* matt = new double[200 * 9];
        //cudaMemcpy(matt,
        //           dev_mat_buf[0],
        //           200 * 9 * sizeof(RFLOAT),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
    
    
    //double* ctfc = new double[200 * npxl];
    //Complex* tranc = new Complex[200 * npxl];
    //double* matc = new double[200 * 9];

    //FILE* pfile;
    //pfile = fopen("ctfP.dat", "rb");
    //if (pfile == NULL)
    //    printf("open ctf error!\n");
    //if (fread(ctfc, sizeof(double), 200 * npxl, pfile) != 200 * npxl)
    //    printf("read ctf error!\n");
    //fclose(pfile);
    //printf("i:%d,ctfc:%.16lf,ctfg:%.16lf\n",0,ctfc[0],ctfP[0]);
    //int t = 0;
    //for (t = 0; t < 200 * npxl; t++){
    //    if (fabs(ctfP[t] - ctfc[t]) >= 1e-13){
    //        printf("i:%d,ctfc:%.16lf,ctfg:%.16lf\n",t,ctfc[t],ctfP[t]);
    //        break;
    //    }
    //}
    //if (t == 200 * npxl)
    //    printf("successCTF:%d\n", 200 * npxl);
    //
    //pfile = fopen("tranP.dat", "rb");
    //if (pfile == NULL)
    //    printf("open tran error!\n");
    //if (fread(tranc, sizeof(Complex), 200 * npxl, pfile) != 200 * npxl)
    //    printf("read tran error!\n");
    //fclose(pfile);
    //printf("i:%d,tranc:%.16lf,trang:%.16lf\n",0,tranc[0].imag(),tranP[0].imag());
    //for (t = 0; t < 200 * npxl; t++){
    //    if (fabs(tranP[t].imag() - tranc[t].imag()) >= 1e-12){
    //        printf("i:%d,tranc:%.16lf,trang:%.16lf\n",t,tranc[t].imag(),tranP[t].imag());
    //        break;
    //    }
    //}
    //if (t == 200 * npxl)
    //    printf("successTran:%d\n", 200 * 9);

    //pfile = fopen("mat.dat", "rb");
    //if (pfile == NULL)
    //    printf("open mat error!\n");
    //if (fread(matc, sizeof(double), 200 * 9, pfile) != 200 * 9)
    //    printf("read mat error!\n");
    //fclose(pfile);
    //printf("i:%d,matc:%.16lf,matg:%.16lf\n",0,matc[0],matt[0]);
    //for (t = 0; t < 200 * 9; t++){
    //    if (fabs(matt[t] - matc[t]) >= 1e-10){
    //        printf("i:%d,matc:%.16lf,matg:%.16lf\n",t,matc[t],matt[t]);
    //        break;
    //    }
    //}
    //if (t == 200 * 9)
    //    printf("successmat:%d\n", 200 * 9);

    //Complex* f3d = new Complex[dimSize];
    //Complex* t3d = new Complex[dimSize];

    //FILE* pfile;
    //pfile = fopen("f3db.dat", "rb");
    //if (pfile == NULL)
    //    printf("open f3d error!\n");
    //if (fread(f3d, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read f3d error!\n");
    //fclose(pfile);
    //printf("i:%d,f3d:%.16lf,F3D:%.16lf\n",0,f3d[0].real(),F3D[0].real());
    //int t = 0;
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(F3D[t].real() - f3d[t].real()) >= 1e-9){
    //        printf("i:%d,f3d:%.16lf,F3D:%.16lf\n",t,f3d[t].real(),F3D[t].real());
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successF:%d\n", dimSize);

    //pfile = fopen("t3db.dat", "rb");
    //if (pfile == NULL)
    //    printf("open t3d error!\n");
    //if (fread(t3d, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read t3d error!\n");
    //fclose(pfile);
    //printf("i:%d,t3d:%.16lf,T3D:%.16lf\n",0,t3d[0].real(),T3D[0]);
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(T3D[t] - t3d[t].real()) >= 1e-9){
    //        printf("i:%d,t3d:%.16lf,T3D:%.16lf\n",t,t3d[t].real(),T3D[t]);
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successT:%d\n", dimSize); 

    LOG(INFO) << "rank" << nranksO << ":Step3: Copy done, free Volume and Nccl object.";
    //printf("rank%d:Step4: Copy done, free Volume and Nccl object.\n", nranks);
    
    cudaHostUnregister(F3D);
    cudaHostUnregister(T3D);
    cudaHostUnregister(O3D);
    cudaHostUnregister(counter);
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(devDataF[n]); 
        cudaFree(devDataT[n]); 
        cudaFree(devO[n]); 
        cudaFree(devC[n]); 
        cudaFree(deviCol[n]); 
        cudaFree(deviRow[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamDestroy(stream[i + baseS]);
    } 
   
    //finalizing NCCL 
    for (int i = 0; i < aviDevs; i++) 
    { 
        ncclCommDestroy(commF[i]); 
        ncclCommDestroy(commO[i]); 
    }
    
    delete[] gpus; 
}

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
             int vdim)
{
    RFLOAT pixel = pixelSize * idim;
    int dimSize = (vdim / 2 + 1) * vdim * vdim;
    
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    //cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    ncclUniqueId commId; 
    ncclComm_t comm[aviDevs];
    
    int nranks, nsize;
    //MPI_Comm_size(MPI_COMM_WORLD, &nsize);
    //MPI_Comm_rank(MPI_COMM_WORLD, &nranks);
    MPI_Comm_size(hemi, &nsize);
    MPI_Comm_rank(hemi, &nranks);
    
    // NCCL Communicator creation
    if (nranks == 0)
        NCCLCHECK(ncclGetUniqueId(&commId));
    MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, hemi);
    //MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclCommInitRank(comm + i, 
                                   nsize * aviDevs, 
                                   commId, 
                                   nranks * aviDevs + i)); 
    } 
    NCCLCHECK(ncclGroupEnd());
    
    int streamNum = aviDevs * NUM_STREAM;

    Complex *devdatP[streamNum];
    Complex *devtranP[streamNum];
    RFLOAT *devctfP[streamNum];
    double *dev_nr_buf[streamNum];
    double *dev_nt_buf[streamNum];
    double *dev_nd_buf[streamNum];
    double *dev_offs_buf[streamNum];

    CTFAttr *dev_ctfas_buf[streamNum];
    double *dev_ramD_buf[streamNum];
    
    double *dev_ramR_buf[streamNum];
    double *dev_mat_buf[streamNum];
    double *dev_tran_buf[streamNum];

    LOG(INFO) << "rank" << nranks << ": Step1: Insert Image.";
    //printf("rank%d: Step1: Insert Image.\n", nranks);
    
    Complex *devDataF[aviDevs];
    RFLOAT *devDataT[aviDevs];
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    RFLOAT *devsigRcpP[streamNum];

    cudaHostRegister(sigRcpP, imgNum * npxl * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register sigRcpP data.");
#endif
    //register pglk_memory
    cudaHostRegister(F3D, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register F3D data.");
    
    cudaHostRegister(T3D, dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register T3D data.");
    
    cudaHostRegister(datP, imgNum * npxl * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register datP data.");

    cudaHostRegister(ctfP, imgNum * npxl * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register ctfP data.");

    cudaHostRegister(offS, imgNum * 2 * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register offset data.");

    cudaHostRegister(w, imgNum * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register w data.");
    
    cudaHostRegister(nR, rSize * imgNum * 4 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register nR data.");
    
    cudaHostRegister(nT, tSize * imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register nT data.");
    
    if (cSearch)
    {
        cudaHostRegister(nD, mReco * imgNum * sizeof(double), cudaHostRegisterDefault);
        //cudaCheckErrors("Register nT data.");
        
        cudaHostRegister(ctfaData, imgNum * sizeof(CTFAttr), cudaHostRegisterDefault);
        //cudaCheckErrors("Register ctfAdata data.");
    }

    /* Create and setup cuda stream */
    cudaStream_t stream[streamNum];

    //cudaEvent_t start[streamNum], stop[streamNum];

    int baseS;
    const int BATCH_SIZE = BUFF_SIZE;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        
        cudaSetDevice(gpus[n]); 
        cudaMalloc((void**)&devDataF[n], dimSize * sizeof(Complex));
        //cudaCheckErrors("Allocate devDataF data.");

        cudaMalloc((void**)&devDataT[n], dimSize * sizeof(RFLOAT));
        //cudaCheckErrors("Allocate devDataT data.");
        
        cudaMalloc((void**)&deviCol[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iCol data.");
        
        cudaMalloc((void**)&deviRow[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iRow data.");
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            if (cSearch)
            {
                allocDeviceCTFAttrBuffer(&dev_ctfas_buf[i + baseS], BATCH_SIZE);
                allocDeviceParamBufferD(&dev_nd_buf[i + baseS], BATCH_SIZE * dSize);
                allocDeviceParamBufferD(&dev_ramD_buf[i + baseS], BATCH_SIZE * mReco);
            }
            
            allocDeviceComplexBuffer(&devdatP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceComplexBuffer(&devtranP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBufferD(&dev_offs_buf[i + baseS], BATCH_SIZE * 2);
            allocDeviceParamBufferD(&dev_nr_buf[i + baseS], BATCH_SIZE * rSize * 4);
            allocDeviceParamBufferD(&dev_nt_buf[i + baseS], BATCH_SIZE * tSize * 2);
            allocDeviceParamBufferD(&dev_ramR_buf[i + baseS], BATCH_SIZE * mReco * 4);
            allocDeviceParamBufferD(&dev_mat_buf[i + baseS], BATCH_SIZE * mReco * 9);
            allocDeviceParamBufferD(&dev_tran_buf[i + baseS], BATCH_SIZE * mReco * 2);
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            allocDeviceParamBuffer(&devsigRcpP[i + baseS], BATCH_SIZE * npxl);
            //cudaCheckErrors("Allocate sigRcp data.");
#endif        
            
            cudaStreamCreate(&stream[i + baseS]);
            
            //cudaEventCreate(&start[i + baseS]); 
            //cudaEventCreate(&stop[i + baseS]);
            //cudaCheckErrors("CUDA event init.");
        }       
    }

    LOG(INFO) << "alloc memory done, begin to cpy...";
    //printf("alloc memory done, begin to cpy...\n");
    
    for (int n = 0; n < aviDevs; ++n) 
    {     
        cudaSetDevice(gpus[n]); 
        
        cudaMemcpy(deviCol[n],
                   iCol,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iCol.");
        
        cudaMemcpy(deviRow[n],
                   iRow,
                   npxl * sizeof(int),
                   cudaMemcpyHostToDevice);
        //cudaCheckErrors("for memcpy iRow.");
        
    }        
        
    cudaSetDevice(gpus[0]); 
    
    cudaMemcpyAsync(devDataF[0],
                    F3D,
                    dimSize * sizeof(Complex),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    //cudaCheckErrors("for memcpy F3D.");
    
    cudaMemcpyAsync(devDataT[0],
                    T3D,
                    dimSize * sizeof(RFLOAT),
                    cudaMemcpyHostToDevice,
                    stream[0]);
    //cudaCheckErrors("for memcpy T3D.");

    for (int n = 1; n < aviDevs; ++n)
    {
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaMemsetAsync(devDataF[n],
                        0.0,
                        dimSize * sizeof(Complex),
                        stream[0 + baseS]);
        //cudaCheckErrors("for memset F3D.");
        
        cudaMemsetAsync(devDataT[n],
                        0.0,
                        dimSize * sizeof(RFLOAT),
                        stream[0 + baseS]);
        //cudaCheckErrors("for memset T3D.");
    }

    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
   
    LOG(INFO) << "Volume memcpy done...";
    //printf("device%d:Volume memcpy done...\n", n);
        
    unsigned long out;  
    struct timeval tm;
    int batchSize = 0, smidx = 0;
    
    for (int i = 0; i < imgNum;) 
    {    
        for (int n = 0; n < aviDevs; ++n) 
        {     
            if (i >= imgNum)
                break;
           
            baseS = n * NUM_STREAM;
            batchSize = (i + BATCH_SIZE < imgNum) ? BATCH_SIZE : (imgNum - i);
            //printf("batch:%d, smidx:%d, baseS:%d\n", batchSize, smidx, baseS);
            
            out = 0xc96a3ea3d89ceb52UL;
            gettimeofday(&tm, NULL);
            out ^= (unsigned long)tm.tv_sec;
            out ^= (unsigned long)tm.tv_usec << 7;
            out ^= (unsigned long)(uintptr_t)&tm;
            out ^= (unsigned long)getpid() << 3;
        
            cudaSetDevice(gpus[n]); 

            cudaMemcpyAsync(dev_nt_buf[smidx + baseS],
                            nT + i * tSize * 2,
                            batchSize * tSize * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy nt to device.");
            
            cudaMemcpyAsync(dev_nr_buf[smidx + baseS],
                            nR + i * rSize * 4,
                            batchSize * rSize * 4 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy nr to device.");
            
            if (cSearch)
            {
                cudaMemcpyAsync(dev_nd_buf[smidx + baseS],
                                nD + i * dSize,
                                batchSize * dSize * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                //cudaCheckErrors("memcpy nd to device.");
                   
                kernel_getRandomCTD<<<batchSize, 
                                      mReco, 
                                      0, 
                                      stream[smidx + baseS]>>>(dev_nt_buf[smidx + baseS],
                                                               dev_tran_buf[smidx + baseS],
                                                               dev_nd_buf[smidx + baseS],
                                                               dev_ramD_buf[smidx + baseS],
                                                               dev_nr_buf[smidx + baseS],
                                                               dev_ramR_buf[smidx + baseS],
                                                               out,
                                                               rSize,
                                                               tSize,
                                                               dSize
                                                               );
                
                //cudaCheckErrors("getrandomCTD kernel.");
            }
            else
            {
                kernel_getRandomCTD<<<batchSize, 
                                      mReco, 
                                      0, 
                                      stream[smidx + baseS]>>>(dev_nt_buf[smidx + baseS],
                                                               dev_tran_buf[smidx + baseS],
                                                               dev_nr_buf[smidx + baseS],
                                                               dev_ramR_buf[smidx + baseS],
                                                               out,
                                                               rSize,
                                                               tSize
                                                               );
                
                //cudaCheckErrors("getrandomCTD kernel.");
            }
            
            kernel_getRandomR<<<batchSize, 
                                mReco, 
                                mReco * 18 * sizeof(double), 
                                stream[smidx + baseS]>>>(dev_mat_buf[smidx + baseS],
                                                         dev_ramR_buf[smidx + baseS]);
            //cudaCheckErrors("getrandomR kernel.");
            
            cudaMemcpyAsync(devdatP[smidx + baseS],
                            datP + i * npxl,
                            batchSize * npxl * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy image to device.");
            
            if (cSearch)
            {
                cudaMemcpyAsync(dev_ctfas_buf[smidx + baseS],
                                ctfaData + i,
                                batchSize * sizeof(CTFAttr),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                //cudaCheckErrors("memcpy CTFAttr to device.");
            }
            else
            {
                cudaMemcpyAsync(devctfP[smidx + baseS],
                                ctfP + i * npxl,
                                batchSize * npxl * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                //cudaCheckErrors("memcpy ctf to device.");
            } 
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaMemcpyAsync(devsigRcpP[smidx + baseS],
                            sigRcpP + i * npxl,
                            batchSize * npxl * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memcpy sigRcp.");
#endif            
            cudaMemcpyAsync(dev_offs_buf[smidx + baseS],
                            offS + 2 * i,
                            batchSize * 2 * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy offset to device.");
            
            cudaMemcpyToSymbolAsync(dev_ws_data,
                                    w + i,
                                    batchSize * sizeof(RFLOAT),
                                    smidx * batchSize * sizeof(RFLOAT),
                                    cudaMemcpyHostToDevice,
                                    stream[smidx + baseS]);
            //cudaCheckErrors("memcpy w to device constant memory.");
           
            //cudaEventRecord(start[smidx + baseS], stream[smidx + baseS]);
    
            for (int m = 0; m < mReco; m++)
            {
                kernel_Translate<<<batchSize, 
                                   512, 
                                   0, 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            devtranP[smidx + baseS],
                                                            dev_offs_buf[smidx + baseS],
                                                            dev_tran_buf[smidx + baseS],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            m,
                                                            opf,
                                                            npxl,
                                                            mReco,
                                                            idim);
                
                //cudaCheckErrors("translate kernel.");
                
                if (cSearch)
                {
                    kernel_CalculateCTF<<<batchSize, 
                                          512, 
                                          0, 
                                          stream[smidx + baseS]>>>(devctfP[smidx + baseS],
                                                                   dev_ctfas_buf[smidx + baseS],
                                                                   dev_ramD_buf[smidx + baseS],
                                                                   deviCol[n],
                                                                   deviRow[n],
                                                                   pixel,
                                                                   m,
                                                                   opf,
                                                                   npxl,
                                                                   mReco);
                    
                    //cudaCheckErrors("calculateCTF kernel.");
                }
                
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
                kernel_InsertT<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataT[n],
                                                          devctfP[smidx + baseS],
                                                          devsigRcpP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                //cudaCheckErrors("InsertT error.");
                
                kernel_InsertF<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataF[n],
                                                          devtranP[smidx + baseS],
                                                          devctfP[smidx + baseS],
                                                          devsigRcpP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                //cudaCheckErrors("InsertF error.");
#else
                kernel_InsertT<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataT[n],
                                                          devctfP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                //cudaCheckErrors("InsertT error.");
                
                kernel_InsertF<<<batchSize, 
                                 512, 
                                 9 * sizeof(double), 
                                 stream[smidx + baseS]>>>(devDataF[n],
                                                          devtranP[smidx + baseS],
                                                          devctfP[smidx + baseS],
                                                          dev_mat_buf[smidx + baseS],
                                                          deviCol[n],
                                                          deviRow[n],
                                                          m,
                                                          npxl,
                                                          mReco,
                                                          vdim,
                                                          smidx);
                //cudaCheckErrors("InsertF error.");

#endif            
            }

            //cudaEventRecord(stop[smidx + baseS], stream[smidx + baseS]);

            i += batchSize;
        }
        smidx = (++smidx) % NUM_STREAM;
    }

    //synchronizing on CUDA streams to wait for start of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            
            cudaStreamSynchronize(stream[i + baseS]); 
            //cudaCheckErrors("Stream synchronize.");
            //cudaEventSynchronize(stop[i + baseS]);
            //float elapsed_time;
            //cudaEventElapsedTime(&elapsed_time, start[i + baseS], stop[i + baseS]);
            //if (n == 0 && i == 0)
            //{
            //    printf("insertF:%f\n", elapsed_time);
            //}

            if (cSearch)
            {
                cudaFree(dev_ctfas_buf[i + baseS]);
                cudaFree(dev_nd_buf[i + baseS]);
                cudaFree(dev_ramD_buf[i + baseS]);
            }
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaFree(devsigRcpP[i + baseS]);
#endif
            cudaFree(dev_offs_buf[i + baseS]);
            cudaFree(devdatP[i + baseS]);
            cudaFree(devtranP[i + baseS]);
            cudaFree(devctfP[i + baseS]);
            cudaFree(dev_nr_buf[i + baseS]);
            cudaFree(dev_nt_buf[i + baseS]);
            cudaFree(dev_mat_buf[i + baseS]);
            cudaFree(dev_tran_buf[i + baseS]);
            cudaFree(dev_ramR_buf[i + baseS]);
           /* 
            cudaEventDestroy(start[i + baseS]); 
            cudaEventDestroy(stop[i + baseS]);
            //cudaCheckErrors("Event destory.");
            */          
        }
    } 
    
    //unregister pglk_memory
    cudaHostUnregister(datP);
    cudaHostUnregister(ctfP);
    cudaHostUnregister(offS);
    cudaHostUnregister(w);
    cudaHostUnregister(nR);
    cudaHostUnregister(nT);
    if (cSearch)
    {
        cudaHostUnregister(nD);
        cudaHostUnregister(ctfaData);
    }
   
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    cudaHostUnregister(sigRcpP);
#endif

    LOG(INFO) << "Insert done.";
    //printf("Insert done.\n");

    MPI_Barrier(hemi);
    //MPI_Barrier(MPI_COMM_WORLD);

    LOG(INFO) << "rank" << nranks << ": Step2: Reduce Volume.";
    //printf("rank%d: Step2: Reduce Volume.\n", nranks);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devDataF[i], 
                                (void*)devDataF[i], 
                                dimSize * 2, 
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                comm[i], 
                                stream[0 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(hemi);

    cudaSetDevice(gpus[0]);
    cudaMemcpy(F3D,
               devDataF[0],
               dimSize * sizeof(Complex),
               cudaMemcpyDeviceToHost);
    //cudaCheckErrors("copy F3D from device to host.");
        
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(hemi);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclAllReduce((const void*)devDataT[i], 
                                (void*)devDataT[i], 
                                dimSize, 
#ifdef SINGLE_PRECISION
                                ncclFloat,
#else
                                ncclDouble,
#endif
                                ncclSum,
                                comm[i], 
                                stream[1 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(hemi);
    
    cudaSetDevice(gpus[0]);
    cudaMemcpy(T3D,
               devDataT[0],
               dimSize * sizeof(RFLOAT),
               cudaMemcpyDeviceToHost);
    //cudaCheckErrors("copy T3D from device to host.");
    
    //MPI_Comm_rank(MPI_COMM_WORLD, &nranks);
    //if (nranks == 2){
        //cudaMemcpy(ctfP,
        //           devctfP[0],
        //           200 * npxl * sizeof(RFLOAT),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
    
        //Complex* tranP = new Complex[200 * npxl];
        //cudaMemcpy(tranP,
        //           devtranP[0],
        //           200 * npxl * sizeof(Complex),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
        //
        //double* matt = new double[200 * 9];
        //cudaMemcpy(matt,
        //           dev_mat_buf[0],
        //           200 * 9 * sizeof(RFLOAT),
        //           cudaMemcpyDeviceToHost);
        ////cudaCheckErrors("memcpy ctf to device.");
    
    
    //double* ctfc = new double[200 * npxl];
    //Complex* tranc = new Complex[200 * npxl];
    //double* matc = new double[200 * 9];

    //FILE* pfile;
    //pfile = fopen("ctfP.dat", "rb");
    //if (pfile == NULL)
    //    printf("open ctf error!\n");
    //if (fread(ctfc, sizeof(double), 200 * npxl, pfile) != 200 * npxl)
    //    printf("read ctf error!\n");
    //fclose(pfile);
    //printf("i:%d,ctfc:%.16lf,ctfg:%.16lf\n",0,ctfc[0],ctfP[0]);
    //int t = 0;
    //for (t = 0; t < 200 * npxl; t++){
    //    if (fabs(ctfP[t] - ctfc[t]) >= 1e-13){
    //        printf("i:%d,ctfc:%.16lf,ctfg:%.16lf\n",t,ctfc[t],ctfP[t]);
    //        break;
    //    }
    //}
    //if (t == 200 * npxl)
    //    printf("successCTF:%d\n", 200 * npxl);
    //
    //pfile = fopen("tranP.dat", "rb");
    //if (pfile == NULL)
    //    printf("open tran error!\n");
    //if (fread(tranc, sizeof(Complex), 200 * npxl, pfile) != 200 * npxl)
    //    printf("read tran error!\n");
    //fclose(pfile);
    //printf("i:%d,tranc:%.16lf,trang:%.16lf\n",0,tranc[0].imag(),tranP[0].imag());
    //for (t = 0; t < 200 * npxl; t++){
    //    if (fabs(tranP[t].imag() - tranc[t].imag()) >= 1e-12){
    //        printf("i:%d,tranc:%.16lf,trang:%.16lf\n",t,tranc[t].imag(),tranP[t].imag());
    //        break;
    //    }
    //}
    //if (t == 200 * npxl)
    //    printf("successTran:%d\n", 200 * 9);

    //pfile = fopen("mat.dat", "rb");
    //if (pfile == NULL)
    //    printf("open mat error!\n");
    //if (fread(matc, sizeof(double), 200 * 9, pfile) != 200 * 9)
    //    printf("read mat error!\n");
    //fclose(pfile);
    //printf("i:%d,matc:%.16lf,matg:%.16lf\n",0,matc[0],matt[0]);
    //for (t = 0; t < 200 * 9; t++){
    //    if (fabs(matt[t] - matc[t]) >= 1e-10){
    //        printf("i:%d,matc:%.16lf,matg:%.16lf\n",t,matc[t],matt[t]);
    //        break;
    //    }
    //}
    //if (t == 200 * 9)
    //    printf("successmat:%d\n", 200 * 9);

    //Complex* f3d = new Complex[dimSize];
    //Complex* t3d = new Complex[dimSize];

    //FILE* pfile;
    //pfile = fopen("f3d", "rb");
    //if (pfile == NULL)
    //    printf("open f3d error!\n");
    //if (fread(f3d, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read f3d error!\n");
    //fclose(pfile);
    //printf("i:%d,f3d:%.16lf,F3D:%.16lf\n",0,f3d[0].real(),F3D[0].real());
    //int t = 0;
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(F3D[t].real() - f3d[t].real()) >= 1e-9){
    //        printf("i:%d,f3d:%.16lf,F3D:%.16lf\n",t,f3d[t].real(),F3D[t].real());
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successF:%d\n", dimSize);

    //pfile = fopen("t3d", "rb");
    //if (pfile == NULL)
    //    printf("open t3d error!\n");
    //if (fread(t3d, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read t3d error!\n");
    //fclose(pfile);
    //printf("i:%d,t3d:%.16lf,T3D:%.16lf\n",0,t3d[0].real(),T3D[0]);
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(T3D[t] - t3d[t].real()) >= 1e-9){
    //        printf("i:%d,t3d:%.16lf,T3D:%.16lf\n",t,t3d[t].real(),T3D[t]);
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successT:%d\n", dimSize); 
    //}

    LOG(INFO) << "rank" << nranks << ":Step3: Copy done, free Volume and Nccl object.";
    //printf("rank%d:Step4: Copy done, free Volume and Nccl object.\n", nranks);
    
    cudaHostUnregister(F3D);
    cudaHostUnregister(T3D);
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(devDataF[n]); 
        cudaFree(devDataT[n]); 
        cudaFree(deviCol[n]); 
        cudaFree(deviRow[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamDestroy(stream[i + baseS]);
    } 
   
    //finalizing NCCL 
    for (int i = 0; i < aviDevs; i++) 
    { 
        ncclCommDestroy(comm[i]); 
    }
    
    delete[] gpus; 
}

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
               const int r)
{
    cudaSetDevice(gpuIdx);

    int batchSize = 8 * dim * (dim / 2 + 1);
    int dimSize = (dim / 2 + 1) * dim * dim;
    int symMatsize = nSymmetryElement * 9;

    cudaHostRegister(T3D, dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");
    
    cudaHostRegister(F3D, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");
    
    LOG(INFO) << "Step1: NormalizeT.";
    
    Complex *devDataF;
    cudaMalloc((void**)&devDataF, dimSize * sizeof(Complex));
    cudaCheckErrors("Allocate devDataF data.");

    RFLOAT *devPartT[3];
    for (int i = 0; i < 3; i++)
    {
        cudaMalloc((void**)&devPartT[i], batchSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devDataT data.");
    }

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    double *devSymmat;
    cudaMalloc((void**)&devSymmat, symMatsize * sizeof(double));
    cudaCheckErrors("Allocate devSymmat data.");
#endif    
    
    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);
    
    int batch, smidx = 0;
    for (int i = 0; i < dimSize;)
    {
        batch = (i + batchSize > dimSize) ? (dimSize - i) : batchSize;
        
        cudaMemcpyAsync(devDataF + i,
                        F3D + i,
                        batch * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

        cudaMemcpyAsync(devPartT[smidx],
                        T3D + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeTF<<<dim, 
                             dim / 2 + 1, 
                             0, 
                             stream[smidx]>>>(devDataF,
                                              devPartT[smidx], 
                                              batch, 
                                              i, 
                                              sf);
        cudaCheckErrors("normalTF.");
#endif
        
        cudaMemcpyAsync(T3D + i,
                        devPartT[smidx],
                        batch * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

        i += batch;
        smidx = (++smidx) % 3;
    }
   
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaMemcpyAsync(devSymmat, 
                    symMat, 
                    symMatsize * sizeof(double), 
                    cudaMemcpyHostToDevice, 
                    stream[2]);
    cudaCheckErrors("copy symmat for memcpy 0.");
#endif

    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");

    for (int i = 0; i < 3; i++)
    {
        cudaFree(devPartT[i]);
        cudaCheckErrors("Free device memory devDataT.");
    }
    
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);
    
    cudaTextureDesc td;
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;

#ifdef SINGLE_PRECISION
    cudaChannelFormatDesc channelDescF =cudaCreateChannelDesc(32, 32, 0, 0,cudaChannelFormatKindFloat);
#else
    cudaChannelFormatDesc channelDescF =cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindSigned);
#endif    
    cudaArray *symArrayF; 
    cudaMalloc3DArray(&symArrayF, &channelDescF, extent);

    cudaMemcpy3DParms copyParamsF = {0};
#ifdef SINGLE_PRECISION
    copyParamsF.srcPtr   = make_cudaPitchedPtr((void*)devDataF, (dim / 2 + 1) * sizeof(float2), dim / 2 + 1, dim);
#else
    copyParamsF.srcPtr   = make_cudaPitchedPtr((void*)devDataF, (dim / 2 + 1) * sizeof(int4), dim / 2 + 1, dim);
#endif    
    copyParamsF.dstArray = symArrayF;
    copyParamsF.extent   = extent;
    copyParamsF.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParamsF);
    cudaCheckErrors("memcpy error\n.");
    
    struct cudaResourceDesc resDescF;
    memset(&resDescF, 0, sizeof(resDescF));
    resDescF.resType = cudaResourceTypeArray;
    resDescF.res.array.array = symArrayF;

    cudaTextureObject_t texObjectF;
    cudaCreateTextureObject(&texObjectF, &resDescF, &td, NULL);

    LOG(INFO) << "Step2: SymmetrizeF.";
    
    smidx = 0;
    for (int i = 0; i < dimSize;)
    {
        batch = (i + batchSize > dimSize) ? (dimSize - i) : batchSize;

        kernel_SymmetrizeF<<<dim, 
                             dim / 2 + 1, 
                             0, 
                             stream[smidx]>>>(devDataF,
                                              devSymmat, 
                                              nSymmetryElement, 
                                              r, 
                                              interp, 
                                              i,
                                              dim,
                                              batch,
                                              texObjectF);
        cudaCheckErrors("symT for stream 0");

        cudaMemcpyAsync(F3D + i,
                        devDataF + i,
                        batch * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");
        
        i += batch;
        
        smidx = (++smidx) % 3;
    }
   
    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream2 synchronization.");

    cudaDestroyTextureObject(texObjectF);
    
    cudaFreeArray(symArrayF);
    cudaCheckErrors("Free device memory SymArrayF.");

    cudaFree(devDataF);
    cudaCheckErrors("Free device memory devDataF.");

#ifdef SINGLE_PRECISION
    cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
#else
    cudaChannelFormatDesc channelDescT = cudaCreateChannelDesc(32, 32, 0, 0,cudaChannelFormatKindSigned);
#endif    
    cudaArray *symArrayT; 
    cudaMalloc3DArray(&symArrayT, &channelDescT, extent);

    cudaMemcpy3DParms copyParamsT = {0};
#ifdef SINGLE_PRECISION
    copyParamsT.srcPtr   = make_cudaPitchedPtr((void*)T3D, (dim / 2 + 1) * sizeof(float), dim / 2 + 1, dim);
#else
    copyParamsT.srcPtr   = make_cudaPitchedPtr((void*)T3D, (dim / 2 + 1) * sizeof(int2), dim / 2 + 1, dim);
#endif    
    copyParamsT.dstArray = symArrayT;
    copyParamsT.extent   = extent;
    copyParamsT.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParamsT);
    cudaCheckErrors("memcpy error");
    
    struct cudaResourceDesc resDescT;
    memset(&resDescT, 0, sizeof(resDescT));
    resDescT.resType = cudaResourceTypeArray;
    resDescT.res.array.array = symArrayT;

    cudaTextureObject_t texObjectT;
    cudaCreateTextureObject(&texObjectT, &resDescT, &td, NULL);
    
    LOG(INFO) << "Step3: SymmetrizeT.";
    
    RFLOAT *devDataT;
    cudaMalloc((void**)&devDataT, dimSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDataF data.");

    smidx = 0;
    for (int i = 0; i < dimSize;)
    {
        batch = (i + batchSize > dimSize) ? (dimSize - i) : batchSize;

        cudaMemcpyAsync(devDataT + i,
                        T3D + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");
        
        kernel_SymmetrizeT<<<dim, 
                             dim / 2 + 1, 
                             0, 
                             stream[smidx]>>>(devDataT,
                                              devSymmat, 
                                              nSymmetryElement, 
                                              r, 
                                              interp, 
                                              i,
                                              dim,
                                              batch,
                                              texObjectT);
        cudaCheckErrors("symT for stream 0");

        cudaMemcpyAsync(T3D + i,
                        devDataT + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");
        
        i += batch;
        
        smidx = (++smidx) % 3;
    }
   
    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream2 synchronization.");

    cudaDestroyTextureObject(texObjectT);
    
    cudaFreeArray(symArrayT);
    cudaCheckErrors("Free device memory SymArrayT.");

#endif

    cudaHostUnregister(F3D);
    cudaHostUnregister(T3D);
    
    //Complex* dd = new Complex[dimSize];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("f3d.dat", "rb");
    //if (pfile == NULL)
    //    printf("open f3d error!\n");
    //if (fread(dd, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read f3d error!\n");
    //fclose(pfile);
    //printf("i:%d,cdst:%.16lf,gdst:%.16lf\n",823032,dd[823032].real(),F3D[823032].real());
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(F3D[t].real() - dd[t].real()) >= 1e-9){
    //        printf("i:%d,cdst:%.16lf,gdst:%.16lf\n",t,dd[t].real(),F3D[t].real());
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successT3D:%d\n", dimSize);

    //Complex* dd = new Complex[dimSize];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("t3d.dat", "rb");
    //if (pfile == NULL)
    //    printf("open t3d error!\n");
    //if (fread(dd, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read t3d error!\n");
    //fclose(pfile);
    //printf("i:%d,cdst:%.16lf,gdst:%.16lf\n",0,dd[0].real(),T3D[0]);
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(T3D[t] - dd[t].real()) >= 1e-10){
    //        printf("i:%d,cdst:%.16lf,gdst:%.16lf\n",t,dd[t].real(),T3D[t]);
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successT3D:%d\n", dimSize);

    LOG(INFO) << "Step6: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaFree(devSymmat);
    cudaCheckErrors("Free device memory devSymmat.");
    
#endif    
    
    cudaFree(devDataT);
    cudaCheckErrors("Free device memory devDataT.");
}

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
                  const int pf)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim;
    int vecSize = maxRadius * pf + 1;

    RFLOAT *devDataT;
    cudaMalloc((void**)&devDataT, dimSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDataT data.");

    RFLOAT *devAvg;
    cudaMalloc((void**)&devAvg, vecSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devAvg data.");
    
    cudaMemcpy(devDataT, T2D, dimSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("out for memcpy 0.");
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    LOG(INFO) << "Step1: ShellAverage.";
    
    RFLOAT *devAvg2D;
    int *devCount2D;
    int *devCount;
    cudaMalloc((void**)&devAvg2D, (vecSize - 2) * dim * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devAvg data.");

    cudaMalloc((void**)&devCount2D, (vecSize - 2) * dim * sizeof(int));
    cudaCheckErrors("Allocate devCount data.");

    cudaMalloc((void**)&devCount, vecSize * sizeof(int));
    cudaCheckErrors("Allocate devCount data.");
    
    kernel_ShellAverage2D<<<dim, 
                            dim / 2 + 1,
                            (vecSize - 2) 
                             * (sizeof(RFLOAT) 
                                + sizeof(int))>>>(devAvg2D, 
                                                  devCount2D, 
                                                  devDataT,
                                                  dim, 
                                                  vecSize - 2);
    cudaCheckErrors("Shell for stream default.");
    
    kernel_CalculateAvg2D<<<1, vecSize - 2>>>(devAvg2D,
                                              devCount2D,
                                              devAvg,
                                              devCount,
                                              dim,
                                              vecSize - 2);
    cudaCheckErrors("calAvg for stream default.");
#endif

    LOG(INFO) << "Step2: Calculate WIENER_FILTER.";
    
    RFLOAT *devFSC;
    cudaMalloc((void**)&devFSC, fscMatsize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devFSC data.");
    cudaMemcpy(devFSC, FSC, fscMatsize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy FSC to device.");

    int wiener = pow(wienerF * pf, 2);
    int r = pow(maxRadius * pf, 2);
    
    kernel_CalculateFSC2D<<<dim, 
                            dim / 2 + 1>>>(devDataT, 
                                           devFSC, 
                                           devAvg,
                                           joinHalf,
                                           fscMatsize,
                                           wiener,
                                           dim,
                                           pf,
                                           r);
    cudaCheckErrors("calFSC for stream 0.");

    cudaMemcpy(T2D,
               devDataT,
               dimSize * sizeof(RFLOAT),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("out for memcpy 0.");
    
    LOG(INFO) << "Step6: Clean up the streams and memory.";

    cudaFree(devAvg);
    cudaCheckErrors("Free device memory devAvg.");

    cudaFree(devFSC);
    cudaCheckErrors("Free device memory devFSC.");
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    cudaFree(devAvg2D);
    cudaCheckErrors("Free device memory devAvg2D.");

    cudaFree(devCount2D);
    cudaCheckErrors("Free device memory devCount2D.");
    
    cudaFree(devCount);
    cudaCheckErrors("Free device memory devCount.");
#endif
    cudaFree(devDataT);
    cudaCheckErrors("Free device memory devDataT.");
}

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
                const int pf)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim * dim;
    int vecSize = maxRadius * pf + 1;

    cudaHostRegister(T3D, dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");
    
    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);
    
    RFLOAT *devDataT;
    cudaMalloc((void**)&devDataT, dimSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDataT data.");

    RFLOAT *devAvg;
    cudaMalloc((void**)&devAvg, vecSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devAvg data.");
    
    cudaMemcpy(devDataT, T3D, dimSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("out for memcpy 0.");
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    LOG(INFO) << "Step1: ShellAverage.";
    
    RFLOAT *devAvg2D;
    int *devCount2D;
    int *devCount;
    cudaMalloc((void**)&devAvg2D, (vecSize - 2) * dim * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devAvg data.");

    cudaMalloc((void**)&devCount2D, (vecSize - 2) * dim * sizeof(int));
    cudaCheckErrors("Allocate devCount data.");

    cudaMalloc((void**)&devCount, vecSize * sizeof(int));
    cudaCheckErrors("Allocate devCount data.");
    
    kernel_ShellAverage<<<dim, 
                          dim, 
                          dim * (sizeof(RFLOAT) 
                                + sizeof(int))>>>(devAvg2D, 
                                                  devCount2D, 
                                                  devDataT,
                                                  dim, 
                                                  vecSize - 2,
                                                  dimSize);
    cudaCheckErrors("Shell for stream default.");
    
    kernel_CalculateAvg<<<1, vecSize - 2>>>(devAvg2D,
                                            devCount2D,
                                            devAvg,
                                            devCount,
                                            dim,
                                            vecSize - 2);
    cudaCheckErrors("calAvg for stream default.");
#endif

    LOG(INFO) << "Step2: Calculate WIENER_FILTER.";
    
    RFLOAT *devFSC;
    cudaMalloc((void**)&devFSC, fscMatsize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devFSC data.");
    cudaMemcpy(devFSC, FSC, fscMatsize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy FSC to device.");

    int wiener = pow(wienerF * pf, 2);
    int r = pow(maxRadius * pf, 2);
    
    int batchSize = 8 * dim * dim;
    int len = dimSize / batchSize;
    int streamfsc = len / 3;
    
    for (int i = 0; i < streamfsc; i++)
    {
        int shift = i * 3 * batchSize;
        kernel_CalculateFSC<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                        devFSC, 
                                                        devAvg,
                                                        fscMatsize,
                                                        joinHalf,
                                                        wiener,
                                                        r, 
                                                        shift,
                                                        dim,
                                                        batchSize);
        cudaCheckErrors("calFSC for stream 0.");

        cudaMemcpyAsync(T3D + shift,
                        devDataT + shift,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        cudaCheckErrors("out for memcpy 0.");
    
        kernel_CalculateFSC<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                        devFSC, 
                                                        devAvg,
                                                        fscMatsize,
                                                        joinHalf,
                                                        wiener,
                                                        r, 
                                                        shift + batchSize,
                                                        dim,
                                                        batchSize);
        cudaCheckErrors("calFSC for stream 1.");

        cudaMemcpyAsync(T3D + shift + batchSize,
                        devDataT + shift + batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        cudaCheckErrors("out for memcpy 1.");

        kernel_CalculateFSC<<<dim, dim, 0, stream[2]>>>(devDataT, 
                                                        devFSC, 
                                                        devAvg,
                                                        fscMatsize,
                                                        joinHalf,
                                                        wiener,
                                                        r, 
                                                        shift + 2 * batchSize,
                                                        dim,
                                                        batchSize);
        cudaCheckErrors("calFSC for stream 2.");

        cudaMemcpyAsync(T3D + shift + 2 * batchSize,
                        devDataT + shift + 2 * batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[2]);
        cudaCheckErrors("out for memcpy 2.");
    }
   
    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
        kernel_CalculateFSC<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                        devFSC, 
                                                        devAvg,
                                                        fscMatsize,
                                                        joinHalf,
                                                        wiener,
                                                        r, 
                                                        shift,
                                                        dim,
                                                        batchSize);
        cudaCheckErrors("calFSC last for stream 0.");

        cudaMemcpyAsync(T3D + shift,
                        devDataT + shift,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        cudaCheckErrors("out for memcpy 0.");

        kernel_CalculateFSC<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                        devFSC, 
                                                        devAvg,
                                                        fscMatsize,
                                                        joinHalf,
                                                        wiener,
                                                        r, 
                                                        shift + batchSize,
                                                        dim,
                                                        batchSize);
        cudaCheckErrors("calFSC last for stream 1.");
        
        cudaMemcpyAsync(T3D + shift + batchSize,
                        devDataT + shift + batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        cudaCheckErrors("out for memcpy 1.");
        
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
            kernel_CalculateFSC<<<dim, dim, 0, stream[2]>>>(devDataT, 
                                                            devFSC, 
                                                            devAvg,
                                                            fscMatsize,
                                                            joinHalf,
                                                            wiener,
                                                            r, 
                                                            shift,
                                                            dim,
                                                            dimSize - shift);
            cudaCheckErrors("calFSC last for stream 2.");
        
            cudaMemcpyAsync(T3D + shift,
                            devDataT + shift,
                            (dimSize - shift) * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[2]);
            cudaCheckErrors("out for memcpy 2.");
        }    

    }
    
    else
    {
        if (len % 3 == 1)
        {
            int shift = (len - 1) * batchSize;
            kernel_CalculateFSC<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                            devFSC, 
                                                            devAvg,
                                                            fscMatsize,
                                                            joinHalf,
                                                            wiener,
                                                            r, 
                                                            shift,
                                                            dim,
                                                            batchSize);
            cudaCheckErrors("calFSC last for stream 0.");
        
            cudaMemcpyAsync(T3D + shift,
                            devDataT + shift,
                            batchSize * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[0]);
            cudaCheckErrors("out for memcpy 0.");
            
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                kernel_CalculateFSC<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                                devFSC, 
                                                                devAvg,
                                                                fscMatsize,
                                                                joinHalf,
                                                                wiener,
                                                                r, 
                                                                shift,
                                                                dim,
                                                                dimSize - shift);
                cudaCheckErrors("calFSC last for stream 1.");

                cudaMemcpyAsync(T3D + shift,
                                devDataT + shift,
                                (dimSize - shift) * sizeof(RFLOAT),
                                cudaMemcpyDeviceToHost,
                                stream[1]);
                cudaCheckErrors("out for memcpy 1.");
            }    

        }
        else
        {
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                kernel_CalculateFSC<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                                devFSC, 
                                                                devAvg,
                                                                fscMatsize,
                                                                joinHalf,
                                                                wiener,
                                                                r, 
                                                                shift,
                                                                dim,
                                                                dimSize - shift);
                cudaCheckErrors("calFSC last for stream 0.");

                cudaMemcpyAsync(T3D + shift,
                                devDataT + shift,
                                (dimSize - shift) * sizeof(RFLOAT),
                                cudaMemcpyDeviceToHost,
                                stream[0]);
                cudaCheckErrors("out for memcpy 0.");
            }    


        }
    } 
    
    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream2 synchronization.");
    
    cudaHostUnregister(T3D);
    
    LOG(INFO) << "Step6: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

    cudaFree(devAvg);
    cudaCheckErrors("Free device memory devAvg.");

    cudaFree(devFSC);
    cudaCheckErrors("Free device memory devFSC.");
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    cudaFree(devAvg2D);
    cudaCheckErrors("Free device memory devAvg2D.");

    cudaFree(devCount2D);
    cudaCheckErrors("Free device memory devCount2D.");
    
    cudaFree(devCount);
    cudaCheckErrors("Free device memory devCount.");
#endif
    cudaFree(devDataT);
    cudaCheckErrors("Free device memory devDataT.");
}

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
                  const int padSize)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim;
    int dimSizeRL = dim * dim;
    
    LOG(INFO) << "Step1: InitialW.";
    
    RFLOAT *devDataW;
    cudaMalloc((void**)&devDataW, dimSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDataW data.");
    
    kernel_InitialW2D<<<dim, 
                        dim / 2 + 1>>>(devDataW,  
                                       r, 
                                       dim);
    
    
    LOG(INFO) << "Step2: Calculate C.";

    /* Upload tabfunction to device */
    TabFunction tabfunc(begin, end, step, NULL, tabsize);

    uploadTabFunction(tabfunc, tabdata);

    cufftDoubleReal *diffc;
    cudaMalloc((void**)&diffc, dimSize * sizeof(cufftDoubleReal));
    cudaCheckErrors("Allocate device memory for C.");

#ifdef SINGLE_PRECISION
    cufftComplex *devDataC;
    cudaMalloc((void**)&devDataC, dimSize * sizeof(cufftComplex));
    cudaCheckErrors("Allocate device memory for C.");

    cufftReal *devDoubleC;
    cudaMalloc((void**)&devDoubleC, dimSizeRL * sizeof(cufftReal));
    cudaCheckErrors("Allocate device memory for C.");
#else
    cufftDoubleComplex *devDataC;
    cudaMalloc((void**)&devDataC, dimSize * sizeof(cufftDoubleComplex));
    cudaCheckErrors("Allocate device memory for C.");

    cufftDoubleReal *devDoubleC;
    cudaMalloc((void**)&devDoubleC, dimSizeRL * sizeof(cufftDoubleReal));
    cudaCheckErrors("Allocate device memory for C.");
#endif    
    
    RFLOAT *devDataT;
    cudaMalloc((void**)&devDataT, dimSize *sizeof(RFLOAT));
    cudaCheckErrors("Allocate device memory for T.");
    cudaMemcpy(devDataT, T2D, dimSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devDataT volume to device.");

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
    RFLOAT *diff = new RFLOAT[dim];
    int *counter = new int[dim];
    
    RFLOAT *devDiff;
    int *devCount;
    cudaMalloc((void**)&devDiff, dim *sizeof(RFLOAT));
    cudaCheckErrors("Allocate device memory for devDiff.");
    cudaMalloc((void**)&devCount, dim *sizeof(int));
    cudaCheckErrors("Allocate device memory for devCount.");
        
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX        
    RFLOAT *cmax = new RFLOAT[dim];
    
    RFLOAT *devMax;
    cudaMalloc((void**)&devMax, dim *sizeof(RFLOAT));
    cudaCheckErrors("Allocate device memory for devMax.");
#endif
    
    RFLOAT diffC;
    RFLOAT diffCPrev;
    cufftHandle planc2r, planr2c;
#ifdef SINGLE_PRECISION
    diffC = FLT_MAX;
    diffCPrev = FLT_MAX;
    cufftPlan2d(&planc2r, dim, dim, CUFFT_C2R);
    cufftPlan2d(&planr2c, dim, dim, CUFFT_R2C);
#else
    diffC = DBL_MAX;
    diffCPrev = DBL_MAX;
    cufftPlan2d(&planc2r, dim, dim, CUFFT_Z2D);
    cufftPlan2d(&planr2c, dim, dim, CUFFT_D2Z);
#endif    
    
    int nDiffCNoDecrease = 0;

    for(int m = 0; m < maxIter; m++)
    {
        //LOG(INFO) << "SubStep1: Determining C.";
        kernel_DeterminingC2D<<<dim, 
                                dim / 2 + 1>>>((Complex*)devDataC,
                                               devDataT, 
                                               devDataW); 

        //LOG(INFO) << "SubStep2: Convoluting C.";

#ifdef SINGLE_PRECISION
        cufftExecC2R(planc2r, devDataC, devDoubleC);

        kernel_convoluteC2D<<<dim, 
                              dim>>>((RFLOAT*)devDoubleC,
                                     tabfunc,
                                     nf,
                                     padSize,
                                     dim,
                                     dimSizeRL);
        
        cufftExecR2C(planr2c, devDoubleC, devDataC);
#else
        cufftExecZ2D(planc2r, devDataC, devDoubleC);

        kernel_convoluteC2D<<<dim, 
                              dim>>>((RFLOAT*)devDoubleC,
                                     tabfunc,
                                     nf,
                                     padSize,
                                     dim,
                                     dimSizeRL);
        
        cufftExecD2Z(planr2c, devDoubleC, devDataC);
#endif    
    
        kernel_RecalculateW2D<<<dim, 
                                dim / 2 + 1>>>(devDataW,
                                               (Complex*)devDataC,
                                               r, 
                                               dim);

        diffCPrev = diffC;
        
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
        kernel_CheckCAVG2D<<<dim, 
                             dim / 2 + 1, 
                             (dim / 2 + 1) * (sizeof(RFLOAT) 
                                           + sizeof(int))>>>(devDiff,
                                                             devCount,
                                                             (Complex*)devDataC,  
                                                             r, 
                                                             dim);
        
        cudaMemcpy(diff, devDiff, dim *sizeof(RFLOAT), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy devDiff array to host.");
        cudaMemcpy(counter, devCount, dim *sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy devCount array to host.");
        
        RFLOAT tempD = 0; 
        int tempC = 0;
        for(int i = 0;i < dim;i++)
        {
            tempD += diff[i];
            tempC += counter[i];
        }
        diffC = tempD / tempC;

#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
        kernel_CheckCMAX2D<<<dim, 
                             dim / 2 + 1, 
                             (dim / 2 + 1) 
                              * sizeof(RFLOAT)>>>(devMax,
                                                  (Complex*)devDataC,  
                                                  r, 
                                                  dim);
        
        cudaMemcpy(cmax, devMax, dim * sizeof(RFLOAT), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy devMax array to host.");
        
        RFLOAT temp = 0.0;
        for(int i = 0;i < dim;i++)
        {
            if (temp <= cmax[i])
                temp = cmax[i];
        }
        diffC = temp;
#endif

        if (diffC > diffCPrev * DIFF_C_DECREASE_THRES)
            nDiffCNoDecrease += 1;
        else
            nDiffCNoDecrease = 0;

        if ((diffC < DIFF_C_THRES) ||
            ((m >= minIter) &&
            (nDiffCNoDecrease == N_DIFF_C_NO_DECREASE))) break;

    }

    cudaMemcpy(W2D, devDataW, dimSize * sizeof(RFLOAT), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Copy devDataW volume to host.");  
    
    LOG(INFO) << "Step3: Clean up the streams and memory.";

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
    delete[] diff;
    delete[] counter;

    cudaFree(devDiff);
    cudaCheckErrors("Free device memory devDiff.");
    
    cudaFree(devCount);
    cudaCheckErrors("Free device memory devCount.");
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX        
    delete[] cmax;

    cudaFree(devMax);
    cudaCheckErrors("Free device memory devMax.");
#endif

    cudaFree(devDataW);
    cudaCheckErrors("Free device memory devDataW.");
    
    cudaFree(devDataC);
    cudaCheckErrors("Free device memory devDataC.");
    
    cudaFree(devDoubleC);
    cudaCheckErrors("Free device memory devRFLOATC.");

    cufftDestroy(planc2r);
    cudaCheckErrors("DestroyPlan planc2r.");

    cufftDestroy(planr2c);
    cudaCheckErrors("DestroyPlan planr2c.");
    
    cudaFree(devDataT);
    cudaCheckErrors("Free device memory devDataT.");
    
    cudaFree(tabfunc.devPtr());
    cudaCheckErrors("Free operations.");
}

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
                const int padSize)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim * dim;
    int dimSizeRL = dim * dim * dim;
    
    LOG(INFO) << "Step1: InitialW.";
    
    RFLOAT *devDataW;
    cudaMalloc((void**)&devDataW, dimSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDataW data.");
    
    kernel_InitialW<<<dim, dim>>>(devDataW,  
                                  r, 
                                  dim,
                                  dimSize);
    
    
    LOG(INFO) << "Step2: Calculate C.";

    /* Upload tabfunction to device */
    TabFunction tabfunc(begin, end, step, NULL, tabsize);

    uploadTabFunction(tabfunc, tabdata);

#ifdef SINGLE_PRECISION
    cufftComplex *devDataC;
    cudaMalloc((void**)&devDataC, dimSize * sizeof(cufftComplex));
    cudaCheckErrors("Allocate device memory for C.");

    cufftReal *devDoubleC;
    cudaMalloc((void**)&devDoubleC, dimSizeRL * sizeof(cufftReal));
    cudaCheckErrors("Allocate device memory for C.");
#else
    cufftDoubleComplex *devDataC;
    cudaMalloc((void**)&devDataC, dimSize * sizeof(cufftDoubleComplex));
    cudaCheckErrors("Allocate device memory for C.");

    cufftDoubleReal *devDoubleC;
    cudaMalloc((void**)&devDoubleC, dimSizeRL * sizeof(cufftDoubleReal));
    cudaCheckErrors("Allocate device memory for C.");
#endif    
    
    RFLOAT *devDataT;
    cudaMalloc((void**)&devDataT, dimSize *sizeof(RFLOAT));
    cudaCheckErrors("Allocate device memory for T.");
    cudaMemcpy(devDataT, T3D, dimSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devDataT volume to device.");

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
    RFLOAT *diff = new RFLOAT[dim];
    int *counter = new int[dim];
    
    RFLOAT *devDiff;
    int *devCount;
    cudaMalloc((void**)&devDiff, dim *sizeof(RFLOAT));
    cudaCheckErrors("Allocate device memory for devDiff.");
    cudaMalloc((void**)&devCount, dim *sizeof(int));
    cudaCheckErrors("Allocate device memory for devCount.");
        
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX        
    RFLOAT *cmax = new RFLOAT[dim];
    
    RFLOAT *devMax;
    cudaMalloc((void**)&devMax, dim *sizeof(RFLOAT));
    cudaCheckErrors("Allocate device memory for devMax.");
#endif
    
    RFLOAT diffC;
    RFLOAT diffCPrev;
    cufftHandle planc2r, planr2c;
#ifdef SINGLE_PRECISION
    diffC = FLT_MAX;
    diffCPrev = FLT_MAX;
    cufftPlan3d(&planc2r, dim, dim, dim, CUFFT_C2R);
    cufftPlan3d(&planr2c, dim, dim, dim, CUFFT_R2C);
#else
    diffC = DBL_MAX;
    diffCPrev = DBL_MAX;
    cufftPlan3d(&planc2r, dim, dim, dim, CUFFT_Z2D);
    cufftPlan3d(&planr2c, dim, dim, dim, CUFFT_D2Z);
#endif    
    
    int nDiffCNoDecrease = 0;

    for(int m = 0; m < maxIter; m++)
    {
        //LOG(INFO) << "SubStep1: Determining C.";
        kernel_DeterminingC<<<dim, dim>>>((Complex*)devDataC,
                                          devDataT, 
                                          devDataW, 
                                          dimSize);

        //LOG(INFO) << "SubStep2: Convoluting C.";

    //cudaMemcpy(C3D, devDataC, dimSize * sizeof(Complex), cudaMemcpyDeviceToHost);
    //cudaCheckErrors("Copy devDataW volume to host.");  
    //
    //Complex* dd = new Complex[dimSize];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("c3d.dat", "rb");
    //if (pfile == NULL)
    //    printf("open c3d error!\n");
    //if (fread(dd, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read c3d error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%.7lf,gw:%.7lf\n",0,dd[0].real(),C3D[0].real());
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(C3D[t].real() - dd[t].real()) >= 1e-4){
    //        printf("i:%d,cw:%.7lf,gw:%.7lf\n",t,dd[t].real(),C3D[t].real());
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successw:%d\n", dimSize);

#ifdef SINGLE_PRECISION
        cufftExecC2R(planc2r, devDataC, devDoubleC);

        kernel_convoluteC<<<dim, dim>>>((RFLOAT*)devDoubleC,
                                        tabfunc,
                                        nf,
                                        padSize,
                                        dim,
                                        dimSizeRL);
        
    //float* c3d = new float[dimSizeRL];
    //cudaMemcpy(c3d, devDoubleC, dimSizeRL * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaCheckErrors("Copy devDataW volume to host.");  
    //
    //float* dd = new float[dimSizeRL];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("c3d.dat", "rb");
    //if (pfile == NULL)
    //    printf("open c3d error!\n");
    //if (fread(dd, sizeof(float), dimSizeRL, pfile) != dimSizeRL)
    //    printf("read c3d error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%.7lf,gw:%.7lf\n",0,dd[0],c3d[0]);
    //for (t = 0; t < dimSizeRL; t++){
    //    if (fabs(c3d[t] - dd[t]) >= 1e-5){
    //        printf("i:%d,cw:%.7lf,gw:%.7lf\n",t,dd[t],c3d[t]);
    //        break;
    //    }
    //}
    //if (t == dimSizeRL)
    //    printf("successC:%d\n", dimSizeRL);

        cufftExecR2C(planr2c, devDoubleC, devDataC);
#else
        cufftExecZ2D(planc2r, devDataC, devDoubleC);

        kernel_convoluteC<<<dim, dim>>>((RFLOAT*)devDoubleC,
                                        tabfunc,
                                        nf,
                                        padSize,
                                        dim,
                                        dimSizeRL);
        
        cufftExecD2Z(planr2c, devDoubleC, devDataC);
#endif    
    
        kernel_RecalculateW<<<dim, dim>>>(devDataW,
                                          (Complex*)devDataC,  
                                          r, 
                                          dim,
                                          dimSize);

        diffCPrev = diffC;
        
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
        kernel_CheckCAVG<<<dim, 
                           dim, 
                           dim * (sizeof(RFLOAT) 
                                  + sizeof(int))>>>(devDiff,
                                                    devCount,
                                                    (Complex*)devDataC,  
                                                    r, 
                                                    dim,
                                                    dimSize);
        
        cudaMemcpy(diff, devDiff, dim *sizeof(RFLOAT), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy devDiff array to host.");
        cudaMemcpy(counter, devCount, dim *sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy devCount array to host.");
        
        RFLOAT tempD = 0; 
        int tempC = 0;
        for(int i = 0;i < dim;i++)
        {
            tempD += diff[i];
            tempC += counter[i];
        }
        diffC = tempD / tempC;

#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
        kernel_CheckCMAX<<<dim, 
                           dim, 
                           dim * sizeof(RFLOAT)>>>(devMax,
                                                   (Complex*)devDataC,  
                                                   r, 
                                                   dim,
                                                   dimSize);
        
        cudaMemcpy(cmax, devMax, dim * sizeof(RFLOAT), cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy devMax array to host.");
        
        RFLOAT temp = 0.0;
        for(int i = 0;i < dim;i++)
        {
            if (temp <= cmax[i])
                temp = cmax[i];
        }
        diffC = temp;
#endif

        //printf("diffC:%.16lf\n", diffC);

        if (diffC > diffCPrev * DIFF_C_DECREASE_THRES)
            nDiffCNoDecrease += 1;
        else
            nDiffCNoDecrease = 0;

        if ((diffC < DIFF_C_THRES) ||
            ((m >= minIter) &&
            (nDiffCNoDecrease == N_DIFF_C_NO_DECREASE))) break;

    }

    cudaMemcpy(W3D, devDataW, dimSize * sizeof(RFLOAT), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Copy devDataW volume to host.");  
    
    //Complex* dd = new Complex[dimSize];
    //FILE* pfile;
    //int t = 0;
    //pfile = fopen("w3d.dat", "rb");
    //if (pfile == NULL)
    //    printf("open w3d error!\n");
    //if (fread(dd, sizeof(Complex), dimSize, pfile) != dimSize)
    //    printf("read w3d error!\n");
    //fclose(pfile);
    //printf("i:%d,cw:%.16lf,gw:%.16lf\n",0,dd[0].real(),W3D[0]);
    //for (t = 0; t < dimSize; t++){
    //    if (fabs(W3D[t] - dd[t].real()) >= 1e-10){
    //        printf("i:%d,cw:%.16lf,gw:%.16lf\n",t,dd[t].real(),W3D[t]);
    //        break;
    //    }
    //}
    //if (t == dimSize)
    //    printf("successw:%d\n", dimSize);

    LOG(INFO) << "Step3: Clean up the streams and memory.";

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
    delete[] diff;
    delete[] counter;

    cudaFree(devDiff);
    cudaCheckErrors("Free device memory devDiff.");
    
    cudaFree(devCount);
    cudaCheckErrors("Free device memory devCount.");
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX        
    delete[] cmax;

    cudaFree(devMax);
    cudaCheckErrors("Free device memory devMax.");
#endif

    cudaFree(devDataW);
    cudaCheckErrors("Free device memory devDataW.");
    
    cudaFree(devDataC);
    cudaCheckErrors("Free device memory devDataC.");
    
    cudaFree(devDoubleC);
    cudaCheckErrors("Free device memory devRFLOATC.");

    cufftDestroy(planc2r);
    cudaCheckErrors("DestroyPlan planc2r.");

    cufftDestroy(planr2c);
    cudaCheckErrors("DestroyPlan planr2c.");
    
    cudaFree(devDataT);
    cudaCheckErrors("Free device memory devDataT.");
    
    cudaFree(tabfunc.devPtr());
    cudaCheckErrors("Free operations.");
}

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
                  const int r)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim;
    
    RFLOAT *devDataW;
    cudaMalloc((void**)&devDataW, dimSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDataW data.");
    
    RFLOAT *devDataT;
    cudaMalloc((void**)&devDataT, dimSize *sizeof(RFLOAT));
    cudaCheckErrors("Allocate device memory for T.");

    LOG(INFO) << "Step1: CalculateW.";
    
    cudaMemcpy(devDataT, 
               T2D, 
               dimSize * sizeof(RFLOAT), 
               cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devDataT volume to device stream0.");
    
    kernel_CalculateW2D<<<dim, 
                          dim / 2 + 1>>>(devDataW,
                                         devDataT, 
                                         dim,
                                         r);

    cudaMemcpy(W2D, 
               devDataW, 
               dimSize * sizeof(RFLOAT), 
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("Copy devDataW volume to host stream0.");  
    
    LOG(INFO) << "Step3: Clean up the streams and memory.";

    cudaFree(devDataW);
    cudaCheckErrors("Free device memory devDataW.");
    
    cudaFree(devDataT);
    cudaCheckErrors("Free device memory devDataT.");
}

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
                const int r)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim * dim;
    
    cudaHostRegister(T3D, dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register T3D data.");

    cudaHostRegister(W3D, dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register W3D data.");
    
    RFLOAT *devDataW;
    cudaMalloc((void**)&devDataW, dimSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDataW data.");
    
    RFLOAT *devDataT;
    cudaMalloc((void**)&devDataT, dimSize *sizeof(RFLOAT));
    cudaCheckErrors("Allocate device memory for T.");

    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);
    
    LOG(INFO) << "Step1: CalculateW.";
    
    int batchSize = 8 * dim * dim;
    int len = dimSize / batchSize;
    int streamN = len / 3;
   
    for (int i = 0; i < streamN; i++)
    {
        int shift = i * 3 * batchSize;
        cudaMemcpyAsync(devDataT + shift, 
                        T3D + shift, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyHostToDevice,
                        stream[0]);
        cudaCheckErrors("Copy devDataT volume to device stream0.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[0]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift,
                                                      dim,
                                                      r);

        cudaMemcpyAsync(W3D + shift, 
                        devDataW + shift, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        cudaCheckErrors("Copy devDataW volume to host stream0.");  
    
        cudaMemcpyAsync(devDataT + shift + batchSize, 
                        T3D + shift + batchSize, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyHostToDevice,
                        stream[1]);
        cudaCheckErrors("Copy devDataT volume to device stream1.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[1]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift + batchSize,
                                                      dim,
                                                      r);

        cudaMemcpyAsync(W3D + shift + batchSize, 
                        devDataW + shift + batchSize, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        cudaCheckErrors("Copy devDataW volume to host stream1.");  
    
        cudaMemcpyAsync(devDataT + shift + 2 * batchSize, 
                        T3D + shift + 2 * batchSize, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyHostToDevice,
                        stream[2]);
        cudaCheckErrors("Copy devDataT volume to device stream2.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[2]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift + 2 * batchSize,
                                                      dim,
                                                      r);

        cudaMemcpyAsync(W3D + shift + 2 * batchSize, 
                        devDataW + shift + 2 * batchSize, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyDeviceToHost,
                        stream[2]);
        cudaCheckErrors("Copy devDataW volume to host stream2.");  
    
    }

    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
        cudaMemcpyAsync(devDataT + shift, 
                        T3D + shift, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyHostToDevice,
                        stream[0]);
        cudaCheckErrors("Copy devDataT volume to device stream0.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[0]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift,
                                                      dim,
                                                      r);

        cudaMemcpyAsync(W3D + shift, 
                        devDataW + shift, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        cudaCheckErrors("Copy devDataW volume to host stream0.");  
        
        cudaMemcpyAsync(devDataT + shift + batchSize, 
                        T3D + shift + batchSize, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyHostToDevice,
                        stream[1]);
        cudaCheckErrors("Copy devDataT volume to device stream1.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[1]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift + batchSize,
                                                      dim,
                                                      r);


        cudaMemcpyAsync(W3D + shift + batchSize, 
                        devDataW + shift + batchSize, 
                        batchSize * sizeof(RFLOAT), 
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        cudaCheckErrors("Copy devDataW volume to host stream1.");  
    
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
            cudaMemcpyAsync(devDataT + shift, 
                            T3D + shift, 
                            (dimSize - shift) * sizeof(RFLOAT), 
                            cudaMemcpyHostToDevice,
                            stream[2]);
            cudaCheckErrors("Copy devDataT volume to device stream2.");
            
            kernel_CalculateW<<<dim, dim, 0, stream[2]>>>(devDataW,
                                                          devDataT, 
                                                          dimSize - shift, 
                                                          shift,
                                                          dim,
                                                          r);

            cudaMemcpyAsync(W3D + shift, 
                            devDataW + shift, 
                            (dimSize - shift) * sizeof(RFLOAT), 
                            cudaMemcpyDeviceToHost,
                            stream[2]);
            cudaCheckErrors("Copy devDataW volume to host stream2.");
        }
    }
    else
    {
        
        if (len % 3 == 1)
        {
            int shift = (len - 1) * batchSize;
            cudaMemcpyAsync(devDataT + shift, 
                            T3D + shift, 
                            batchSize * sizeof(RFLOAT), 
                            cudaMemcpyHostToDevice,
                            stream[0]);
            cudaCheckErrors("Copy devDataT volume to device stream0.");
            
            kernel_CalculateW<<<dim, dim, 0, stream[0]>>>(devDataW,
                                                          devDataT, 
                                                          batchSize, 
                                                          shift,
                                                          dim,
                                                          r);

            cudaMemcpyAsync(W3D + shift, 
                            devDataW + shift, 
                            batchSize * sizeof(RFLOAT), 
                            cudaMemcpyDeviceToHost,
                            stream[0]);
            cudaCheckErrors("Copy devDataW volume to host stream0.");  
        
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDataT + shift, 
                                T3D + shift, 
                                (dimSize - shift) * sizeof(RFLOAT), 
                                cudaMemcpyHostToDevice,
                                stream[1]);
                cudaCheckErrors("Copy devDataT volume to device stream1.");
                
                kernel_CalculateW<<<dim, dim, 0, stream[1]>>>(devDataW,
                                                              devDataT, 
                                                              dimSize - shift, 
                                                              shift,
                                                              dim,
                                                              r);
                cudaMemcpyAsync(W3D + shift, 
                                devDataW + shift, 
                                (dimSize - shift) * sizeof(RFLOAT), 
                                cudaMemcpyDeviceToHost,
                                stream[1]);
                cudaCheckErrors("Copy devDataW volume to host stream1.");
            }
        }
        else
        {
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDataT + shift, 
                                T3D + shift, 
                                (dimSize - shift) * sizeof(RFLOAT), 
                                cudaMemcpyHostToDevice,
                                stream[0]);
                cudaCheckErrors("Copy devDataT volume to device stream0.");
                
                kernel_CalculateW<<<dim, dim, 0, stream[0]>>>(devDataW,
                                                              devDataT, 
                                                              dimSize - shift, 
                                                              shift,
                                                              dim,
                                                              r);
                cudaMemcpyAsync(W3D + shift, 
                                devDataW + shift, 
                                (dimSize - shift) * sizeof(RFLOAT), 
                                cudaMemcpyDeviceToHost,
                                stream[0]);
                cudaCheckErrors("Copy devDataW volume to host stream0.");
           }
        }
    }
    

    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    
    cudaHostUnregister(T3D);
    cudaHostUnregister(W3D);
    
    LOG(INFO) << "Step3: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

    cudaFree(devDataW);
    cudaCheckErrors("Free device memory devDataW.");
    
    cudaFree(devDataT);
    cudaCheckErrors("Free device memory devDataT.");
}

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
                  const int fdim)
{
    cudaSetDevice(gpuIdx);

    int dimSizeP = (pdim / 2 + 1) * pdim;
    int dimSizePRL = pdim * pdim;
    int dimSizeF = (fdim / 2 + 1) * fdim;

    Complex *devDst;
    cudaMalloc((void**)&devDst, dimSizeP * sizeof(Complex));
    cudaCheckErrors("Allocate devDataF data.");

    cudaMemset(devDst, 0.0, dimSizeP * sizeof(Complex));
    cudaCheckErrors("Memset devDst data.");

    Complex *devF;
    cudaMalloc((void**)&devF, dimSizeF * sizeof(Complex));
    cudaCheckErrors("Allocate devDataW data.");

    RFLOAT *devW;
    cudaMalloc((void**)&devW, dimSizeF * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDataW data.");
    
    LOG(INFO) << "Step1: CalculateFW.";
    
    cudaMemcpy(devF,
               F2D,
               dimSizeF * sizeof(Complex),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy.");

    cudaMemcpy(devW,
               W2D,
               dimSizeF * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy.");

    kernel_NormalizeFW2D<<<fdim, 
                           fdim / 2 + 1>>>(devDst,
                                           devF, 
                                           devW, 
                                           r,
                                           pdim,
                                           fdim);

#ifdef SINGLE_PRECISION
    cufftReal *devDstR;
    cudaMalloc((void**)&devDstR, dimSizePRL * sizeof(cufftReal));
    cudaCheckErrors("Allocate device memory for devDstR.");
#else
    cufftDoubleReal *devDstR;
    cudaMalloc((void**)&devDstR, dimSizePRL * sizeof(cufftDoubleReal));
    cudaCheckErrors("Allocate device memory for devDstR.");
#endif    
    
    cufftHandle planc2r;
#ifdef SINGLE_PRECISION
    cufftPlan2d(&planc2r, pdim, pdim, CUFFT_C2R);
#else
    cufftPlan2d(&planc2r, pdim, pdim, CUFFT_Z2D);
#endif    
    
#ifdef SINGLE_PRECISION
    cufftExecC2R(planc2r, (cufftComplex*)devDst, devDstR);
#else
    cufftExecZ2D(planc2r, (cufftDoubleComplex*)devDst, devDstR);
#endif    
    
    cufftDestroy(planc2r);
    cudaCheckErrors("DestroyPlan planc2r.");

    cudaFree(devDst);
    cudaCheckErrors("Free device memory devDst.");
    
    kernel_NormalizeP2D<<<pdim, 
                          pdim>>>(devDstR,
                                  dimSizePRL);

    cudaMemcpy(padDstR,
               devDstR,
               dimSizePRL * sizeof(RFLOAT),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("for memcpy.");

    LOG(INFO) << "Step 2: Clean up the streams and memory.";

    cudaFree(devW);
    cudaCheckErrors("Free device memory of W");
    cudaFree(devF);
    cudaCheckErrors("Free device memory devDataF.");
    cudaFree(devDstR);
    cudaCheckErrors("Free device memory devDst.");
}

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
                const int fdim)
{
    cudaSetDevice(gpuIdx);

    int dimSizeP = (pdim / 2 + 1) * pdim * pdim;
    int dimSizePRL = pdim * pdim * pdim;
    int dimSizeF = (fdim / 2 + 1) * fdim * fdim;
    int batchSize = 8 * fdim * fdim;

    cudaHostRegister(F3D, dimSizeF * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register F3D data.");

    cudaHostRegister(W3D, dimSizeF * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register W3D data.");
    
    Complex *devDst;
    cudaMalloc((void**)&devDst, dimSizeP * sizeof(Complex));
    cudaCheckErrors("Allocate devDataF data.");

    cudaMemset(devDst, 0.0, dimSizeP * sizeof(Complex));
    cudaCheckErrors("Memset devDst data.");

    Complex *devPartF[3];
    RFLOAT *devPartW[3];
    for (int i = 0; i < 3; i++)
    {
        cudaMalloc((void**)&devPartF[i], batchSize * sizeof(Complex));
        cudaCheckErrors("Allocate devDataW data.");

        cudaMalloc((void**)&devPartW[i], batchSize * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devDataW data.");
    }

    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);
    
    LOG(INFO) << "Step1: CalculateFW.";
    
    int batch, smidx = 0;
    for (int i = 0; i < dimSizeF;)
    {
        batch = (i + batchSize > dimSizeF) ? (dimSizeF - i) : batchSize;
        
        cudaMemcpyAsync(devPartF[smidx],
                        F3D + i,
                        batch * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

        cudaMemcpyAsync(devPartW[smidx],
                        W3D + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

        kernel_NormalizeFW<<<fdim, fdim, 0, stream[smidx]>>>(devDst,
                                                             devPartF[smidx], 
                                                             devPartW[smidx], 
                                                             batch, 
                                                             i,
                                                             r,
                                                             pdim,
                                                             fdim);

        i += batch;
        smidx = (++smidx) % 3;
    }
   
    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");

    for (int i = 0; i < 3; i++)
    {
        cudaFree(devPartW[i]);
        cudaCheckErrors("Free device memory of W");
        cudaFree(devPartF[i]);
    cudaCheckErrors("Free device memory devDataF.");
    }
    
    cudaHostUnregister(F3D);
    cudaHostUnregister(W3D);
    
#ifdef SINGLE_PRECISION
    cufftReal *devDstR;
    cudaMalloc((void**)&devDstR, dimSizePRL * sizeof(cufftReal));
    cudaCheckErrors("Allocate device memory for devDstR.");
#else
    cufftDoubleReal *devDstR;
    cudaMalloc((void**)&devDstR, dimSizePRL * sizeof(cufftDoubleReal));
    cudaCheckErrors("Allocate device memory for devDstR.");
#endif    
    
    cufftHandle planc2r;
#ifdef SINGLE_PRECISION
    cufftPlan3d(&planc2r, pdim, pdim, pdim, CUFFT_C2R);
#else
    cufftPlan3d(&planc2r, pdim, pdim, pdim, CUFFT_Z2D);
#endif    
    
#ifdef SINGLE_PRECISION
    cufftExecC2R(planc2r, (cufftComplex*)devDst, devDstR);
#else
    cufftExecZ2D(planc2r, (cufftDoubleComplex*)devDst, devDstR);
#endif    
    
    cufftDestroy(planc2r);
    cudaCheckErrors("DestroyPlan planc2r.");
    
    cudaFree(devDst);
    cudaCheckErrors("Free device memory devDst.");
    
    int pbatchSize = 8 * pdim * pdim;
    smidx = 0;
    for (int i = 0; i < dimSizePRL;)
    {
        batch = (i + pbatchSize > dimSizePRL) ? (dimSizePRL - i) : pbatchSize;
        
        kernel_NormalizeP<<<pdim, 
                            pdim, 
                            0, 
                            stream[smidx]>>>(devDstR,
                                             batch, 
                                             i,
                                             dimSizePRL);

        cudaMemcpyAsync(padDstR + i,
                        devDstR + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        cudaCheckErrors("for memcpy.");

        i += batch;
        smidx = (++smidx) % 3;
    }
   
    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");

    LOG(INFO) << "Step 5: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

    cudaFree(devDstR);
    cudaCheckErrors("Free device memory devDst.");
}

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
                     const int dim)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim;
    int dimSizeRL = dim * dim;
    
    cudaHostRegister(dstI, dimSizeRL * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register dst data.");
    
    RFLOAT *devDstI;
    cudaMalloc((void**)&devDstI, dimSizeRL * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDst data.");
    
#ifdef SINGLE_PRECISION
    cufftComplex *devDstC;
    cudaMalloc((void**)&devDstC, dimSize * sizeof(cufftComplex));
    cudaCheckErrors("Allocate device memory for devDst.");
#else
    cufftDoubleComplex *devDstC;
    cudaMalloc((void**)&devDstC, dimSize * sizeof(cufftDoubleComplex));
    cudaCheckErrors("Allocate device memory for devDst.");
#endif    
    
    LOG(INFO) << "Step2: Correcting Convolution Kernel."; 
    
#ifdef RECONSTRUCTOR_MKB_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devMkb;
    cudaMalloc((void**)&devMkb, mkbSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devMkb data.");
    cudaMemcpy(devMkb, mkbRL, mkbSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devMkb to device.");

#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devTik;
    cudaMalloc((void**)&devTik, mkbSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devTik data.");
    cudaMemcpy(devTik, mkbRL, mkbSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devTik to device.");
#endif
    
    cudaMemcpy(devDstI,
               dstI,
               dimSizeRL * sizeof(RFLOAT),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_MKB_KERNEL
    kernel_CorrectF2D<<<dim, 
                        dim>>>(devDstI, 
                               devMkb,
                               nf,
                               dim);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    kernel_CorrectF2D<<<dim, 
                        dim>>>(devDstI, 
                               devTik,
                               dim);
#endif

    cufftHandle planr2c;
#ifdef SINGLE_PRECISION
    cufftPlan2d(&planr2c, dim, dim, CUFFT_R2C);
#else
    cufftPlan2d(&planr2c, dim, dim, CUFFT_D2Z);
#endif    
    
#ifdef SINGLE_PRECISION
    cufftExecR2C(planr2c, (cufftReal*)devDstI, devDstC);
#else
    cufftExecD2Z(planr2c, (cufftDoubleReal*)devDstI, devDstC);
#endif    
    
    cudaMemcpy(dst,
               devDstC,
               dimSize * sizeof(Complex),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("out for memcpy 0.");
    
    LOG(INFO) << "Step 3: Clean up the streams and memory.";

    cufftDestroy(planr2c);
    cudaCheckErrors("DestroyPlan planr2c.");
    
#ifdef RECONSTRUCTOR_MKB_KERNEL
    cudaFree(devMkb);
    cudaCheckErrors("Free device memory devDst.");
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    cudaFree(devTik);
    cudaCheckErrors("Free device memory devDst.");
#endif
    cudaFree(devDstI);
    cudaCheckErrors("Free device memory devDst.");

    cudaFree(devDstC);
    cudaCheckErrors("Free device memory devDst.");
}

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
                   const int dim)
{
    cudaSetDevice(gpuIdx);

    int dimSizeRL = dim * dim * dim;
    int dimSize = (dim / 2 + 1) * dim * dim;

    cudaHostRegister(dstN, dimSizeRL * sizeof(RFLOAT), cudaHostRegisterDefault);
    cudaCheckErrors("Register dst data.");
    
    RFLOAT *devDstN;
    cudaMalloc((void**)&devDstN, dimSizeRL * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devDst data.");
    
    LOG(INFO) << "Step2: Correcting Convolution Kernel."; 
    
#ifdef RECONSTRUCTOR_MKB_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devMkb;
    cudaMalloc((void**)&devMkb, mkbSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devMkb data.");
    cudaMemcpy(devMkb, mkbRL, mkbSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devMkb to device.");

#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devTik;
    cudaMalloc((void**)&devTik, mkbSize * sizeof(RFLOAT));
    cudaCheckErrors("Allocate devTik data.");
    cudaMemcpy(devTik, mkbRL, mkbSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    cudaCheckErrors("Copy devTik to device.");
#endif
    
    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);

    int batchSize = 8 * dim * dim;
    
    int batch, smidx = 0;
    for (int i = 0; i < dimSizeRL;)
    {
        batch = (i + batchSize > dimSizeRL) ? (dimSizeRL - i) : batchSize;
        
        cudaMemcpyAsync(devDstN + i,
                        dstN + i,
                        batch * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_MKB_KERNEL
        kernel_CorrectF<<<dim, 
                          dim, 
                          0, 
                          stream[smidx]>>>(devDstN, 
                                           devMkb,
                                           nf,
                                           dim,
                                           batch,
                                           i);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        kernel_CorrectF<<<dim, 
                          dim, 
                          0, 
                          stream[smidx]>>>(devDstN, 
                                           devTik,
                                           dim,
                                           batch, 
                                           i);
#endif

        i += batch;
        smidx = (++smidx) % 3;
    }
   
    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    
    cudaHostUnregister(dstN);
    
#ifdef SINGLE_PRECISION
    cufftComplex *devDst;
    cudaMalloc((void**)&devDst, dimSize * sizeof(cufftComplex));
    cudaCheckErrors("Allocate device memory for devDst.");
#else
    cufftDoubleComplex *devDst;
    cudaMalloc((void**)&devDst, dimSize * sizeof(cufftDoubleComplex));
    cudaCheckErrors("Allocate device memory for devDst.");
#endif    
    
    cufftHandle planr2c;
#ifdef SINGLE_PRECISION
    cufftPlan3d(&planr2c, dim, dim, dim, CUFFT_R2C);
#else
    cufftPlan3d(&planr2c, dim, dim, dim, CUFFT_D2Z);
#endif    
    
#ifdef SINGLE_PRECISION
    cufftExecR2C(planr2c, (cufftReal*)devDstN, devDst);
#else
    cufftExecD2Z(planr2c, (cufftDoubleReal*)devDstN, devDst);
#endif    
    
    cudaMemcpy(dst,
               devDst,
               dimSize * sizeof(Complex),
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("out for memcpy 1.");

    LOG(INFO) << "Step 3: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

    cufftDestroy(planr2c);
    cudaCheckErrors("DestroyPlan planr2c.");
    
#ifdef RECONSTRUCTOR_MKB_KERNEL
    cudaFree(devMkb);
    cudaCheckErrors("Free device memory devDst.");
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    cudaFree(devTik);
    cudaCheckErrors("Free device memory devDst.");
#endif
    cudaFree(devDstN);
    cudaCheckErrors("Free device memory devDst.");

    cudaFree(devDst);
    cudaCheckErrors("Free device memory devDst.");
}

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
                  int dim)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim;

    cudaHostRegister(src, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register src data.");
    
    Complex *devSrc;
    cudaMalloc((void**)&devSrc, dimSize * sizeof(Complex));
    cudaCheckErrors("Allocate devSrc data.");
    
    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);

    int batchSize = dim / 2 + 1;
    
    int batch, smidx = 0;
    for (int i = 0; i < dimSize;)
    {
        batch = (i + batchSize > dimSize) ? (dimSize - i) : batchSize;
        
        cudaMemcpyAsync(devSrc + i,
                        src + i,
                        batch * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy 0.");

        kernel_TranslateI2D<<<1, 
                              dim / 2 + 1, 
                              0, 
                              stream[smidx]>>>(devSrc, 
                                               ox,
                                               oy,
                                               r,
                                               i,
                                               dim);
        
        cudaMemcpyAsync(src + i,
                        devSrc + i,
                        batch * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        cudaCheckErrors("for memcpy 0.");

        i += batch;
        smidx = (++smidx) % 3;
    }
   
    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    
    cudaHostUnregister(src);
    
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

    cudaFree(devSrc);
    cudaCheckErrors("Free device memory devDst.");
}

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
                int dim)
{
    cudaSetDevice(gpuIdx);

    int dimSize = (dim / 2 + 1) * dim * dim;

    cudaHostRegister(ref, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    cudaCheckErrors("Register src data.");
    
    Complex *devRef;
    cudaMalloc((void**)&devRef, dimSize * sizeof(Complex));
    cudaCheckErrors("Allocate devSrc data.");
    
    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);

    int batchSize = 8 * dim * dim;
    
    int batch, smidx = 0;
    for (int i = 0; i < dimSize;)
    {
        batch = (i + batchSize > dimSize) ? (dimSize - i) : batchSize;
        
        cudaMemcpyAsync(devRef + i,
                        ref + i,
                        batch * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[smidx]);
        cudaCheckErrors("for memcpy 0.");

        kernel_TranslateI<<<dim, 
                            dim / 2 + 1, 
                            0, 
                            stream[smidx]>>>(devRef, 
                                             ox,
                                             oy,
                                             oz,
                                             r,
                                             i,
                                             dim,
                                             batch);
        
        cudaMemcpyAsync(ref + i,
                        devRef + i,
                        batch * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[smidx]);
        cudaCheckErrors("for memcpy 0.");

        i += batch;
        smidx = (++smidx) % 3;
    }
   
    cudaStreamSynchronize(stream[0]);
    cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    cudaCheckErrors("CUDA stream1 synchronization.");
    
    cudaHostUnregister(ref);
    
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    cudaCheckErrors("Destroy stream.");

    cudaFree(devRef);
    cudaCheckErrors("Free device memory devDst.");
}

/**
 * @brief Pre-calculation in expectation.
 *
 * @param
 * @param
 */
void reMask(vector<Complex*>& imgData,
            RFLOAT maskRadius,
            RFLOAT pixelSize,
            RFLOAT ew,
            int idim,
            int imgNum)
{
    LOG(INFO) << "ReMask begin.";
    
    int numDevs;
    cudaGetDeviceCount(&numDevs);
    //cudaCheckErrors("get devices num.");

    int* gpus = new int[numDevs];
    int aviDevs = 0;
    for (int n = 0; n < numDevs; n++)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, n);
        if (deviceProperties.major >= 3 && deviceProperties.minor >= 0)
        {
            gpus[aviDevs++] = n;
        }
    }

    RFLOAT *devMask[aviDevs];
    RFLOAT r = maskRadius / pixelSize;
    int imgSize = idim * (idim / 2 + 1);
    int imgSizeRL = idim * idim;
    int blockDim = idim > 512 ? 512 : idim;
    
    for (int n = 0; n < aviDevs; ++n) 
    {     
        cudaSetDevice(gpus[n]); 
    
        cudaMalloc((void**)&devMask[n], imgSizeRL * sizeof(RFLOAT));
        cudaCheckErrors("Allocate devMask data.");
        
        kernel_SoftMask<<<idim, blockDim>>>(devMask[n],
                                            r,
                                            ew,
                                            idim,
                                            imgSizeRL);
        cudaCheckErrors("kernel SoftMask error.");
    }

    int streamNum = aviDevs * NUM_STREAM;
    
    Complex *pglk_imageI_buf[streamNum];
    Complex *pglk_imageO_buf[streamNum];
    Complex *dev_image_buf[streamNum];
    RFLOAT *dev_imageF_buf[streamNum];
    CB_UPIB_t cbArgsA[streamNum], cbArgsB[streamNum];
    
    LOG(INFO) << "alloc Memory.";
    
    cudaStream_t stream[streamNum];
    cufftHandle* planc2r = (cufftHandle*)malloc(sizeof(cufftHandle) * streamNum);
    cufftHandle* planr2c = (cufftHandle*)malloc(sizeof(cufftHandle) * streamNum);
    
    int baseS;
    const int BATCH_SIZE = BUFF_SIZE;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]); 
    
        for (int i = 0; i < NUM_STREAM; i++)
        {
            allocPGLKImagesBuffer(&pglk_imageI_buf[i + baseS], idim, BATCH_SIZE);
            allocPGLKImagesBuffer(&pglk_imageO_buf[i + baseS], idim, BATCH_SIZE);
            allocDeviceComplexBuffer(&dev_image_buf[i + baseS], BATCH_SIZE * imgSize);
            allocDeviceParamBuffer(&dev_imageF_buf[i + baseS], BATCH_SIZE * imgSizeRL);
            
            cudaStreamCreate(&stream[i + baseS]);
#ifdef SINGLE_PRECISION
            cufftPlan2d(&planc2r[i + baseS], idim, idim, CUFFT_C2R);
            cufftPlan2d(&planr2c[i + baseS], idim, idim, CUFFT_R2C);
#else
            cufftPlan2d(&planc2r[i + baseS], idim, idim, CUFFT_Z2D);
            cufftPlan2d(&planr2c[i + baseS], idim, idim, CUFFT_D2Z);
#endif   
            cufftSetStream(planc2r[i + baseS], stream[i + baseS]);
            cufftSetStream(planr2c[i + baseS], stream[i + baseS]);
        }       
    }

    LOG(INFO) << "alloc memory done, begin to calculate...";
    
    int batchSize = 0, smidx = 0;
    
    for (int i = 0; i < imgNum;) 
    {    
        for (int n = 0; n < aviDevs; ++n) 
        {     
            if (i >= imgNum)
                break;
           
            baseS = n * NUM_STREAM;
            batchSize = (i + BATCH_SIZE < imgNum) ? BATCH_SIZE : (imgNum - i);
            //printf("batch:%d, smidx:%d, baseS:%d\n", batchSize, smidx, baseS);
            
            cudaSetDevice(gpus[n]); 

            cbArgsA[smidx + baseS].pglkptr = pglk_imageI_buf[smidx + baseS];
            cbArgsA[smidx + baseS].images = &imgData;
            cbArgsA[smidx + baseS].imageSize = imgSize;
            cbArgsA[smidx + baseS].batchSize = batchSize;
            cbArgsA[smidx + baseS].basePos = i;
            
            cudaStreamAddCallback(stream[smidx + baseS], 
                                  cbUpdatePGLKImagesBuffer, 
                                  (void*)&cbArgsA[smidx + baseS], 
                                  0);
           
            cudaMemcpyAsync(dev_image_buf[smidx + baseS],
                            pglk_imageI_buf[smidx + baseS],
                            batchSize * imgSize * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy image to device.");
            
            for (int r = 0; r < batchSize; r++)
            {
#ifdef SINGLE_PRECISION
                cufftExecC2R(planc2r[smidx + baseS], 
                             (cufftComplex *)dev_image_buf[smidx + baseS] + r * imgSize, 
                             dev_imageF_buf[smidx + baseS] + r * imgSizeRL);
#else
                cufftExecZ2D(planc2r[smidx + baseS], 
                             (cufftComplex *)dev_image_buf[smidx + baseS] + r * imgSize, 
                             dev_imageF_buf[smidx + baseS] + r * imgSizeRL);
#endif    
               
                kernel_MulMask<<<idim, 
                                 blockDim, 
                                 0, 
                                 stream[smidx + baseS]>>>(dev_imageF_buf[smidx + baseS],
                                                          devMask[n],
                                                          r,
                                                          idim,
                                                          imgSizeRL);
                cudaCheckErrors("kernel MulMask error.");

#ifdef SINGLE_PRECISION
                cufftExecR2C(planr2c[smidx + baseS], 
                             dev_imageF_buf[smidx + baseS] + r * imgSizeRL, 
                             (cufftComplex *)dev_image_buf[smidx + baseS] + r * imgSize);
#else
                cufftExecD2Z(planr2c[smidx + baseS], 
                             dev_imageF_buf[smidx + baseS] + r * imgSizeRL, 
                             (cufftComplex *)dev_image_buf[smidx + baseS] + r * imgSize);
#endif    
            }
            
            cudaMemcpyAsync(pglk_imageO_buf[smidx + baseS],
                            dev_image_buf[smidx + baseS],
                            batchSize * imgSize * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            cudaCheckErrors("memcpy image to host.");
            
            cbArgsB[smidx + baseS].pglkptr = pglk_imageO_buf[smidx + baseS];
            cbArgsB[smidx + baseS].images = &imgData;
            cbArgsB[smidx + baseS].imageSize = imgSize;
            cbArgsB[smidx + baseS].batchSize = batchSize;
            cbArgsB[smidx + baseS].basePos = i;
            
            cudaStreamAddCallback(stream[smidx + baseS], 
                                  cbUpdateImagesBuffer, 
                                  (void*)&cbArgsB[smidx + baseS], 
                                  0);
           
            i += batchSize;
        }
        
        smidx = (++smidx) % NUM_STREAM;
    }

    //synchronizing on CUDA streams 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        //cudaDeviceSynchronize();
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamSynchronize(stream[i + baseS]); 
            cudaCheckErrors("Stream synchronize.");

            cudaFreeHost(pglk_imageI_buf[i + baseS]);
            cudaFreeHost(pglk_imageO_buf[i + baseS]);
            cudaFree(dev_image_buf[i + baseS]);
            cudaFree(dev_imageF_buf[i + baseS]);
        }
    } 
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(devMask[n]); 
        
        for (int i = 0; i < NUM_STREAM; i++)
        {
            cudaStreamDestroy(stream[i + baseS]);
            cudaCheckErrors("Stream destory.");
    
            cufftDestroy(planc2r[i + baseS]);
            cudaCheckErrors("DestroyPlan planc2r.");

            cufftDestroy(planr2c[i + baseS]);
            cudaCheckErrors("DestroyPlan planr2c.");
        }
    } 
  
    delete[] gpus; 
    LOG(INFO) << "ReMask done.";
}

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
                   const int edgeWidth)
{
    cudaSetDevice(gpuIdx);

    int dimSize = dim * dim * dim;
    
    cudaHostRegister(dst, dimSize * sizeof(RFLOAT), cudaHostRegisterDefault);
    //cudaCheckErrors("Register dst data.");
    
    RFLOAT *devDst;
    cudaMalloc((void**)&devDst, dimSize * sizeof(RFLOAT));
    //cudaCheckErrors("Allocate devDst data.");
    
    LOG(INFO) << "Step2: Correcting Convolution Kernel."; 
    
#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devMkb;
    cudaMalloc((void**)&devMkb, mkbSize * sizeof(RFLOAT));
    //cudaCheckErrors("Allocate devMkb data.");
    cudaMemcpy(devMkb, mkbRL, mkbSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    //cudaCheckErrors("Copy devMkb to device.");

#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *devTik;
    cudaMalloc((void**)&devTik, mkbSize * sizeof(RFLOAT));
    //cudaCheckErrors("Allocate devTik data.");
    cudaMemcpy(devTik, mkbRL, mkbSize * sizeof(RFLOAT), cudaMemcpyHostToDevice);
    //cudaCheckErrors("Copy devTik to device.");
#endif

#endif
    
    
    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);

    int batchSize = 8 * dim * dim;
    int len = dimSize / batchSize;
    int streamN = len / 3;
    
    for (int i = 0; i < streamN; i++)
    {
        int shift = i * 3 * batchSize;
        
        cudaMemcpyAsync(devDst + shift,
                        dst + shift,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[0]>>>(devDst, 
                                                    devMkb,
                                                    nf,
                                                    dim,
                                                    batchSize,
                                                    shift);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[0]>>>(devDst, 
                                                    devTik,
                                                    dim,
                                                    batchSize, 
                                                    shift);
#endif

#endif

        cudaMemcpyAsync(devDst + shift + batchSize,
                        dst + shift + batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("for memcpy 1.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[1]>>>(devDst, 
                                                    devMkb,
                                                    nf,
                                                    dim,
                                                    batchSize,
                                                    shift + batchSize);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[1]>>>(devDst, 
                                                    devTik,
                                                    dim,
                                                    batchSize, 
                                                    shift + batchSize);
#endif

#endif
        
        cudaMemcpyAsync(devDst + shift + 2 * batchSize,
                        dst + shift + 2 * batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[2]);
        //cudaCheckErrors("for memcpy 2.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[2]>>>(devDst, 
                                                    devMkb,
                                                    nf,
                                                    dim,
                                                    batchSize,
                                                    shift + 2 * batchSize);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[2]>>>(devDst, 
                                                    devTik,
                                                    dim,
                                                    batchSize, 
                                                    shift + 2 * batchSize);
#endif

#endif
    }

    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
        
        cudaMemcpyAsync(devDst + shift,
                        dst + shift,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[0]>>>(devDst, 
                                                    devMkb,
                                                    nf,
                                                    dim,
                                                    batchSize,
                                                    shift);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[0]>>>(devDst, 
                                                    devTik,
                                                    dim,
                                                    batchSize, 
                                                    shift);
#endif

#endif
    
        cudaMemcpyAsync(devDst + shift + batchSize,
                        dst + shift + batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("for memcpy 1.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[1]>>>(devDst, 
                                                    devMkb,
                                                    nf,
                                                    dim,
                                                    batchSize,
                                                    shift + batchSize);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        kernel_CorrectF<<<dim, dim, 0, stream[1]>>>(devDst, 
                                                    devTik,
                                                    dim,
                                                    batchSize, 
                                                    shift + batchSize);
#endif

#endif
        
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
            cudaMemcpyAsync(devDst + shift,
                            dst + shift,
                            (dimSize - shift) * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[2]);
            //cudaCheckErrors("for memcpy 2.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
            kernel_CorrectF<<<dim, dim, 0, stream[2]>>>(devDst, 
                                                        devMkb,
                                                        nf,
                                                        dim,
                                                        dimSize - shift,
                                                        shift);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            kernel_CorrectF<<<dim, dim, 0, stream[2]>>>(devDst, 
                                                        devTik,
                                                        dim,
                                                        dimSize - shift,
                                                        shift);
#endif

#endif
    
        }    

    }
    
    else
    {
        if (len % 3 == 1)
        {
            int shift = (len - 1) * batchSize;
            cudaMemcpyAsync(devDst + shift,
                            dst + shift,
                            batchSize * sizeof(RFLOAT),
                            cudaMemcpyHostToDevice,
                            stream[0]);
            //cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
            kernel_CorrectF<<<dim, dim, 0, stream[0]>>>(devDst, 
                                                        devMkb,
                                                        nf,
                                                        dim,
                                                        batchSize,
                                                        shift);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            kernel_CorrectF<<<dim, dim, 0, stream[0]>>>(devDst, 
                                                        devTik,
                                                        dim,
                                                        batchSize, 
                                                        shift);
#endif

#endif
            
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDst + shift,
                                dst + shift,
                                (dimSize - shift) * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[1]);
                //cudaCheckErrors("for memcpy 1.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
                kernel_CorrectF<<<dim, dim, 0, stream[1]>>>(devDst, 
                                                            devMkb,
                                                            nf,
                                                            dim,
                                                            dimSize - shift,
                                                            shift);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                kernel_CorrectF<<<dim, dim, 0, stream[1]>>>(devDst, 
                                                            devTik,
                                                            dim,
                                                            dimSize - shift,
                                                            shift);
#endif

#endif
    
            }    
        }
        else
        {
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDst + shift,
                                dst + shift,
                                (dimSize - shift) * sizeof(RFLOAT),
                                cudaMemcpyHostToDevice,
                                stream[0]);
                //cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
                kernel_CorrectF<<<dim, dim, 0, stream[0]>>>(devDst, 
                                                            devMkb,
                                                            nf,
                                                            dim,
                                                            dimSize - shift,
                                                            shift);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                kernel_CorrectF<<<dim, dim, 0, stream[0]>>>(devDst, 
                                                            devTik,
                                                            dim,
                                                            dimSize - shift,
                                                            shift);
#endif

#endif
            }    
        }
    } 
    
#ifdef RECONSTRUCTOR_REMOVE_CORNER
    LOG(INFO) << "Step2: SoftMask dst."; 
    
    RFLOAT *bg;
    cudaMalloc((void**)&bg, sizeof(RFLOAT));
    //cudaCheckErrors("Allocate device memory for devSumG.");

#ifdef RECONSTRUCTOR_REMOVE_CORNER_MASK_ZERO
    cudaMemset(bg, 0.0, sizeof(RFLOAT));
#else
    
    RFLOAT *devSumG;
    RFLOAT *devSumWG;
    cudaMalloc((void**)&devSumG, dim * sizeof(RFLOAT));
    //cudaCheckErrors("Allocate device memory for devSumG.");
    cudaMalloc((void**)&devSumWG, dim * sizeof(RFLOAT));
    //cudaCheckErrors("Allocate device memory for devSumWG.");
    
    kernel_Background<<<dim, dim, 2 * dim * sizeof(RFLOAT)>>>(devDst,
                                                              devSumG,
                                                              devSumWG,
                                                              size / 2,
                                                              edgeWidth,
                                                              dim,
                                                              dimSize);
    
    kernel_CalculateBg<<<1, dim>>>(devSumG,
                                   devSumWG,
                                   bg,
                                   dim);

    cudaFree(devSumG);
    //cudaCheckErrors("Free device memory devSumG.");
    
    cudaFree(devSumWG);
    //cudaCheckErrors("Free device memory devSumWG.");
#endif
    
#endif

    for (int i = 0; i < streamN; i++)
    {
        int shift = i * 3 * batchSize;
#ifdef RECONSTRUCTOR_REMOVE_CORNER
        kernel_SoftMaskD<<<dim, dim, 0, stream[0]>>>(devDst,
                                                     bg,
                                                     size / 2,
                                                     edgeWidth,
                                                     dim,
                                                     batchSize,
                                                     shift);
#endif

        cudaMemcpyAsync(dst + shift,
                        devDst + shift,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("out for memcpy 0.");
    
#ifdef RECONSTRUCTOR_REMOVE_CORNER
        kernel_SoftMaskD<<<dim, dim, 0, stream[1]>>>(devDst,
                                                     bg,
                                                     size / 2,
                                                     edgeWidth,
                                                     dim,
                                                     batchSize,
                                                     shift + batchSize);
#endif
        
        cudaMemcpyAsync(dst + shift + batchSize,
                        devDst + shift + batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("out for memcpy 1.");

#ifdef RECONSTRUCTOR_REMOVE_CORNER
        kernel_SoftMaskD<<<dim, dim, 0, stream[2]>>>(devDst,
                                                     bg,
                                                     size / 2,
                                                     edgeWidth,
                                                     dim,
                                                     batchSize,
                                                     shift + 2 * batchSize);
#endif
        
        cudaMemcpyAsync(dst + shift + 2 * batchSize,
                        devDst + shift + 2 * batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[2]);
        //cudaCheckErrors("out for memcpy 2.");
    }
   
    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
#ifdef RECONSTRUCTOR_REMOVE_CORNER
        kernel_SoftMaskD<<<dim, dim, 0, stream[0]>>>(devDst,
                                                     bg,
                                                     size / 2,
                                                     edgeWidth,
                                                     dim,
                                                     batchSize,
                                                     shift);
#endif

        cudaMemcpyAsync(dst + shift,
                        devDst + shift,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("out for memcpy 0.");
    
#ifdef RECONSTRUCTOR_REMOVE_CORNER
        kernel_SoftMaskD<<<dim, dim, 0, stream[1]>>>(devDst,
                                                     bg,
                                                     size / 2,
                                                     edgeWidth,
                                                     dim,
                                                     batchSize,
                                                     shift + batchSize);
#endif
        
        cudaMemcpyAsync(dst + shift + batchSize,
                        devDst + shift + batchSize,
                        batchSize * sizeof(RFLOAT),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("out for memcpy 1.");

        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
#ifdef RECONSTRUCTOR_REMOVE_CORNER
            kernel_SoftMaskD<<<dim, dim, 0, stream[2]>>>(devDst,
                                                         bg,
                                                         size / 2,
                                                         edgeWidth,
                                                         dim,
                                                         dimSize - shift,
                                                         shift);
#endif
        
            cudaMemcpyAsync(dst + shift,
                            devDst + shift,
                            (dimSize - shift) * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[2]);
            //cudaCheckErrors("out for memcpy 2.");
        }    

    }
    
    else
    {
        if (len % 3 == 1)
        {
            int shift = (len - 1) * batchSize;
#ifdef RECONSTRUCTOR_REMOVE_CORNER
            kernel_SoftMaskD<<<dim, dim, 0, stream[0]>>>(devDst,
                                                         bg,
                                                         size / 2,
                                                         edgeWidth,
                                                         dim,
                                                         batchSize,
                                                         shift);
#endif

            cudaMemcpyAsync(dst + shift,
                            devDst + shift,
                            batchSize * sizeof(RFLOAT),
                            cudaMemcpyDeviceToHost,
                            stream[0]);
            //cudaCheckErrors("out for memcpy 0.");
            
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
#ifdef RECONSTRUCTOR_REMOVE_CORNER
                kernel_SoftMaskD<<<dim, dim, 0, stream[1]>>>(devDst,
                                                             bg,
                                                             size / 2,
                                                             edgeWidth,
                                                             dim,
                                                             dimSize - shift,
                                                             shift);
#endif
        
                cudaMemcpyAsync(dst + shift,
                                devDst + shift,
                                (dimSize - shift) * sizeof(RFLOAT),
                                cudaMemcpyDeviceToHost,
                                stream[1]);
                //cudaCheckErrors("out for memcpy 1.");
            }    

        }
        else
        {
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
#ifdef RECONSTRUCTOR_REMOVE_CORNER
                kernel_SoftMaskD<<<dim, dim, 0, stream[0]>>>(devDst,
                                                             bg,
                                                             size / 2,
                                                             edgeWidth,
                                                             dim,
                                                             dimSize - shift,
                                                             shift);
#endif
        
                cudaMemcpyAsync(dst + shift,
                                devDst + shift,
                                (dimSize - shift) * sizeof(RFLOAT),
                                cudaMemcpyDeviceToHost,
                                stream[0]);
                //cudaCheckErrors("out for memcpy 0.");
            }    


        }
    } 
    
    cudaStreamSynchronize(stream[0]);
    //cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    
    cudaHostUnregister(dst);
    
    LOG(INFO) << "Step 3: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    //cudaCheckErrors("Destroy stream.");

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL
#ifdef RECONSTRUCTOR_MKB_KERNEL
    cudaFree(devMkb);
    //cudaCheckErrors("Free device memory devDst.");
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    cudaFree(devTik);
    //cudaCheckErrors("Free device memory devDst.");
#endif
#endif

#ifdef RECONSTRUCTOR_REMOVE_CORNER
    cudaFree(bg);
    //cudaCheckErrors("Free device memory devDst.");
#endif
    
    cudaFree(devDst);
    //cudaCheckErrors("Free device memory devDst.");
}


////////////////////////////////////////////////////////////////
// TODO cudarize more modules.
//

////////////////////////////////////////////////////////////////

} // end namespace cuthunder

////////////////////////////////////////////////////////////////
