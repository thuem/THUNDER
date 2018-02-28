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

    for (int i = 0; i < args->batchSize; i++) {
        memcpy((void*)(args->pglkptr + i * args->imageSize),
               (void*)(*args->images)[args->basePos + i],
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
void allocDeviceImagesBuffer(Complex **devptr, int ndim, int length)
{
    size_t size = length * (ndim / 2 + 1) * ndim;

    cudaMalloc((void**)devptr, size * sizeof(Complex));
    //cudaCheckErrors("Alloc device memory of batch size images.");
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
void allocDeviceNCBuffer(unsigned int **devptr, int length)
{
    cudaMalloc((void**)devptr, length * sizeof(unsigned int));
    //cudaCheckErrors("Alloc device memory of batch NC.");
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
void allocDeviceParamBuffer(double **devptr, int length)
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
void uploadTabFunction(TabFunction& tabfunc, const double *tab)
{
    double *devptr;

    cudaMalloc((void**)&devptr, tabfunc.size() * sizeof(double));
    //cudaCheckErrors("Alloc device memory for tabfunction.");

    cudaMemcpy((void*)devptr,
               (void*)tab,
               tabfunc.size() * sizeof(double),
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
                  double* def,
                  double* k1,
                  double* k2,
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
    double *dev_def_buf[streamNum];
    double *dev_k1_buf[streamNum];
    double *dev_k2_buf[streamNum];

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
                            batchSize * npxl * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy def to host.");
            
            cudaMemcpyAsync(k1 + i,
                            dev_k1_buf[smidx + baseS],
                            batchSize * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy k1 to host.");
            
            cudaMemcpyAsync(k2 + i,
                            dev_k2_buf[smidx + baseS],
                            batchSize * sizeof(double),
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
                    double* ctfP,
                    double* sigRcpP,
                    double* trans,
                    double* wC,
                    double* wR,
                    double* wT,
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
    double* devnR[aviDevs]; 
    double* devRotm[aviDevs]; 
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];

    cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindSigned);
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
    
    cudaHostRegister(ctfP, imgNum * npxl * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register ctfP data.");
    
    cudaHostRegister(sigRcpP, imgNum * npxl * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register sigRcpP data.");
    
    cudaHostRegister(wC, imgNum * nK * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register wC data.");
    
    cudaHostRegister(wR, imgNum * nR * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register wR data.");
    
    cudaHostRegister(wT, imgNum * nT * sizeof(double), cudaHostRegisterDefault);
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
    double* devctfP[streamNum];
    double* devsigP[streamNum];
    double* devDvp[streamNum];
    double* devbaseL[streamNum];
    double* devwC[streamNum];
    double* devwR[streamNum];
    double* devwT[streamNum];

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
            allocDeviceParamBuffer(&devDvp[i + baseS], BATCH_SIZE * BATCH_SIZE * nT);
            allocDeviceParamBuffer(&devbaseL[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&devwC[i + baseS], BATCH_SIZE * nK);
            allocDeviceParamBuffer(&devwR[i + baseS], BATCH_SIZE * nR);
            allocDeviceParamBuffer(&devwT[i + baseS], BATCH_SIZE * nT);
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
                            batchSize * npxl * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy ctfP to device.");
            
            cudaMemcpyAsync(devsigP[smidx + baseS],
                            sigRcpP + i * npxl,
                            batchSize * npxl * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy sigP to device.");
                
            cudaMemsetAsync(devbaseL[smidx + baseS],
                            0.0,
                            batchSize * sizeof(double),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset baseL.");
        
            cudaMemsetAsync(devwC[smidx + baseS],
                            0.0,
                            batchSize * nK * sizeof(double),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset wC.");
        
            cudaMemsetAsync(devwR[smidx + baseS],
                            0.0,
                            batchSize * nR * sizeof(double),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset wR.");
        
            cudaMemsetAsync(devwT[smidx + baseS],
                            0.0,
                            batchSize * nT * sizeof(double),
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
                                       512, 
                                       512 * sizeof(double), 
                                       stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                                priRotP[smidx + baseS],
                                                                devtraP[n],
                                                                devctfP[smidx + baseS],
                                                                devsigP[smidx + baseS],
                                                                devDvp[smidx + baseS],
                                                                nT,
                                                                rbatch,
                                                                npxl);

                    kernel_UpdateW2D<<<1,
                                       batchSize,
                                       0,
                                       stream[smidx + baseS]>>>(devDvp[smidx + baseS],
                                                                devbaseL[smidx + baseS],
                                                                devwC[smidx + baseS],
                                                                devwR[smidx + baseS],
                                                                devwT[smidx + baseS],
                                                                k,
                                                                r,
                                                                nK,
                                                                nR,
                                                                nT,
                                                                rbatch * nT);
                        
                    r += rbatch; 
                }
            }

            cudaMemcpyAsync(wC + i * nK,
                            devwC[smidx + baseS],
                            batchSize * nK * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy wC to host.");
                
            cudaMemcpyAsync(wR + i * nR,
                            devwR[smidx + baseS],
                            batchSize * nR * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy wR to host.");
            
            cudaMemcpyAsync(wT + i * nT,
                            devwT[smidx + baseS],
                            batchSize * nT * sizeof(double),
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
        }
    } 
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(devnR[n]);
        cudaFree(deviCol[n]); 
        cudaFree(deviRow[n]); 
        cudaFree(devtraP[n]); 
        cudaFree(devRotm[n]); 
        
        for (int k = 0; k < nK; k++)
            cudaFreeArray(symArray[k + n * nK]);
        
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
void expectGlobal3D(Complex* volume,
                    Complex* datP,
                    double* ctfP,
                    double* sigRcpP,
                    double* trans,
                    double* wC,
                    double* wR,
                    double* wT,
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
    double* devnR[aviDevs]; 
    double* devRotm[aviDevs]; 
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];

    cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindSigned);
    cudaArray *symArray[aviDevs]; 
    struct cudaResourceDesc resDesc[aviDevs];
    cudaExtent extent = make_cudaExtent(vdim / 2 + 1, vdim, vdim);
    cudaMemcpy3DParms copyParams[aviDevs];
    cudaTextureObject_t texObject[aviDevs];
    
    dimSize = (vdim / 2 + 1) * vdim * vdim;
    
    cudaHostRegister(volume, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register volume data.");
    
    cudaHostRegister(rot, nR * 4 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register rot data.");
    
    int baseS;
    const int BATCH_SIZE = BUFF_SIZE;

    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]); 
        
        cudaMalloc3DArray(&symArray[n], &channelDesc, extent);
        //cudaCheckErrors("Allocate symArray data.");
        
        cudaMalloc((void**)&devtraP[n], nT * npxl * sizeof(Complex));
        //cudaCheckErrors("Allocate traP data.");
        
        cudaMalloc((void**)&deviCol[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iCol data.");
        
        cudaMalloc((void**)&deviRow[n], npxl * sizeof(int));
        //cudaCheckErrors("Allocate iRow data.");
        
        cudaMalloc((void**)&dev_trans[n], nT * 2 * sizeof(double));
        //cudaCheckErrors("Allocate trans data.");
        
        cudaMalloc((void**)&devnR[n], nR * 4 * sizeof(double));
        //cudaCheckErrors("Allocate nR data.");
        
        cudaMalloc((void**)&devRotm[n], nR * 9 * sizeof(double));
        //cudaCheckErrors("Allocate rot data.");
        
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
    
    }        
    
    int rblock;
    
    for (int n = 0; n < aviDevs; ++n) 
    {     
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]); 
        
        if (nR % 200 != 0)
            rblock = nR / 200 + 1;
        else
            rblock = nR / 200;

        cudaMemcpyAsync(devnR[n],
                        rot,
                        nR * 4 * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[0 + baseS]);
        //cudaCheckErrors("memcpy rot to device.");
        
        kernel_getRotMat<<<rblock, 
                           200, 
                           200 * 18 * sizeof(double), 
                           stream[0 + baseS]>>>(devRotm[n],
                                                devnR[n],
                                                nR);
        //cudaCheckErrors("getRotMat3D kernel.");
        
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
        
        copyParams[n] = {0};
        copyParams[n].srcPtr   = make_cudaPitchedPtr((void*)volume, 
                                                     (vdim / 2 + 1) * sizeof(int4), 
                                                     vdim / 2 + 1, 
                                                     vdim);
        copyParams[n].dstArray = symArray[n];
        copyParams[n].extent   = extent;
        copyParams[n].kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3DAsync(&copyParams[n], stream[2 + baseS]);
        //cudaCheckErrors("memcpy array error");
    }        
    
    cudaHostRegister(datP, imgNum * npxl * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register datP data.");
    
    cudaHostRegister(ctfP, imgNum * npxl * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register ctfP data.");
    
    cudaHostRegister(sigRcpP, imgNum * npxl * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register sigRcpP data.");
    
    cudaHostRegister(wC, imgNum * nK * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register wC data.");
    
    cudaHostRegister(wR, imgNum * nR * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register wR data.");
    
    cudaHostRegister(wT, imgNum * nT * sizeof(double), cudaHostRegisterDefault);
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
        memset(&resDesc[n], 0, sizeof(resDesc[0]));
        resDesc[n].resType = cudaResourceTypeArray;
        resDesc[n].res.array.array = symArray[n];
        
        cudaSetDevice(gpus[n]);
        cudaCreateTextureObject(&texObject[n], &resDesc[n], &td, NULL);
        //cudaCheckErrors("create TexObject.");
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
    double* devctfP[streamNum];
    double* devsigP[streamNum];
    double* devDvp[streamNum];
    double* devbaseL[streamNum];
    double* devwC[streamNum];
    double* devwR[streamNum];
    double* devwT[streamNum];

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
            allocDeviceParamBuffer(&devDvp[i + baseS], BATCH_SIZE * BATCH_SIZE * nT);
            allocDeviceParamBuffer(&devbaseL[i + baseS], BATCH_SIZE);
            allocDeviceParamBuffer(&devwC[i + baseS], BATCH_SIZE * nK);
            allocDeviceParamBuffer(&devwR[i + baseS], BATCH_SIZE * nR);
            allocDeviceParamBuffer(&devwT[i + baseS], BATCH_SIZE * nT);
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
                            batchSize * npxl * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy ctfP to device.");
            
            cudaMemcpyAsync(devsigP[smidx + baseS],
                            sigRcpP + i * npxl,
                            batchSize * npxl * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy sigP to device.");
                
            cudaMemsetAsync(devbaseL[smidx + baseS],
                            0.0,
                            batchSize * sizeof(double),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset baseL.");
        
            cudaMemsetAsync(devwC[smidx + baseS],
                            0.0,
                            batchSize * nK * sizeof(double),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset wC.");
        
            cudaMemsetAsync(devwR[smidx + baseS],
                            0.0,
                            batchSize * nR * sizeof(double),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset wR.");
        
            cudaMemsetAsync(devwT[smidx + baseS],
                            0.0,
                            batchSize * nT * sizeof(double),
                            stream[smidx + baseS]);
            //cudaCheckErrors("for memset wT.");
        
            for (int r = 0; r < nR;)
            {
                rbatch = (r + BATCH_SIZE < nR) ? BATCH_SIZE : (nR - r);
            
                kernel_Project3D<<<rbatch,
                                   512,
                                   0,
                                   stream[smidx + baseS]>>>(priRotP[smidx + baseS],
                                                            devRotm[n],
                                                            deviCol[n],
                                                            deviRow[n],
                                                            r,
                                                            pf,
                                                            vdim,
                                                            npxl,
                                                            interp,
                                                            texObject[n]);

                kernel_logDataVS<<<rbatch * batchSize * nT, 
                                   512, 
                                   512 * sizeof(double), 
                                   stream[smidx + baseS]>>>(devdatP[smidx + baseS],
                                                            priRotP[smidx + baseS],
                                                            devtraP[n],
                                                            devctfP[smidx + baseS],
                                                            devsigP[smidx + baseS],
                                                            devDvp[smidx + baseS],
                                                            nT,
                                                            rbatch,
                                                            npxl);

                kernel_UpdateW3D<<<1,
                                   batchSize,
                                   0,
                                   stream[smidx + baseS]>>>(devDvp[smidx + baseS],
                                                            devbaseL[smidx + baseS],
                                                            devwC[smidx + baseS],
                                                            devwR[smidx + baseS],
                                                            devwT[smidx + baseS],
                                                            r,
                                                            nK,
                                                            nR,
                                                            nT,
                                                            rbatch * nT);
                    
                r += rbatch; 
            }
            
            cudaMemcpyAsync(wC + i * nK,
                            devwC[smidx + baseS],
                            batchSize * nK * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy wC to host.");
                
            cudaMemcpyAsync(wR + i * nR,
                            devwR[smidx + baseS],
                            batchSize * nR * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            stream[smidx + baseS]);
            //cudaCheckErrors("memcpy wR to host.");
            
            cudaMemcpyAsync(wT + i * nT,
                            devwT[smidx + baseS],
                            batchSize * nT * sizeof(double),
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
        }
    } 
    
    //free device buffers 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        
        cudaFree(devnR[n]);
        cudaFree(deviCol[n]); 
        cudaFree(deviRow[n]); 
        cudaFree(devtraP[n]); 
        cudaFree(devRotm[n]); 
        cudaFreeArray(symArray[n]);
        
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
 * @brief Insert images into volume.
 *
 * @param
 * @param
 */
void InsertF(Complex *F3D,
             double *T3D,
             MPI_Comm& hemi,
             Complex *datP,
             double *ctfP,
             double *sigRcpP,
             CTFAttr *ctfaData,
             double *offS,
             double *w,
             double *nR,
             double *nT,
             double *nD,
             const int *iCol,
             const int *iRow,
             double pixelSize,
             bool cSearch,
             int npxl,
             int rSize,
             int tSize,
             int dSize,
             int mReco,
             int imgNum,
             int idim,
             int vdim)
{
    double pixel = pixelSize * idim;
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
    MPI_Comm_size(hemi, &nsize);
    MPI_Comm_rank(hemi, &nranks);
    
    // NCCL Communicator creation
    if (nranks == 0)
        NCCLCHECK(ncclGetUniqueId(&commId));
    MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, hemi);
    
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
    double *devctfP[streamNum];
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
    double *devDataT[aviDevs];
    int *deviCol[aviDevs];
    int *deviRow[aviDevs];

#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    double *devsigRcpP[streamNum];

    cudaHostRegister(sigRcpP, imgNum * npxl * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register sigRcpP data.");
#endif
    //register pglk_memory
    cudaHostRegister(F3D, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register F3D data.");
    
    cudaHostRegister(T3D, dimSize * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register T3D data.");
    
    cudaHostRegister(datP, imgNum * npxl * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register datP data.");

    cudaHostRegister(ctfP, imgNum * npxl * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register ctfP data.");

    cudaHostRegister(offS, imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register offset data.");

    cudaHostRegister(w, imgNum * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register w data.");
    
    cudaHostRegister(nR, rSize * imgNum * 4 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register nR data.");
    
    cudaHostRegister(nT, tSize * imgNum * 2 * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register nT data.");
    
    cudaHostRegister(nD, dSize * imgNum * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register nD data.");
    
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

        cudaMalloc((void**)&devDataT[n], dimSize * sizeof(double));
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
                allocDeviceParamBuffer(&dev_nd_buf[i + baseS], BATCH_SIZE * dSize);
                allocDeviceParamBuffer(&dev_ramD_buf[i + baseS], BATCH_SIZE * mReco);
            }
            
            allocDeviceComplexBuffer(&devdatP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceComplexBuffer(&devtranP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&devctfP[i + baseS], BATCH_SIZE * npxl);
            allocDeviceParamBuffer(&dev_offs_buf[i + baseS], BATCH_SIZE * 2);
            allocDeviceParamBuffer(&dev_nr_buf[i + baseS], BATCH_SIZE * rSize * 4);
            allocDeviceParamBuffer(&dev_nt_buf[i + baseS], BATCH_SIZE * tSize * 2);
            allocDeviceParamBuffer(&dev_ramR_buf[i + baseS], BATCH_SIZE * mReco * 4);
            allocDeviceParamBuffer(&dev_mat_buf[i + baseS], BATCH_SIZE * mReco * 9);
            allocDeviceParamBuffer(&dev_tran_buf[i + baseS], BATCH_SIZE * mReco * 2);
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
                    dimSize * sizeof(double),
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
                        dimSize * sizeof(double),
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
                                batchSize * npxl * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[smidx + baseS]);
                //cudaCheckErrors("memcpy ctf to device.");
            } 
            
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
            cudaMemcpyAsync(devsigRcpP[n],
                            sigRcpP + i * npxl,
                            batchSize * npxl * sizeof(double),
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
                                    batchSize * sizeof(double),
                                    smidx * batchSize * sizeof(double),
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
    cudaHostUnregister(nD);
   
#ifdef OPTIMISER_RECONSTRUCT_SIGMA_REGULARISE
    cudaHostUnregister(sigRcpP);
#endif

    LOG(INFO) << "Insert done.";
    //printf("Insert done.\n");

    LOG(INFO) << "rank" << nranks << ": Step2: Reduce Volume.";
    //printf("rank%d: Step2: Reduce Volume.\n", nranks);
    
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
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
    
    cudaSetDevice(gpus[0]);
    cudaMemcpyAsync(F3D,
                    devDataF[0],
                    dimSize * sizeof(Complex),
                    cudaMemcpyDeviceToHost,
                    stream[0]);
    //cudaCheckErrors("copy F3D from device to host.");
        
    NCCLCHECK(ncclGroupStart()); 
    for (int i = 0; i < aviDevs; i++) 
    { 
        cudaSetDevice(gpus[i]); 
        NCCLCHECK(ncclReduce((const void*)devDataT[i], 
                             (void*)devDataT[0], 
                             dimSize, 
                             ncclDouble, 
                             ncclSum,
                             0, 
                             comm[i], 
                             stream[1 + i * NUM_STREAM]));
    } 
    NCCLCHECK(ncclGroupEnd());
    
    cudaSetDevice(gpus[0]);
    cudaMemcpyAsync(T3D,
                    devDataT[0],
                    dimSize * sizeof(double),
                    cudaMemcpyDeviceToHost,
                    stream[1]);
    //cudaCheckErrors("copy T3D from device to host.");
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation 
    for (int n = 0; n < aviDevs; ++n) 
    { 
        baseS = n * NUM_STREAM;
        cudaSetDevice(gpus[n]);
        for (int i = 0; i < NUM_STREAM; i++)
            cudaStreamSynchronize(stream[i + baseS]); 
    } 
    
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
void PrepareT(double *T3D,
              const int dim,
              double *FSC,
              const int fscMatsize,
              const double *symMat,
              const int nSymmetryElement,
              const int interp,
              const bool joinHalf,
              const int maxRadius,
              const int pf,
              const int wienerF,
              const double sf)
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
            cudaSetDevice(n);
            break; 
        }
    }

    int dimSize = (dim / 2 + 1) * dim * dim;
    int symMatsize = nSymmetryElement * 9;

    cudaHostRegister(T3D, dimSize * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register T3D data.");
    
    LOG(INFO) << "Step1: NormalizeT.";
    
    double *devDataT;
    cudaMalloc((void**)&devDataT, dimSize * sizeof(double));
    //cudaCheckErrors("Allocate devDataT data.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    double *devSymmat;
    cudaMalloc((void**)&devSymmat, symMatsize * sizeof(double));
    //cudaCheckErrors("Allocate devSymmat data.");
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
        cudaMemcpyAsync(devDataT + shift,
                        T3D + shift,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeT<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                      batchSize, 
                                                      shift, 
                                                      sf);
        //cudaCheckErrors("normalT for stream 0.");
#endif

        cudaMemcpyAsync(devDataT + shift + batchSize,
                        T3D + shift + batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("for memcpy 1.");

#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeT<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                      batchSize, 
                                                      shift + batchSize, 
                                                      sf);
        //cudaCheckErrors("normalT for stream 1.");
#endif
        
        cudaMemcpyAsync(devDataT + shift + 2 * batchSize,
                        T3D + shift + 2 * batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[2]);
        //cudaCheckErrors("for memcpy 2.");

#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeT<<<dim, dim, 0, stream[2]>>>(devDataT, 
                                                      batchSize, 
                                                      shift + 2 * batchSize, 
                                                      sf);
        //cudaCheckErrors("normalT for stream 2.");
#endif
    }
   
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaMemcpyAsync(devSymmat, symMat, symMatsize * sizeof(double), cudaMemcpyHostToDevice, stream[0]);
    //cudaCheckErrors("copy symmat for memcpy 0.");
#endif

    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
        cudaMemcpyAsync(devDataT + shift,
                        T3D + shift,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("out for memcpy 0.");
        
#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeT<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                      batchSize, 
                                                      shift, 
                                                      sf);
        //cudaCheckErrors("normalT last for stream 0.");
#endif
        
        cudaMemcpyAsync(devDataT + shift + batchSize,
                        T3D + shift + batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("out for memcpy 1.");
#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeT<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                      batchSize, 
                                                      shift + batchSize, 
                                                      sf);
        //cudaCheckErrors("normalT last for stream 1.");
#endif
    
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
            cudaMemcpyAsync(devDataT + shift,
                            T3D + shift,
                            (dimSize - shift) * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[2]);
            //cudaCheckErrors("out for memcpy 2.");
            
#ifdef RECONSTRUCTOR_NORMALISE_T_F
            kernel_NormalizeT<<<dim, dim, 0, stream[2]>>>(devDataT, 
                                                          dimSize - shift, 
                                                          shift, 
                                                          sf);
            //cudaCheckErrors("normalT last for stream 2.");
#endif
  
        }
    }
    else
    {
        
        if (len % 3 == 1)
        {
            int shift = (len - 1) * batchSize;
            cudaMemcpyAsync(devDataT + shift,
                            T3D + shift,
                            batchSize * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[0]);
            //cudaCheckErrors("out for memcpy 0.");
            
#ifdef RECONSTRUCTOR_NORMALISE_T_F
            kernel_NormalizeT<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                          batchSize, 
                                                          shift, 
                                                          sf);
            //cudaCheckErrors("normalT last for stream 0.");
#endif
        
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDataT + shift,
                                T3D + shift,
                                (dimSize - shift) * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[1]);
                //cudaCheckErrors("out for memcpy 1.");
                
#ifdef RECONSTRUCTOR_NORMALISE_T_F
                kernel_NormalizeT<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                              dimSize - shift, 
                                                              shift, 
                                                              sf);
                //cudaCheckErrors("normalT last for stream 1.");
#endif
  
            }
        }
        else
        {
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDataT + shift,
                                T3D + shift,
                                (dimSize - shift) * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[0]);
                //cudaCheckErrors("out for memcpy 0.");
                
#ifdef RECONSTRUCTOR_NORMALISE_T_F
                kernel_NormalizeT<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                              dimSize - shift, 
                                                              shift, 
                                                              sf);
                //cudaCheckErrors("normalT last for stream 0.");
#endif
          
           }
        }
    }

    cudaStreamSynchronize(stream[0]);
    //cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    //cudaCheckErrors("CUDA stream1 synchronization.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc(32, 32, 0, 0,cudaChannelFormatKindSigned);
    cudaArray *symArray; 
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);
    cudaMalloc3DArray(&symArray, &channelDesc, extent);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)devDataT, (dim / 2 + 1) * sizeof(int2), dim / 2 + 1, dim);
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);
    //cudaCheckErrors("memcpy error");
    
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
#endif

    LOG(INFO) << "Step2: SymmetrizeT.";
    
    int vecSize = maxRadius * pf + 1;
    int streamSym = len / 2;
    
    for (int i = 0; i < streamSym; i++)
    {
        int shift = i * 2 * batchSize;

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        kernel_SymmetrizeT<<<dim, dim, 0, stream[0]>>>(devDataT,
                                                       devSymmat, 
                                                       nSymmetryElement, 
                                                       (double)vecSize, 
                                                       interp, 
                                                       shift,
                                                       dim,
                                                       batchSize,
                                                       texObject);
        //cudaCheckErrors("symT for stream 0");
#endif

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        kernel_SymmetrizeT<<<dim, dim, 0, stream[1]>>>(devDataT,
                                                       devSymmat, 
                                                       nSymmetryElement, 
                                                       (double)vecSize, 
                                                       interp, 
                                                       shift + batchSize,
                                                       dim,
                                                       batchSize,
                                                       texObject);
        //cudaCheckErrors("symT for stream 1");
#endif

    }
   
    if (len % 2 == 1)
    {
        int shift = (len - 1) * batchSize;
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        kernel_SymmetrizeT<<<dim, dim, 0, stream[0]>>>(devDataT,
                                                       devSymmat, 
                                                       nSymmetryElement, 
                                                       (double)vecSize, 
                                                       interp, 
                                                       shift,
                                                       dim,
                                                       batchSize,
                                                       texObject);
        //cudaCheckErrors("symT last for stream 0");
#endif
    
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
            kernel_SymmetrizeT<<<dim, dim, 0, stream[1]>>>(devDataT,
                                                           devSymmat, 
                                                           nSymmetryElement, 
                                                           (double)vecSize, 
                                                           interp, 
                                                           shift,
                                                           dim,
                                                           dimSize - shift,
                                                           texObject);
            //cudaCheckErrors("symT last for stream 1");
#endif
        }

    }
    else 
    {
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
            kernel_SymmetrizeT<<<dim, dim, 0, stream[0]>>>(devDataT,
                                                           devSymmat, 
                                                           nSymmetryElement, 
                                                           (double)vecSize, 
                                                           interp, 
                                                           shift,
                                                           dim,
                                                           dimSize - shift,
                                                           texObject);
            //cudaCheckErrors("symT last for stream 0");
#endif
        }
    }

    cudaStreamSynchronize(stream[0]);
    //cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    //cudaCheckErrors("CUDA stream2 synchronization.");

    cudaDestroyTextureObject(texObject);

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
    double *devAvg;

    cudaMalloc((void**)&devAvg, vecSize * sizeof(double));
    //cudaCheckErrors("Allocate devAvg data.");
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    LOG(INFO) << "Step3: ShellAverage.";
    
    double *devAvg2D;
    int *devCount2D;
    int *devCount;

    cudaMalloc((void**)&devAvg2D, (vecSize - 2) * dim * sizeof(double));
    //cudaCheckErrors("Allocate devAvg data.");

    cudaMalloc((void**)&devCount2D, (vecSize - 2) * dim * sizeof(int));
    //cudaCheckErrors("Allocate devCount data.");

    cudaMalloc((void**)&devCount, vecSize * sizeof(int));
    //cudaCheckErrors("Allocate devCount data.");
    
    kernel_ShellAverage<<<dim, 
                          dim, 
                          dim * (sizeof(double) 
                                + sizeof(int))>>>(devAvg2D, 
                                                  devCount2D, 
                                                  devDataT,
                                                  dim, 
                                                  vecSize - 2,
                                                  dimSize);
    //cudaCheckErrors("Shell for stream default.");
    
    kernel_CalculateAvg<<<1, vecSize - 2>>>(devAvg2D,
                                            devCount2D,
                                            devAvg,
                                            devCount,
                                            dim,
                                            vecSize - 2);
    //cudaCheckErrors("calAvg for stream default.");
#endif

#endif

    LOG(INFO) << "Step4: Calculate WIENER_FILTER.";
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
    double *devFSC;
    
    cudaMalloc((void**)&devFSC, fscMatsize * sizeof(double));
    //cudaCheckErrors("Allocate devFSC data.");
    cudaMemcpy(devFSC, FSC, fscMatsize * sizeof(double), cudaMemcpyHostToDevice);
    //cudaCheckErrors("Copy FSC to device.");

#endif

    int wiener = pow(wienerF * pf, 2);
    int r = pow(maxRadius * pf, 2);
    int streamfsc = len / 3;
    
    for (int i = 0; i < streamfsc; i++)
    {
        int shift = i * 3 * batchSize;
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
        //cudaCheckErrors("calFSC for stream 0.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
        kernel_WienerConst<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                       wiener, 
                                                       r, 
                                                       shift,
                                                       dim,
                                                       dimSize);
#endif 
        
        cudaMemcpyAsync(T3D + shift,
                        devDataT + shift,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("out for memcpy 0.");
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
        //cudaCheckErrors("calFSC for stream 1.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
        kernel_WienerConst<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                       wiener, 
                                                       r, 
                                                       shift + batchSize,
                                                       dim,
                                                       dimSize);
#endif 
        
        cudaMemcpyAsync(T3D + shift + batchSize,
                        devDataT + shift + batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("out for memcpy 1.");

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
        //cudaCheckErrors("calFSC for stream 2.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
        kernel_WienerConst<<<dim, dim, 0, stream[2]>>>(devDataT, 
                                                       wiener, 
                                                       r, 
                                                       shift + 2 * batchSize,
                                                       dim,
                                                       dimSize);
#endif 
        
        cudaMemcpyAsync(T3D + shift + 2 * batchSize,
                        devDataT + shift + 2 * batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[2]);
        //cudaCheckErrors("out for memcpy 2.");
    }
   
    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
        //cudaCheckErrors("calFSC last for stream 0.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
        kernel_WienerConst<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                       wiener, 
                                                       r, 
                                                       shift,
                                                       dim,
                                                       dimSize);
#endif 
        
        cudaMemcpyAsync(T3D + shift,
                        devDataT + shift,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("out for memcpy 0.");

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
        //cudaCheckErrors("calFSC last for stream 1.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
        kernel_WienerConst<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                       wiener, 
                                                       r, 
                                                       shift + batchSize,
                                                       dim,
                                                       dimSize);
#endif 
        
        cudaMemcpyAsync(T3D + shift + batchSize,
                        devDataT + shift + batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("out for memcpy 1.");
        
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
            //cudaCheckErrors("calFSC last for stream 2.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
            kernel_WienerConst<<<dim, dim, 0, stream[2]>>>(devDataT, 
                                                           wiener, 
                                                           r, 
                                                           shift,
                                                           dim,
                                                           dimSize - shift);
#endif 
        
            cudaMemcpyAsync(T3D + shift,
                            devDataT + shift,
                            (dimSize - shift) * sizeof(double),
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
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
            //cudaCheckErrors("calFSC last for stream 0.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
            kernel_WienerConst<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                           wiener, 
                                                           r, 
                                                           shift,
                                                           dim,
                                                           dimSize);
#endif 
        
            cudaMemcpyAsync(T3D + shift,
                            devDataT + shift,
                            batchSize * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            stream[0]);
            //cudaCheckErrors("out for memcpy 0.");
            
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
                //cudaCheckErrors("calFSC last for stream 1.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
                kernel_WienerConst<<<dim, dim, 0, stream[1]>>>(devDataT, 
                                                               wiener, 
                                                               r, 
                                                               shift,
                                                               dim,
                                                               dimSize - shift);
#endif 
        
                cudaMemcpyAsync(T3D + shift,
                                devDataT + shift,
                                (dimSize - shift) * sizeof(double),
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
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
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
                //cudaCheckErrors("calFSC last for stream 0.");
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_CONST
                kernel_WienerConst<<<dim, dim, 0, stream[0]>>>(devDataT, 
                                                               wiener, 
                                                               r, 
                                                               shift,
                                                               dim,
                                                               dimSize - shift);
#endif 
        
                cudaMemcpyAsync(T3D + shift,
                                devDataT + shift,
                                (dimSize - shift) * sizeof(double),
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
    //cudaCheckErrors("CUDA stream2 synchronization.");
    
    cudaHostUnregister(T3D);
    
    LOG(INFO) << "Step6: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    //cudaCheckErrors("Destroy stream.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaFree(devSymmat);
    //cudaCheckErrors("Free device memory devSymmat.");
    
    cudaFreeArray(symArray);
    //cudaCheckErrors("Free device memory SymArray.");

    //cudaFree(devSym);
    ////cudaCheckErrors("Free device memory devSym.");

#endif    
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
    cudaFree(devAvg);
    //cudaCheckErrors("Free device memory devAvg.");

    cudaFree(devFSC);
    //cudaCheckErrors("Free device memory devFSC.");
    
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
    cudaFree(devAvg2D);
    //cudaCheckErrors("Free device memory devAvg2D.");

    cudaFree(devCount2D);
    //cudaCheckErrors("Free device memory devCount2D.");
    
    cudaFree(devCount);
    //cudaCheckErrors("Free device memory devCount.");

#endif

#endif
    cudaFree(devDataT);
    //cudaCheckErrors("Free device memory devDataT.");

}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW(Complex *C3D,
                double *T3D,
                double *W3D,
                double *tabdata,
                double begin,
                double end,
                double step,
                int tabsize, 
                const int dim,
                const int r,
                const double nf,
                const int maxIter,
                const int minIter,
                const int padSize)
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
            cudaSetDevice(n);
            break; 
        }
    }

    int dimSize = (dim / 2 + 1) * dim * dim;
    int dimSizeRL = dim * dim * dim;
    
    LOG(INFO) << "Step1: InitialW.";
    
    double *devDataW;
    cudaMalloc((void**)&devDataW, dimSize * sizeof(double));
    //cudaCheckErrors("Allocate devDataW data.");
    
    kernel_InitialW<<<dim, dim>>>(devDataW,  
                                  r, 
                                  dim,
                                  dimSize);
    
    
    LOG(INFO) << "Step2: Calculate C.";

    /* Upload tabfunction to device */
    TabFunction tabfunc(begin, end, step, NULL, tabsize);

    uploadTabFunction(tabfunc, tabdata);

    cufftDoubleComplex *devDataC;
    cudaMalloc((void**)&devDataC, dimSize * sizeof(cufftDoubleComplex));
    //cudaCheckErrors("Allocate device memory for T.");

    cufftDoubleReal *devDoubleC;
    cudaMalloc((void**)&devDoubleC, dim * dim * dim  * sizeof(cufftDoubleReal));
    //cudaCheckErrors("Allocate device memory for T.");
    
    double *devDataT;
    cudaMalloc((void**)&devDataT, dimSize *sizeof(double));
    //cudaCheckErrors("Allocate device memory for T.");
    cudaMemcpy(devDataT, T3D, dimSize * sizeof(double), cudaMemcpyHostToDevice);
    //cudaCheckErrors("Copy devDataT volume to device.");

    double diffC = DBL_MAX;
    double diffCPrev = DBL_MAX;

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
    double *diff = new double[dim];
    double *counter = new double[dim];
    
    double *devDiff;
    double *devCount;
    cudaMalloc((void**)&devDiff, dim *sizeof(double));
    //cudaCheckErrors("Allocate device memory for devDiff.");
    cudaMalloc((void**)&devCount, dim *sizeof(double));
    //cudaCheckErrors("Allocate device memory for devCount.");
        
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX        
    double *cmax = new double[dim];
    
    double *devMax;
    cudaMalloc((void**)&devMax, dim *sizeof(double));
    //cudaCheckErrors("Allocate device memory for devMax.");
#endif
    
    cufftHandle planc2r, planr2c;
    cufftPlan3d(&planc2r, dim, dim, dim, CUFFT_Z2D);
    cufftPlan3d(&planr2c, dim, dim, dim, CUFFT_D2Z);
   
    int nDiffCNoDecrease = 0;

    for(int m = 0; m < maxIter; m++)
    {
        //LOG(INFO) << "SubStep1: Determining C.";
        kernel_DeterminingC<<<dim, dim>>>((Complex*)devDataC,
                                          devDataT, 
                                          devDataW, 
                                          dimSize);

        //LOG(INFO) << "SubStep2: Convoluting C.";

        cufftExecZ2D(planc2r, devDataC, devDoubleC);

        kernel_convoluteC<<<dim, dim>>>(devDoubleC,
                                        tabfunc,
                                        nf,
                                        padSize,
                                        dim,
                                        dimSizeRL);
        
        cufftExecD2Z(planr2c, devDoubleC, devDataC);
       
        kernel_RecalculateW<<<dim, dim>>>(devDataW,
                                          (Complex*)devDataC,  
                                          r, 
                                          dim,
                                          dimSize);

        diffCPrev = diffC;
        
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
        kernel_CheckCAVG<<<dim, 
                           dim, 
                           2 * dim * sizeof(double)>>>(devDiff,
                                                       devCount,
                                                       (Complex*)devDataC,  
                                                       r, 
                                                       dim,
                                                       dimSize);
        
        cudaMemcpy(diff, devDiff, dim *sizeof(double), cudaMemcpyDeviceToHost);
        //cudaCheckErrors("Copy devDiff array to host.");
        cudaMemcpy(counter, devCount, dim *sizeof(double), cudaMemcpyDeviceToHost);
        //cudaCheckErrors("Copy devCount array to host.");
        
        double tempD, tempC;
        for(int i = 0;i < dim;i++)
        {
            tempD = diff[i];
            tempC = counter[i];
        }
        diffC = tempD / tempC;

#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
        kernel_CheckCMAX<<<dim, 
                           dim, 
                           dim * sizeof(double)>>>(devMax,
                                                   (Complex*)devDataC,  
                                                   r, 
                                                   dim,
                                                   dimSize);
        
        cudaMemcpy(cmax, devMax, dim * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaCheckErrors("Copy devMax array to host.");
        
        double temp = 0.0;
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

    cudaMemcpy(W3D, devDataW, dimSize *sizeof(double), cudaMemcpyDeviceToHost);
    //cudaCheckErrors("Copy devDataW volume to host.");  
    
    LOG(INFO) << "Step3: Clean up the streams and memory.";

#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
    delete[] diff;
    delete[] counter;

    cudaFree(devDiff);
    //cudaCheckErrors("Free device memory devDiff.");
    
    cudaFree(devCount);
    //cudaCheckErrors("Free device memory devCount.");
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX        
    delete[] cmax;

    cudaFree(devMax);
    //cudaCheckErrors("Free device memory devMax.");
#endif

    cudaFree(devDataW);
    //cudaCheckErrors("Free device memory devDataW.");
    
    cudaFree(devDataC);
    //cudaCheckErrors("Free device memory devDataC.");
    
    cudaFree(devDoubleC);
    //cudaCheckErrors("Free device memory devDoubleC.");

    cufftDestroy(planc2r);
    //cudaCheckErrors("DestroyPlan planc2r.");

    cufftDestroy(planr2c);
    //cudaCheckErrors("DestroyPlan planr2c.");
    
    cudaFree(devDataT);
    //cudaCheckErrors("Free device memory devDataT.");
    
    cudaFree(tabfunc.devPtr());
    //cudaCheckErrors("Free operations.");
}

/**
 * @brief ...
 *
 * @param ..
 * @param ..
 */
void CalculateW(double *T3D,
                double *W3D,
                const int dim,
                const int r)
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
            cudaSetDevice(n);
            break; 
        }
    }

    int dimSize = (dim / 2 + 1) * dim * dim;
    
    Constructor construct;

    cudaHostRegister(T3D, dimSize * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register T3D data.");

    cudaHostRegister(W3D, dimSize * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register W3D data.");
    
    double *devDataW;
    cudaMalloc((void**)&devDataW, dimSize * sizeof(double));
    //cudaCheckErrors("Allocate devDataW data.");
    
    double *devDataT;
    cudaMalloc((void**)&devDataT, dimSize *sizeof(double));
    //cudaCheckErrors("Allocate device memory for T.");

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
                        batchSize * sizeof(double), 
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("Copy devDataT volume to device stream0.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[0]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift,
                                                      dim,
                                                      r);

        cudaMemcpyAsync(W3D + shift, 
                        devDataW + shift, 
                        batchSize * sizeof(double), 
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("Copy devDataW volume to host stream0.");  
    
        cudaMemcpyAsync(devDataT + shift + batchSize, 
                        T3D + shift + batchSize, 
                        batchSize * sizeof(double), 
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("Copy devDataT volume to device stream1.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[1]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift + batchSize,
                                                      dim,
                                                      r);

        cudaMemcpyAsync(W3D + shift + batchSize, 
                        devDataW + shift + batchSize, 
                        batchSize * sizeof(double), 
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("Copy devDataW volume to host stream1.");  
    
        cudaMemcpyAsync(devDataT + shift + 2 * batchSize, 
                        T3D + shift + 2 * batchSize, 
                        batchSize * sizeof(double), 
                        cudaMemcpyHostToDevice,
                        stream[2]);
        //cudaCheckErrors("Copy devDataT volume to device stream2.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[2]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift + 2 * batchSize,
                                                      dim,
                                                      r);

        cudaMemcpyAsync(W3D + shift + 2 * batchSize, 
                        devDataW + shift + 2 * batchSize, 
                        batchSize * sizeof(double), 
                        cudaMemcpyDeviceToHost,
                        stream[2]);
        //cudaCheckErrors("Copy devDataW volume to host stream2.");  
    
    }

    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
        cudaMemcpyAsync(devDataT + shift, 
                        T3D + shift, 
                        batchSize * sizeof(double), 
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("Copy devDataT volume to device stream0.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[0]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift,
                                                      dim,
                                                      r);

        cudaMemcpyAsync(W3D + shift, 
                        devDataW + shift, 
                        batchSize * sizeof(double), 
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("Copy devDataW volume to host stream0.");  
        
        cudaMemcpyAsync(devDataT + shift + batchSize, 
                        T3D + shift + batchSize, 
                        batchSize * sizeof(double), 
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("Copy devDataT volume to device stream1.");
        
        kernel_CalculateW<<<dim, dim, 0, stream[1]>>>(devDataW,
                                                      devDataT, 
                                                      batchSize, 
                                                      shift + batchSize,
                                                      dim,
                                                      r);


        cudaMemcpyAsync(W3D + shift + batchSize, 
                        devDataW + shift + batchSize, 
                        batchSize * sizeof(double), 
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("Copy devDataW volume to host stream1.");  
    
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
            cudaMemcpyAsync(devDataT + shift, 
                            T3D + shift, 
                            (dimSize - shift) * sizeof(double), 
                            cudaMemcpyHostToDevice,
                            stream[2]);
            //cudaCheckErrors("Copy devDataT volume to device stream2.");
            
            kernel_CalculateW<<<dim, dim, 0, stream[2]>>>(devDataW,
                                                          devDataT, 
                                                          dimSize - shift, 
                                                          shift,
                                                          dim,
                                                          r);

            cudaMemcpyAsync(W3D + shift, 
                            devDataW + shift, 
                            (dimSize - shift) * sizeof(double), 
                            cudaMemcpyDeviceToHost,
                            stream[2]);
            //cudaCheckErrors("Copy devDataW volume to host stream2.");
        }
    }
    else
    {
        
        if (len % 3 == 1)
        {
            int shift = (len - 1) * batchSize;
            cudaMemcpyAsync(devDataT + shift, 
                            T3D + shift, 
                            batchSize * sizeof(double), 
                            cudaMemcpyHostToDevice,
                            stream[0]);
            //cudaCheckErrors("Copy devDataT volume to device stream0.");
            
            kernel_CalculateW<<<dim, dim, 0, stream[0]>>>(devDataW,
                                                          devDataT, 
                                                          batchSize, 
                                                          shift,
                                                          dim,
                                                          r);

            cudaMemcpyAsync(W3D + shift, 
                            devDataW + shift, 
                            batchSize * sizeof(double), 
                            cudaMemcpyDeviceToHost,
                            stream[0]);
            //cudaCheckErrors("Copy devDataW volume to host stream0.");  
        
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDataT + shift, 
                                T3D + shift, 
                                (dimSize - shift) * sizeof(double), 
                                cudaMemcpyHostToDevice,
                                stream[1]);
                //cudaCheckErrors("Copy devDataT volume to device stream1.");
                
                kernel_CalculateW<<<dim, dim, 0, stream[1]>>>(devDataW,
                                                              devDataT, 
                                                              dimSize - shift, 
                                                              shift,
                                                              dim,
                                                              r);
                cudaMemcpyAsync(W3D + shift, 
                                devDataW + shift, 
                                (dimSize - shift) * sizeof(double), 
                                cudaMemcpyDeviceToHost,
                                stream[1]);
                //cudaCheckErrors("Copy devDataW volume to host stream1.");
            }
        }
        else
        {
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDataT + shift, 
                                T3D + shift, 
                                (dimSize - shift) * sizeof(double), 
                                cudaMemcpyHostToDevice,
                                stream[0]);
                //cudaCheckErrors("Copy devDataT volume to device stream0.");
                
                kernel_CalculateW<<<dim, dim, 0, stream[0]>>>(devDataW,
                                                              devDataT, 
                                                              dimSize - shift, 
                                                              shift,
                                                              dim,
                                                              r);
                cudaMemcpyAsync(W3D + shift, 
                                devDataW + shift, 
                                (dimSize - shift) * sizeof(double), 
                                cudaMemcpyDeviceToHost,
                                stream[0]);
                //cudaCheckErrors("Copy devDataW volume to host stream0.");
           }
        }
    }
    

    cudaStreamSynchronize(stream[0]);
    //cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    
    cudaHostUnregister(T3D);
    cudaHostUnregister(W3D);
    
    LOG(INFO) << "Step3: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    //cudaCheckErrors("Destroy stream.");

    cudaFree(devDataW);
    //cudaCheckErrors("Free device memory devDataW.");
    
    cudaFree(devDataT);
    //cudaCheckErrors("Free device memory devDataT.");
}

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void PrepareF(Complex *F3D,
              double *W3D,
              const double sf,
              const int nSymmetryElement,
              const double *symMat,
              const int interp,
              const int maxRadius,
              const int edgeWidth,
              const int pf,
              const int dim,
              const int size)
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
            cudaSetDevice(n);
            break; 
        }
    }

    int dimSize = (dim / 2 + 1) * dim * dim;
    int symMatsize = nSymmetryElement * 9;

    cudaHostRegister(F3D, dimSize * sizeof(Complex), cudaHostRegisterDefault);
    //cudaCheckErrors("Register F3D data.");

    cudaHostRegister(W3D, dimSize * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register W3D data.");
    
    Complex *devDataF;
    cudaMalloc((void**)&devDataF, dimSize * sizeof(Complex));
    //cudaCheckErrors("Allocate devDataF data.");

    double *devPartW;
    cudaMalloc((void**)&devPartW, dimSize *sizeof(double));
    //cudaCheckErrors("Allocate device memory for Part W.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    double *devSymmat;
    cudaMalloc((void**)&devSymmat, symMatsize * sizeof(double));
    //cudaCheckErrors("Allocate devSymmat data.");
#endif    
    
#ifdef RECONSTRUCTOR_LOW_PASS    
    double thres = (double)(maxRadius - edgeWidth) / size;
    double ew = (double)edgeWidth / size;
#endif    
    
    cudaStream_t stream[3];

    for(int i = 0; i < 3; i++)
        cudaStreamCreate(&stream[i]);
    
    LOG(INFO) << "Step1: NormalizeF.";
    
    int batchSize = 8 * dim * dim;
    int len = dimSize / batchSize;
    int streamN = len / 3;
   
    for (int i = 0; i < streamN; i++)
    {
        int shift = i * 3 * batchSize;
        cudaMemcpyAsync(devDataF + shift,
                        F3D + shift,
                        batchSize * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeF<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                      batchSize, 
                                                      shift, 
                                                      sf);
#endif

        cudaMemcpyAsync(devDataF + shift + batchSize,
                        F3D + shift + batchSize,
                        batchSize * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("for memcpy 1.");

#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeF<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                      batchSize, 
                                                      shift + batchSize, 
                                                      sf);
#endif
        
        cudaMemcpyAsync(devDataF + shift + 2 * batchSize,
                        F3D + shift + 2 * batchSize,
                        batchSize * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[2]);
        //cudaCheckErrors("for memcpy 2.");

#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeF<<<dim, dim, 0, stream[2]>>>(devDataF, 
                                                      batchSize, 
                                                      shift + 2 * batchSize, 
                                                      sf);
#endif

    }
   
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaMemcpyAsync(devSymmat, 
                    symMat, 
                    symMatsize * sizeof(double), 
                    cudaMemcpyHostToDevice, 
                    stream[2]);
#endif

    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
        cudaMemcpyAsync(devDataF + shift,
                        F3D + shift,
                        batchSize * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("out for memcpy 0.");
        
#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeF<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                      batchSize, 
                                                      shift, 
                                                      sf);
#endif
        
        cudaMemcpyAsync(devDataF + shift + batchSize,
                        F3D + shift + batchSize,
                        batchSize * sizeof(Complex),
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("out for memcpy 1.");
#ifdef RECONSTRUCTOR_NORMALISE_T_F
        kernel_NormalizeF<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                      batchSize, 
                                                      shift + batchSize, 
                                                      sf);
#endif
    
        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
            cudaMemcpyAsync(devDataF + shift,
                            F3D + shift,
                            (dimSize - shift) * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[2]);
            //cudaCheckErrors("out for memcpy 2.");
            
#ifdef RECONSTRUCTOR_NORMALISE_T_F
            kernel_NormalizeF<<<dim, dim, 0, stream[2]>>>(devDataF, 
                                                          dimSize - shift, 
                                                          shift, 
                                                          sf);
#endif
  
        }
    }
    else
    {
        
        if (len % 3 == 1)
        {
            int shift = (len - 1) * batchSize;
            cudaMemcpyAsync(devDataF + shift,
                            F3D + shift,
                            batchSize * sizeof(Complex),
                            cudaMemcpyHostToDevice,
                            stream[0]);
            //cudaCheckErrors("out for memcpy 0.");
            
#ifdef RECONSTRUCTOR_NORMALISE_T_F
            kernel_NormalizeF<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                          batchSize, 
                                                          shift, 
                                                          sf);
#endif
        
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDataF + shift,
                                F3D + shift,
                                (dimSize - shift) * sizeof(Complex),
                                cudaMemcpyHostToDevice,
                                stream[1]);
                //cudaCheckErrors("out for memcpy 1.");
                
#ifdef RECONSTRUCTOR_NORMALISE_T_F
                kernel_NormalizeF<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                              dimSize - shift, 
                                                              shift, 
                                                              sf);
#endif
  
            }
        }
        else
        {
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDataF + shift,
                                F3D + shift,
                                (dimSize - shift) * sizeof(Complex),
                                cudaMemcpyHostToDevice,
                                stream[0]);
                //cudaCheckErrors("out for memcpy 0.");
                
#ifdef RECONSTRUCTOR_NORMALISE_T_F
                kernel_NormalizeF<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                              dimSize - shift, 
                                                              shift, 
                                                              sf);
#endif
          
           }
        }
    }
    

    cudaStreamSynchronize(stream[0]);
    //cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    
    LOG(INFO) << "Step2: SymmetrizeF.";
    
    int vecSize = maxRadius * pf + 1;

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaChannelFormatDesc channelDesc =cudaCreateChannelDesc(32, 32, 32, 32,cudaChannelFormatKindSigned);
    cudaArray *symArray; 
    cudaExtent extent = make_cudaExtent(dim / 2 + 1, dim, dim);
    cudaMalloc3DArray(&symArray, &channelDesc, extent);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)devDataF, (dim / 2 + 1) * sizeof(int4), dim / 2 + 1, dim);
    copyParams.dstArray = symArray;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams);
    //cudaCheckErrors("memcpy error\n.");
    
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
#endif

    LOG(INFO) << "Step3: CalculateFW.";
    
    LOG(INFO) << "Step4: LowpassFilter F.";

    int streamSym = len / 3;
    
    for (int i = 0; i < streamSym; i++)
    {
        int shift = i * 3 * batchSize;

        cudaMemcpyAsync(devPartW + shift,
                        W3D + shift,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        kernel_SymmetrizeF<<<dim, dim, 0, stream[0]>>>(devDataF,
                                                       devSymmat, 
                                                       nSymmetryElement, 
                                                       (double)vecSize, 
                                                       interp, 
                                                       shift,
                                                       dim,
                                                       batchSize,
                                                       texObject);
#endif
       
        kernel_NormalizeFW<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                       devPartW, 
                                                       batchSize, 
                                                       shift, 
                                                       shift);

#ifdef RECONSTRUCTOR_LOW_PASS    
        kernel_LowpassF<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                    thres,
                                                    ew
                                                    shift,
                                                    dim,
                                                    batchSize);
#endif    
        
        cudaMemcpyAsync(F3D + shift,
                        devDataF + shift,
                        batchSize * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("for memcpy 0.");

        cudaMemcpyAsync(devPartW + shift + batchSize,
                        W3D + shift + batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("for memcpy 1.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        kernel_SymmetrizeF<<<dim, dim, 0, stream[1]>>>(devDataF,
                                                       devSymmat, 
                                                       nSymmetryElement, 
                                                       (double)vecSize, 
                                                       interp, 
                                                       shift + batchSize,
                                                       dim,
                                                       batchSize,
                                                       texObject);
#endif

        kernel_NormalizeFW<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                       devPartW, 
                                                       batchSize, 
                                                       shift + batchSize, 
                                                       shift + batchSize);

#ifdef RECONSTRUCTOR_LOW_PASS    
        kernel_LowpassF<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                    thres,
                                                    ew
                                                    shift + batchSize,
                                                    dim,
                                                    batchSize);
#endif    

        cudaMemcpyAsync(F3D + shift + batchSize, 
                        devDataF + shift + batchSize,
                        batchSize * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("for memcpy 1.");
    
        cudaMemcpyAsync(devPartW + shift + 2 * batchSize,
                        W3D + shift + 2 * batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[2]);
        //cudaCheckErrors("for memcpy 2.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        kernel_SymmetrizeF<<<dim, dim, 0, stream[2]>>>(devDataF,
                                                       devSymmat, 
                                                       nSymmetryElement, 
                                                       (double)vecSize, 
                                                       interp, 
                                                       shift + 2 * batchSize,
                                                       dim,
                                                       batchSize,
                                                       texObject);
#endif

        kernel_NormalizeFW<<<dim, dim, 0, stream[2]>>>(devDataF, 
                                                       devPartW, 
                                                       batchSize, 
                                                       shift + 2 * batchSize, 
                                                       shift + 2 * batchSize);

#ifdef RECONSTRUCTOR_LOW_PASS    
        kernel_LowpassF<<<dim, dim, 0, stream[2]>>>(devDataF, 
                                                    thres,
                                                    ew
                                                    shift + 2 * batchSize,
                                                    dim,
                                                    batchSize);
#endif    

        cudaMemcpyAsync(F3D + shift + 2 * batchSize, 
                        devDataF + shift + 2 * batchSize,
                        batchSize * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[2]);
        //cudaCheckErrors("for memcpy 2.");
    
    }
   
    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
        
        cudaMemcpyAsync(devPartW + shift,
                        W3D + shift,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[0]);
        //cudaCheckErrors("for memcpy 0.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        kernel_SymmetrizeF<<<dim, dim, 0, stream[0]>>>(devDataF,
                                                       devSymmat, 
                                                       nSymmetryElement, 
                                                       (double)vecSize, 
                                                       interp, 
                                                       shift,
                                                       dim,
                                                       batchSize,
                                                       texObject);
#endif

    
        kernel_NormalizeFW<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                       devPartW, 
                                                       batchSize, 
                                                       shift,
                                                       shift);

#ifdef RECONSTRUCTOR_LOW_PASS    
        kernel_LowpassF<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                    thres,
                                                    ew
                                                    shift,
                                                    dim,
                                                    batchSize);
#endif    
        
        cudaMemcpyAsync(F3D + shift, 
                        devDataF + shift,
                        batchSize * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("for memcpy 0.");

        cudaMemcpyAsync(devPartW + shift + batchSize,
                        W3D + shift + batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyHostToDevice,
                        stream[1]);
        //cudaCheckErrors("for memcpy 1.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        kernel_SymmetrizeF<<<dim, dim, 0, stream[1]>>>(devDataF,
                                                       devSymmat, 
                                                       nSymmetryElement, 
                                                       (double)vecSize, 
                                                       interp, 
                                                       shift + batchSize,
                                                       dim,
                                                       batchSize,
                                                       texObject);
#endif

        kernel_NormalizeFW<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                       devPartW, 
                                                       batchSize, 
                                                       shift + batchSize, 
                                                       shift + batchSize);

#ifdef RECONSTRUCTOR_LOW_PASS    
        kernel_LowpassF<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                    thres,
                                                    ew
                                                    shift + batchSize,
                                                    dim,
                                                    batchSize);
#endif    

        cudaMemcpyAsync(F3D + shift + batchSize, 
                        devDataF + shift + batchSize,
                        batchSize * sizeof(Complex),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("for memcpy 1.");

        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
            
            cudaMemcpyAsync(devPartW + shift,
                            W3D + shift,
                            (dimSize - shift) * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[2]);
            //cudaCheckErrors("for memcpy 2.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
            kernel_SymmetrizeF<<<dim, dim, 0, stream[2]>>>(devDataF,
                                                           devSymmat, 
                                                           nSymmetryElement, 
                                                           (double)vecSize, 
                                                           interp, 
                                                           shift,
                                                           dim,
                                                           dimSize - shift,
                                                           texObject);
#endif

            kernel_NormalizeFW<<<dim, dim, 0, stream[2]>>>(devDataF, 
                                                           devPartW, 
                                                           dimSize - shift, 
                                                           shift, 
                                                           shift);

#ifdef RECONSTRUCTOR_LOW_PASS    
            kernel_LowpassF<<<dim, dim, 0, stream[2]>>>(devDataF, 
                                                        thres,
                                                        ew
                                                        shift,
                                                        dim,
                                                        dimSize - shift);
#endif    

            cudaMemcpyAsync(F3D + shift, 
                            devDataF + shift,
                            (dimSize - shift) * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            stream[2]);
            //cudaCheckErrors("for memcpy 2.");
    
        }

    }

    else 
    {
        
        if (len % 2 == 1)
        {
            int shift = (len - 1) * batchSize;
            
            cudaMemcpyAsync(devPartW + shift,
                            W3D + shift,
                            batchSize * sizeof(double),
                            cudaMemcpyHostToDevice,
                            stream[0]);
            //cudaCheckErrors("for memcpy 0.");
    
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
            kernel_SymmetrizeF<<<dim, dim, 0, stream[0]>>>(devDataF,
                                                           devSymmat, 
                                                           nSymmetryElement, 
                                                           (double)vecSize, 
                                                           interp, 
                                                           shift,
                                                           dim,
                                                           batchSize,
                                                           texObject);
#endif

    
            kernel_NormalizeFW<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                           devPartW, 
                                                           batchSize, 
                                                           shift,
                                                           shift);
    
#ifdef RECONSTRUCTOR_LOW_PASS    
            kernel_LowpassF<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                        thres,
                                                        ew
                                                        shift,
                                                        dim,
                                                        batchSize);
#endif    
       
            cudaMemcpyAsync(F3D + shift, 
                            devDataF + shift,
                            batchSize * sizeof(Complex),
                            cudaMemcpyDeviceToHost,
                            stream[0]);
            //cudaCheckErrors("for memcpy 0.");


            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                
                cudaMemcpyAsync(devPartW + shift,
                                W3D + shift,
                                (dimSize - shift) * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[1]);
                //cudaCheckErrors("for memcpy 1.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
                kernel_SymmetrizeF<<<dim, dim, 0, stream[1]>>>(devDataF,
                                                               devSymmat, 
                                                               nSymmetryElement, 
                                                               (double)vecSize, 
                                                               interp, 
                                                               shift,
                                                               dim,
                                                               dimSize - shift,
                                                               texObject);
#endif

    
                kernel_NormalizeFW<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                               devPartW, 
                                                               dimSize - shift, 
                                                               shift, 
                                                               shift);
    
#ifdef RECONSTRUCTOR_LOW_PASS    
                kernel_LowpassF<<<dim, dim, 0, stream[1]>>>(devDataF, 
                                                            thres,
                                                            ew
                                                            shift,
                                                            dim,
                                                            dimSize - shift);
#endif    

                cudaMemcpyAsync(F3D + shift, 
                                devDataF + shift,
                                (dimSize - shift) * sizeof(Complex),
                                cudaMemcpyDeviceToHost,
                                stream[1]);
                //cudaCheckErrors("for memcpy 1.");
        
    
            }

        }
        else
        {
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                
                cudaMemcpyAsync(devPartW + shift,
                                W3D + shift,
                                (dimSize - shift) * sizeof(double),
                                cudaMemcpyHostToDevice,
                                stream[0]);
        
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
                kernel_SymmetrizeF<<<dim, dim, 0, stream[0]>>>(devDataF,
                                                               devSymmat, 
                                                               nSymmetryElement, 
                                                               (double)vecSize, 
                                                               interp, 
                                                               shift,
                                                               dim,
                                                               dimSize - shift,
                                                               texObject);
#endif


                kernel_NormalizeFW<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                               devPartW, 
                                                               dimSize - shift, 
                                                               shift, 
                                                               shift);
    
#ifdef RECONSTRUCTOR_LOW_PASS    
                kernel_LowpassF<<<dim, dim, 0, stream[0]>>>(devDataF, 
                                                            thres,
                                                            ew
                                                            shift,
                                                            dim,
                                                            dimSize - shift);
#endif    

                cudaMemcpyAsync(F3D + shift,
                                devDataF + shift,
                                (dimSize - shift) * sizeof(Complex),
                                cudaMemcpyDeviceToHost,
                                stream[0]);
                //cudaCheckErrors("for memcpy 0.");
        
            } 
        } 
        
    }
    cudaStreamSynchronize(stream[0]);
    //cudaCheckErrors("CUDA stream0 synchronization.");
    cudaStreamSynchronize(stream[1]);
    //cudaCheckErrors("CUDA stream1 synchronization.");
    cudaStreamSynchronize(stream[2]);
    //cudaCheckErrors("CUDA stream2 synchronization.");
   
    cudaDestroyTextureObject(texObject);
    cudaHostUnregister(F3D);
    cudaHostUnregister(W3D);
    
    LOG(INFO) << "Step 5: Clean up the streams and memory.";

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaStreamDestroy(stream[2]);
    //cudaCheckErrors("Destroy stream.");

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
    cudaFree(devSymmat);
    //cudaCheckErrors("Free device memory devSymmat.");
    
    cudaFreeArray(symArray);
    //cudaCheckErrors("Free device memory SymArray.");
#endif    
    
    cudaFree(devPartW);
    //cudaCheckErrors("Free device memory of W");

    cudaFree(devDataF);
    //cudaCheckErrors("Free device memory devDataF.");

}

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CorrSoftMaskF(double *dst,
                   double *mkbRL,
                   const int dim)
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
            cudaSetDevice(n);
            break; 
        }
    }

    int dimSize = dim * dim * dim;
    
    cudaHostRegister(dst, dimSize * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register dst data.");
    
    double *devDst;
    cudaMalloc((void**)&devDst, dimSize * sizeof(double));
    //cudaCheckErrors("Allocate devDst data.");
    
    LOG(INFO) << "Step2: Correcting Convolution Kernel."; 
    
#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    double *devMkb;
    cudaMalloc((void**)&devMkb, mkbSize * sizeof(double));
    //cudaCheckErrors("Allocate devMkb data.");
    cudaMemcpy(devMkb, mkbRL, mkbSize * sizeof(double), cudaMemcpyHostToDevice);
    //cudaCheckErrors("Copy devMkb to device.");

#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    double *devTik;
    cudaMalloc((void**)&devTik, mkbSize * sizeof(double));
    //cudaCheckErrors("Allocate devTik data.");
    cudaMemcpy(devTik, mkbRL, mkbSize * sizeof(double), cudaMemcpyHostToDevice);
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
                        batchSize * sizeof(double),
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

        cudaMemcpyAsync(dst + shift,
                        devDst + shift,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("out for memcpy 0.");
    
        cudaMemcpyAsync(devDst + shift + batchSize,
                        dst + shift + batchSize,
                        batchSize * sizeof(double),
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
        
        cudaMemcpyAsync(dst + shift + batchSize,
                        devDst + shift + batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("out for memcpy 1.");

        cudaMemcpyAsync(devDst + shift + 2 * batchSize,
                        dst + shift + 2 * batchSize,
                        batchSize * sizeof(double),
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
        cudaMemcpyAsync(dst + shift + 2 * batchSize,
                        devDst + shift + 2 * batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[2]);
        //cudaCheckErrors("out for memcpy 2.");
    }

    if (len % 3 == 2)
    {
        int shift = (len - 2) * batchSize;
        
        cudaMemcpyAsync(devDst + shift,
                        dst + shift,
                        batchSize * sizeof(double),
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
    
        cudaMemcpyAsync(dst + shift,
                        devDst + shift,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[0]);
        //cudaCheckErrors("out for memcpy 0.");
    
        cudaMemcpyAsync(devDst + shift + batchSize,
                        dst + shift + batchSize,
                        batchSize * sizeof(double),
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
        
        cudaMemcpyAsync(dst + shift + batchSize,
                        devDst + shift + batchSize,
                        batchSize * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        stream[1]);
        //cudaCheckErrors("out for memcpy 1.");

        if (dimSize % batchSize != 0)
        {
            int shift = len * batchSize;
            cudaMemcpyAsync(devDst + shift,
                            dst + shift,
                            (dimSize - shift) * sizeof(double),
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
    
            cudaMemcpyAsync(dst + shift,
                            devDst + shift,
                            (dimSize - shift) * sizeof(double),
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
            cudaMemcpyAsync(devDst + shift,
                            dst + shift,
                            batchSize * sizeof(double),
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
            
            cudaMemcpyAsync(dst + shift,
                            devDst + shift,
                            batchSize * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            stream[0]);
            //cudaCheckErrors("out for memcpy 0.");
            
            if (dimSize % batchSize != 0)
            {
                int shift = len * batchSize;
                cudaMemcpyAsync(devDst + shift,
                                dst + shift,
                                (dimSize - shift) * sizeof(double),
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
    
                cudaMemcpyAsync(dst + shift,
                                devDst + shift,
                                (dimSize - shift) * sizeof(double),
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
                cudaMemcpyAsync(devDst + shift,
                                dst + shift,
                                (dimSize - shift) * sizeof(double),
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
                cudaMemcpyAsync(dst + shift,
                                devDst + shift,
                                (dimSize - shift) * sizeof(double),
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

    cudaFree(devDst);
    //cudaCheckErrors("Free device memory devDst.");

}

/**
 * @brief
 *
 * @param 
 * @param
 * @param
 */
void CorrSoftMaskF(double *dst,
                   double *mkbRL,
                   double nf,
                   const int dim,
                   const int size,
                   const int edgeWidth)
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
            cudaSetDevice(n);
            break; 
        }
    }

    int dimSize = dim * dim * dim;
    
    cudaHostRegister(dst, dimSize * sizeof(double), cudaHostRegisterDefault);
    //cudaCheckErrors("Register dst data.");
    
    double *devDst;
    cudaMalloc((void**)&devDst, dimSize * sizeof(double));
    //cudaCheckErrors("Allocate devDst data.");
    
    LOG(INFO) << "Step2: Correcting Convolution Kernel."; 
    
#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef RECONSTRUCTOR_MKB_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    double *devMkb;
    cudaMalloc((void**)&devMkb, mkbSize * sizeof(double));
    //cudaCheckErrors("Allocate devMkb data.");
    cudaMemcpy(devMkb, mkbRL, mkbSize * sizeof(double), cudaMemcpyHostToDevice);
    //cudaCheckErrors("Copy devMkb to device.");

#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
    
    int mkbSize = (dim / 2 + 1) * (dim / 2 + 1) * (dim / 2 + 1);
    double *devTik;
    cudaMalloc((void**)&devTik, mkbSize * sizeof(double));
    //cudaCheckErrors("Allocate devTik data.");
    cudaMemcpy(devTik, mkbRL, mkbSize * sizeof(double), cudaMemcpyHostToDevice);
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
                        batchSize * sizeof(double),
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
                        batchSize * sizeof(double),
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
                        batchSize * sizeof(double),
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
                        batchSize * sizeof(double),
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
                        batchSize * sizeof(double),
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
                            (dimSize - shift) * sizeof(double),
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
                            batchSize * sizeof(double),
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
                                (dimSize - shift) * sizeof(double),
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
                                (dimSize - shift) * sizeof(double),
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
    
    double *bg;
    cudaMalloc((void**)&bg, sizeof(double));
    //cudaCheckErrors("Allocate device memory for devSumG.");

#ifdef RECONSTRUCTOR_REMOVE_CORNER_MASK_ZERO
    cudaMemset(bg, 0.0, sizeof(double));
#else
    
    double *devSumG;
    double *devSumWG;
    cudaMalloc((void**)&devSumG, dim * sizeof(double));
    //cudaCheckErrors("Allocate device memory for devSumG.");
    cudaMalloc((void**)&devSumWG, dim * sizeof(double));
    //cudaCheckErrors("Allocate device memory for devSumWG.");
    
    kernel_Background<<<dim, dim, 2 * dim * sizeof(double)>>>(devDst,
                                                              devSumG,
                                                              devSumWG,
                                                              size / 2,
                                                              edgeWidth,
                                                              dim,
                                                              dimSize);
    
    kernel_CalculateBg<<<1, dim>>>(construct,
                                   devSumG,
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
                        batchSize * sizeof(double),
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
                        batchSize * sizeof(double),
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
                        batchSize * sizeof(double),
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
                        batchSize * sizeof(double),
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
                        batchSize * sizeof(double),
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
                            (dimSize - shift) * sizeof(double),
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
                            batchSize * sizeof(double),
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
                                (dimSize - shift) * sizeof(double),
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
                                (dimSize - shift) * sizeof(double),
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
