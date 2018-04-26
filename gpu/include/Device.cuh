/***********************************************************************
 * FileName: Device.cuh
 * Author  : Kunpeng WANG
 * Version :
 * Description:
 *
 * History :
 *
 **********************************************************************/
#ifndef DEVICE_CUH
#define DEVICE_CUH

#include <cuda.h>
#include <cstdio>

#define HD_CALLABLE __host__ __device__

#define H_CALLABLE __host__

#define D_CALLABLE __device__

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define NCCLCHECK(cmd) \
    do { \
        ncclResult_t r = cmd; \
        if (r!= ncclSuccess) { \
            printf("Failed, NCCL error %s:%d '%s'\n", \
                __FILE__,__LINE__,ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
        } \
    } while(0) 

#define CU_MAX(a, b) (int)fmax((double)a, (double)b)

#define CU_MIN(a, b) (int)fmin((double)a, (double)b)

#if 9 < 8 || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600)
static __inline__ __device__ double atomicAdd(double* address, const double val)
{
    unsigned long long int* address_as_ull =
                            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(val + 
                                             __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang
        // in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#endif
