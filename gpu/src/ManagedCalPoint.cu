#include "Config.h"

#include "ManagedCalPoint.h"
#include "Device.cuh"

void ManagedCalPoint::Init(int mode, int cSearch, int gpuIdx, int nR, int nT, int mD, int npxl)
{
    _mode = mode;
    _cSearch = cSearch;
    _deviceId = gpuIdx;
    _nR = nR;
    _nT = nT;
    _mD = mD;
    
    cudaSetDevice(gpuIdx); 

    /* Create and setup cuda stream */
    stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));

    cudaStreamCreate((cudaStream_t*)stream);
    cudaCheckErrors("Create Stream.");
    
    cudaMalloc((void**)&priRotP, nR * npxl * sizeof(Complex));
    cudaCheckErrors("Allocate rotP data.");

    cudaMalloc((void**)&devtraP, nT * npxl * sizeof(Complex));
    cudaCheckErrors("Allocate traP data.");
    
    if (cSearch != 2)
    {
        cudaMalloc((void**)&devDvp, nR * nT * sizeof(RFLOAT));
        cudaCheckErrors("Allocate dvP data.");
    
        cudaMalloc((void**)&devD, sizeof(double));
        cudaCheckErrors("Allocate d data.");
    }
    else
    {
        cudaMalloc((void**)&devctfD, mD * npxl * sizeof(RFLOAT));
        cudaCheckErrors("Allocate ctfP data.");
        
        cudaMalloc((void**)&devdP, mD * sizeof(double));
        cudaCheckErrors("Allocate frequence data.");
        
        cudaMalloc((void**)&devDvp, nR * nT * mD * sizeof(RFLOAT));
        cudaCheckErrors("Allocate dvP data.");
        
        cudaMalloc((void**)&devtT, nR * nT * sizeof(RFLOAT));
        cudaCheckErrors("Allocate tT data.");
    
        cudaMalloc((void**)&devtD, nR * mD * sizeof(RFLOAT));
        cudaCheckErrors("Allocate tD data.");
    
        cudaMalloc((void**)&devD, mD * sizeof(double));
        cudaCheckErrors("Allocate d data.");
    } 
    
    cudaMalloc((void**)&devBaseL, sizeof(RFLOAT));
    cudaCheckErrors("Allocate w data.");
    
    cudaMalloc((void**)&devwC, sizeof(RFLOAT));
    cudaCheckErrors("Allocate wc data.");
    
    cudaMalloc((void**)&devwR, nR * sizeof(RFLOAT));
    cudaCheckErrors("Allocate wr data.");
    
    cudaMalloc((void**)&devwT, nT * sizeof(RFLOAT));
    cudaCheckErrors("Allocate wt data.");
    
    cudaMalloc((void**)&devwD, mD * sizeof(RFLOAT));
    cudaCheckErrors("Allocate wd data.");
    
    cudaMalloc((void**)&devR, nR * sizeof(double));
    cudaCheckErrors("Allocate r data.");
    
    cudaMalloc((void**)&devT, nT * sizeof(double));
    cudaCheckErrors("Allocate t data.");
    
    cudaMalloc((void**)&devnR, nR * 4 * sizeof(double));
    cudaCheckErrors("Allocate nR data.");
    
    if (mode == 1)
    {
        cudaMalloc((void**)&devnR, nR * 4 * sizeof(double));
        cudaCheckErrors("Allocate nR data.");
    
        cudaMalloc((void**)&devRotm, nR * 9 * sizeof(double));
        cudaCheckErrors("Allocate rotM data.");
    }
    else
    {
        cudaMalloc((void**)&devnR, nR * 2 * sizeof(double));
        cudaCheckErrors("Allocate nR data.");
    }

    cudaMalloc((void**)&devnT, nT * 2 * sizeof(double));
    cudaCheckErrors("Allocate nT data.");
}

ManagedCalPoint::~ManagedCalPoint()
{
	// do clean up
    cudaSetDevice(_deviceId);
    
    cudaFree(priRotP);
    cudaFree(devtraP);
    if (_cSearch == 2)
    {
        cudaFree(devctfD);
        cudaFree(devdP);
        cudaFree(devtT);
        cudaFree(devtD);
    }
    cudaFree(devBaseL);
    cudaFree(devDvp);
    cudaFree(devwC);
    cudaFree(devwR);
    cudaFree(devwT);
    cudaFree(devwD);
    cudaFree(devR);
    cudaFree(devT);
    cudaFree(devD);
    cudaFree(devnR);
    cudaFree(devnT);
    if (_mode == 1)
    {
        cudaFree(devRotm);
    }

    cudaStreamDestroy(*(cudaStream_t*)stream);
    cudaCheckErrors("Calpoint free error.");
}

Complex* ManagedCalPoint::getPriRotP()
{
	return priRotP;
}

Complex* ManagedCalPoint::getDevtraP()
{
	return devtraP;
}

RFLOAT* ManagedCalPoint::getDevctfD()
{
	return devctfD;
}

RFLOAT* ManagedCalPoint::getDevBaseL()
{
	return devBaseL;
}

RFLOAT* ManagedCalPoint::getDevDvp()
{
	return devDvp;
}

RFLOAT* ManagedCalPoint::getDevwC()
{
	return devwC;
}

RFLOAT* ManagedCalPoint::getDevwR()
{
	return devwR;
}

RFLOAT* ManagedCalPoint::getDevwT()
{
	return devwT;
}

RFLOAT* ManagedCalPoint::getDevwD()
{
	return devwD;
}

double* ManagedCalPoint::getDevR()
{
	return devR;
}

double* ManagedCalPoint::getDevT()
{
	return devT;
}

double* ManagedCalPoint::getDevD()
{
	return devD;
}

RFLOAT* ManagedCalPoint::getDevtT()
{
	return devtT;
}

RFLOAT* ManagedCalPoint::getDevtD()
{
	return devtD;
}

double* ManagedCalPoint::getDevdP()
{
	return devdP;
}

double* ManagedCalPoint::getDevnR()
{
	return devnR;
}

double* ManagedCalPoint::getDevnT()
{
	return devnT;
}

double* ManagedCalPoint::getDevRotm()
{
	return devRotm;
}

void* ManagedCalPoint::getStream()
{
	return stream;
}

int ManagedCalPoint::getMode()
{
    return _mode;
}

int ManagedCalPoint::getCSearch()
{
    return _cSearch;
}

int ManagedCalPoint::getDeviceId()
{
    return _deviceId;    
}

int ManagedCalPoint::getNR()
{
    return _nR;
}

int ManagedCalPoint::getNT()
{
    return _nT;    
}

int ManagedCalPoint::getMD()
{
    return _mD;    
}
