#include "Config.h"

#include "ManagedArrayTexture.h"
#include "Device.cuh"

class ManagedArrayTexture::DeviceStruct {
public:
    ~DeviceStruct();
	void Initialize2D(int vdim);
	void Initialize3D(int vdim);

public:
    int deviceId;
	cudaChannelFormatDesc channelDesc;
	cudaArray* symArray;
	struct cudaResourceDesc resDesc;
	cudaTextureDesc td;
	cudaTextureObject_t texObject;
};

ManagedArrayTexture::DeviceStruct::~DeviceStruct()
{
    cudaSetDevice(deviceId);
    cudaDestroyTextureObject(texObject);
    cudaFreeArray(symArray);
    cudaCheckErrors("symArray free error.");
}

void ManagedArrayTexture::DeviceStruct::Initialize2D(int vdim)
{
    cudaSetDevice(deviceId);
    cudaCheckErrors("Set deviceID error.");

#ifdef SINGLE_PRECISION
    channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
#else
    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
#endif    

    cudaMallocArray(&symArray, &channelDesc, vdim / 2 + 1, vdim);
    cudaCheckErrors("Allocate symArray data.");

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = symArray;
    
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&texObject, &resDesc, &td, NULL);
    cudaCheckErrors("Create symArray texObject.");
}

void ManagedArrayTexture::DeviceStruct::Initialize3D(int vdim)
{
    cudaSetDevice(deviceId);
    cudaCheckErrors("cuda Set Device.");

#ifdef SINGLE_PRECISION
    channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
#else
    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
#endif    

    cudaExtent extent;
    extent = make_cudaExtent(vdim / 2 + 1, vdim, vdim);
    cudaMalloc3DArray(&symArray, &channelDesc, extent);
    cudaCheckErrors("Allocate symArray data.");

    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = symArray;
    
    memset(&td, 0, sizeof(td));
    td.normalizedCoords = 0;
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&texObject, &resDesc, &td, NULL);
    cudaCheckErrors("Create symArray texObject.");
}

void ManagedArrayTexture::Init(int mode, int vdim, int gpuIdx)
{
	_cuda = new DeviceStruct();
    _cuda->deviceId = gpuIdx;
    if (mode == 1)
	    _cuda->Initialize3D(vdim);
    else
        _cuda->Initialize2D(vdim);
}

ManagedArrayTexture::~ManagedArrayTexture()
{
	// do clean up
    delete _cuda;
}

void* ManagedArrayTexture::GetArray()
{
	return static_cast<void*>(_cuda->symArray);
}

void* ManagedArrayTexture::GetTextureObject()
{
	return static_cast<void*>(&_cuda->texObject);
}

int ManagedArrayTexture::getDeviceId()
{
    return _cuda->deviceId;    
}

