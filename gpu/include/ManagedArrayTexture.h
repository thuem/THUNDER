/**************************************************************
 * FileName: ManagedArraytexture.h
 * Author  : Kunpeng WANG, Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#ifndef MANAGEDARRAYTEXTURE_H
#define MANAGEDARRAYTEXTURE_H

class ManagedArrayTexture {
public:
	~ManagedArrayTexture();

	void Init(int mode, int vdim, int gpuIdx);

	void* GetArray();

	void* GetTextureObject();

    int getDeviceId();

private:
    class DeviceStruct;

	DeviceStruct *_cuda;

};

#endif
