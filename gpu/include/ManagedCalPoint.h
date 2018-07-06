/**************************************************************
 * FileName: ManagedCalPoint.h
 * Author  : Zhao Wang
 * Version :
 * Description:
 *
 * History :
 *
 **************************************************************/
#ifndef MANAGEDCALPOINT_H
#define MANAGEDCALPOINT_H

#include "Precision.h"

class ManagedCalPoint {
public:
	~ManagedCalPoint();

	void Init(int mode, 
              int cSearch, 
              int gpuIdx, 
              int nR, 
              int nT, 
              int mD, 
              int npxl);

    Complex* getPriRotP();
    Complex* getDevtraP();
    RFLOAT* getDevctfD();
    RFLOAT* getDevBaseL();
    RFLOAT* getDevDvp();
    RFLOAT* getDevwC();
    RFLOAT* getDevwR();
    RFLOAT* getDevwT();
    RFLOAT* getDevwD();
    double* getDevR();
    double* getDevT();
    double* getDevD();
    RFLOAT* getDevtT();
    RFLOAT* getDevtD();
    double* getDevdP();
    double* getDevnR();
    double* getDevnT();
    double* getDevRotm();
    void* getStream();
    int getMode();
    int getCSearch();
    int getDeviceId();
    int getNR();
    int getNT();
    int getMD();

private:
    Complex* priRotP;
    Complex* devtraP;
    RFLOAT* devctfD;
    RFLOAT* devBaseL;
    RFLOAT* devDvp;
    RFLOAT* devwC;
    RFLOAT* devwR;
    RFLOAT* devwT;
    RFLOAT* devwD;
    double* devR;
    double* devT;
    double* devD;
    RFLOAT* devtT;
    RFLOAT* devtD;
    double* devdP;
    double* devnR;
    double* devnT;
    double* devRotm;
    void* stream;
    int _mode;
    int _cSearch;
    int _deviceId;
    int _nR;
    int _nT;
    int _mD;
};

#endif
