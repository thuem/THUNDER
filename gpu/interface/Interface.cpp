#include "Interface.h"

#include "cuthunder.h"

void getAviDevice(std::vector<int>& gpus)
{
    cuthunder::getAviDevice(gpus);
}

void ExpectPreidx(int gpuIdx,
                  int** deviCol,
                  int** deviRow,
                  int* iCol,
                  int* iRow,
                  int npxl)
{
    cuthunder::expectPreidx(gpuIdx,
                            deviCol,
                            deviRow,
                            iCol,
                            iRow,
                            npxl);
}

void ExpectPrefre(int gpuIdx,
                  RFLOAT** devfreQ,
                  RFLOAT* freQ,
                  int npxl)
{
    cuthunder::expectPrefre(gpuIdx,
                            devfreQ,
                            freQ,
                            npxl);

}

void ExpectLocalIn(int gpuIdx,
                   Complex** devdatP,
                   RFLOAT** devctfP,
                   RFLOAT** devdefO,
                   RFLOAT** devsigP,
                   int nPxl,
                   int cpyNumL,
                   int searchType)
{
    cuthunder::expectLocalIn(gpuIdx,
                             reinterpret_cast<cuthunder::Complex**>(devdatP),
                             devctfP,
                             devdefO,
                             devsigP,
                             nPxl,
                             cpyNumL,
                             searchType);

}

void ExpectLocalV2D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int dimSize)
{
    cuthunder::expectLocalV2D(gpuIdx,
                              mgr,
                              reinterpret_cast<cuthunder::Complex*>(volume),
                              dimSize);

}

void ExpectLocalV3D(int gpuIdx,
                    ManagedArrayTexture* mgr,
                    Complex* volume,
                    int vdim)
{
    cuthunder::expectLocalV3D(gpuIdx,
                              mgr,
                              reinterpret_cast<cuthunder::Complex*>(volume),
                              vdim);

}

void ExpectLocalP(int gpuIdx,
                  Complex* devdatP,
                  RFLOAT* devctfP,
                  RFLOAT* devdefO,
                  RFLOAT* devsigP,
                  Complex* datP,
                  RFLOAT* ctfP,
                  RFLOAT* defO,
                  RFLOAT* sigP,
                  int threadId,
                  int imgId,
                  int npxl,
                  int cSearch)
{
    cuthunder::expectLocalP(gpuIdx,
                            reinterpret_cast<cuthunder::Complex*>(devdatP),
                            devctfP,
                            devdefO,
                            devsigP,
                            reinterpret_cast<cuthunder::Complex*>(datP),
                            ctfP,
                            defO,
                            sigP,
                            threadId,
                            imgId,
                            npxl,
                            cSearch);

}

void ExpectLocalHostA(int gpuIdx,
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
    cuthunder::expectLocalHostA(gpuIdx,
                                wC,
                                wR,
                                wT,
                                wD,
                                oldR,
                                oldT,
                                oldD,
                                trans,
                                rot,
                                dpara,
                                mR,
                                mT,
                                mD,
                                cSearch);
}

void ExpectLocalRTD(int gpuIdx,
                    ManagedCalPoint* mcp,
                    double* oldR,
                    double* oldT,
                    double* oldD,
                    double* trans,
                    double* rot,
                    double* dpara)
{
    cuthunder::expectLocalRTD(gpuIdx,
                              mcp,
                              oldR,
                              oldT,
                              oldD,
                              trans,
                              rot,
                              dpara);
}

void ExpectLocalPreI2D(int gpuIdx,
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
    cuthunder::expectLocalPreI2D(gpuIdx,
                                 datShift,
                                 mgr,
                                 mcp,
                                 devdefO,
                                 devfreQ,
                                 deviCol,
                                 deviRow,
                                 phaseShift,
                                 conT,
                                 k1,
                                 k2,
                                 pf,
                                 idim,
                                 vdim,
                                 npxl,
                                 interp);
}

void ExpectLocalPreI3D(int gpuIdx,
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
    cuthunder::expectLocalPreI3D(gpuIdx,
                                 datShift,
                                 mgr,
                                 mcp,
                                 devdefO,
                                 devfreQ,
                                 deviCol,
                                 deviRow,
                                 phaseShift,
                                 conT,
                                 k1,
                                 k2,
                                 pf,
                                 idim,
                                 vdim,
                                 npxl,
                                 interp);
}

void ExpectLocalM(int gpuIdx,
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
    cuthunder::expectLocalM(gpuIdx,
                            datShift,
                            //l,
                            //dvpA,
                            //baseL,
                            mcp,
                            reinterpret_cast<cuthunder::Complex*>(devdatP),
                            devctfP,
                            devsigP,
                            wC,
                            wR,
                            wT,
                            wD,
                            oldC,
                            npxl);
}

void ExpectLocalHostF(int gpuIdx,
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
    cuthunder::expectLocalHostF(gpuIdx,
                                wC,
                                wR,
                                wT,
                                wD,
                                oldR,
                                oldT,
                                oldD,
                                trans,
                                rot,
                                dpara,
                                cSearch);
}

void ExpectLocalFin(int gpuIdx,
                    Complex** devdatP,
                    RFLOAT** devctfP,
                    RFLOAT** devdefO,
                    RFLOAT** devfreQ,
                    RFLOAT** devsigP,
                    int cSearch)
{
    cuthunder::expectLocalFin(gpuIdx,
                              reinterpret_cast<cuthunder::Complex**>(devdatP),
                              devctfP,
                              devdefO,
                              devfreQ,
                              devsigP,
                              cSearch);
}

void ExpectFreeIdx(int gpuIdx,
                   int** deviCol,
                   int** deviRow)
{
    cuthunder::expectFreeIdx(gpuIdx,
                             deviCol,
                             deviRow);

}

void ExpectPrecal(vector<CTFAttr>& ctfAttr,
                  RFLOAT* def,
                  RFLOAT* k1,
                  RFLOAT* k2,
                  const int *iCol, 
                  const int *iRow,
                  int idim, 
                  int npxl,
                  int imgNum)
{
    LOG(INFO) << "Prepare Parameter for Expectation Pre-cal.";

    std::vector<cuthunder::CTFAttr*> ctfaData;
    for (int i = 0; i < imgNum; i++)
    {
        ctfaData.push_back(reinterpret_cast<cuthunder::CTFAttr*>(&ctfAttr[i])); 
    }

    cuthunder::expectPrecal(ctfaData,
                            def,
                            k1,
                            k2,
                            iCol,
                            iRow,
                            idim,
                            npxl,
                            imgNum);
}

void ExpectGlobal2D(Complex* vol,
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
    LOG(INFO) << "Prepare Parameter for Expectation Global.";
    
    cuthunder::expectGlobal2D(reinterpret_cast<cuthunder::Complex*>(vol),
                              reinterpret_cast<cuthunder::Complex*>(datP),
                              ctfP,
                              sigRcpP,
                              trans,
                              wC,
                              wR,
                              wT,
                              pR,
                              pT,
                              rot,
                              iCol,
                              iRow,
                              nK,
                              nR,
                              nT,
                              pf,
                              interp,
                              idim,
                              vdim,
                              npxl,
                              imgNum);
}

void ExpectRotran(Complex* traP,
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
    LOG(INFO) << "Prepare Parameter for Expectation Rotation and Translate.";
    
    cuthunder::expectRotran(reinterpret_cast<cuthunder::Complex*>(traP),
                            trans,
                            rot,
                            rotMat,
                            iCol,
                            iRow,
                            nR,
                            nT,
                            idim,
                            npxl);
}

void ExpectProject(Complex* volume,
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
    LOG(INFO) << "Prepare Parameter for Expectation Projection.";
    
    cuthunder::expectProject(reinterpret_cast<cuthunder::Complex*>(volume),
                             reinterpret_cast<cuthunder::Complex*>(rotP),
                             rotMat,
                             iCol,
                             iRow,
                             nR,
                             pf,
                             interp,
                             vdim,
                             npxl);
}

void ExpectGlobal3D(Complex* rotP,
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
    LOG(INFO) << "Prepare Parameter for Expectation Global.";
    
    cuthunder::expectGlobal3D(reinterpret_cast<cuthunder::Complex*>(rotP),
                              reinterpret_cast<cuthunder::Complex*>(traP),
                              reinterpret_cast<cuthunder::Complex*>(datP),
                              ctfP,
                              sigRcpP,
                              wC,
                              wR,
                              wT,
                              pR,
                              pT,
                              baseL,
                              kIdx,
                              nK,
                              nR,
                              nT,
                              npxl,
                              imgNum);
}

void InsertI2D(Complex* F2D,
               RFLOAT* T2D,
               double* O2D,
               int* counter,
               MPI_Comm& hemi,
               MPI_Comm& slav,
               Complex* datP,
               RFLOAT* ctfP,
               RFLOAT* sigRcpP,
               RFLOAT *w,
               double *offS,
               int *nC,
               double *nR,
               double *nT,
               double *nD,
               CTFAttr* ctfaData,
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
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    cuthunder::InsertI2D(reinterpret_cast<cuthunder::Complex*>(F2D),
                         T2D,
                         O2D,
                         counter,
                         hemi,
                         slav,
                         reinterpret_cast<cuthunder::Complex*>(datP),
                         ctfP,
                         sigRcpP,
                         w,
                         offS,
                         nC,
                         nR,
                         nT,
                         nD,
                         reinterpret_cast<cuthunder::CTFAttr*>(ctfaData),
                         iCol,
                         iRow,
                         pixelSize,
                         cSearch,
                         nk,
                         opf,
                         npxl,
                         mReco,
                         idim,
                         vdim,
                         imgNum);
}

void InsertFT(Volume& F3D,
              Volume& T3D,
              double* O3D,
              int* counter,
              MPI_Comm& hemi,
              MPI_Comm& slav,
              Complex* datP,
              RFLOAT* ctfP,
              RFLOAT* sigRcpP,
              CTFAttr* ctfaData,
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
              int idim,
              int dimSize,
              int imgNum)
{
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    Complex *comF3D = &F3D[0];
    
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(T3D[i]);
	}

    cuthunder::InsertFT(reinterpret_cast<cuthunder::Complex*>(comF3D),
                        douT3D,
                        O3D,
                        counter,
                        hemi,
                        slav,
                        reinterpret_cast<cuthunder::Complex*>(datP),
                        ctfP,
                        sigRcpP,
                        reinterpret_cast<cuthunder::CTFAttr*>(ctfaData),
                        offS,
                        w,
                        nR,
                        nT,
                        nD,
                        nC,
                        iCol,
                        iRow,
                        pixelSize,
                        cSearch,
                        opf,
                        npxl,
                        mReco,
                        imgNum,
                        idim,
                        F3D.nSlcFT());

    for(int i = 0; i < dimSize; i++)
	{
        T3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
}

void InsertFT(Volume& F3D,
              Volume& T3D,
              double* O3D,
              int* counter,
              MPI_Comm& hemi,
              MPI_Comm& slav,
              Complex* datP,
              RFLOAT* ctfP,
              RFLOAT* sigRcpP,
              CTFAttr* ctfaData,
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
              int idim,
              int dimSize,
              int imgNum)
{
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    Complex *comF3D = &F3D[0];
    
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(T3D[i]);
	}

    cuthunder::InsertFT(reinterpret_cast<cuthunder::Complex*>(comF3D),
                        douT3D,
                        O3D,
                        counter,
                        hemi,
                        slav,
                        reinterpret_cast<cuthunder::Complex*>(datP),
                        ctfP,
                        sigRcpP,
                        reinterpret_cast<cuthunder::CTFAttr*>(ctfaData),
                        offS,
                        w,
                        nR,
                        nT,
                        nD,
                        iCol,
                        iRow,
                        pixelSize,
                        cSearch,
                        opf,
                        npxl,
                        mReco,
                        imgNum,
                        idim,
                        F3D.nSlcFT());

    for(int i = 0; i < dimSize; i++)
	{
        T3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
}

void PrepareTF(int gpuIdx,
               Volume& F3D,
	           Volume& T3D,
	           double* symMat,
               int nSymmetryElement,
               int maxRadius,
	           int pf)
{
	LOG(INFO) << "Step1: Prepare Parameter for NormalizeT.";

	RFLOAT sf = 1.0 / REAL(T3D[0]);
    int dim = T3D.nSlcFT(); 
    int dimSize = dim * dim * (dim / 2 + 1); 
    int r = (maxRadius * pf + 1) * (maxRadius * pf + 1);

    Complex *comF3D = &F3D[0];
	Complex *comT3D = &T3D[0];
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(comT3D[i]);
	}
    

    LOG(INFO) << "Step2: Start PrepareTF...";

    cuthunder::PrepareTF(gpuIdx,
                         reinterpret_cast<cuthunder::Complex*>(comF3D),
                         douT3D,
                         symMat,
                         sf,
                         nSymmetryElement, 
                         LINEAR_INTERP,
                         dim,
                         r);
    
    for(int i = 0; i < dimSize; i++)
	{
        comT3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
}

void ExposePT2D(int gpuIdx,
                Image& T2D,
	            int maxRadius,
	            int pf,
	            vec FSC,
	            bool joinHalf,
                const int wienerF)
{
	LOG(INFO) << "Step1: Prepare Parameter for T.";

    int dim = T2D.nRowFT(); 
    int dimSize = dim * (dim / 2 + 1); 
	Complex *comT2D = &T2D[0];
    RFLOAT *douT2D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT2D[i] = REAL(comT2D[i]);
	}
    
    LOG(INFO) << "Step2: Prepare Paramete for Calculate FSC.";

    int fscMatsize = FSC.size();
    RFLOAT *FSCmat = new RFLOAT[fscMatsize];
    Map<vec>(FSCmat, FSC.rows(), FSC.cols()) = FSC;
    
    LOG(INFO) << "Step3: Start CalculateT...";

    cuthunder::CalculateT2D(gpuIdx,
                            douT2D,
                            FSCmat,
                            fscMatsize,
                            joinHalf,
                            maxRadius,
                            wienerF,
                            dim,
                            pf);
    
    for(int i = 0; i < dimSize; i++)
	{
        comT2D[i] = COMPLEX(douT2D[i], 0);
	}
    
    delete[]douT2D;
    delete[]FSCmat;
}

void ExposePT(int gpuIdx,
              Volume& T3D,
	          int maxRadius,
	          int pf,
	          vec FSC,
	          bool joinHalf,
              const int wienerF)
{
	LOG(INFO) << "Step1: Prepare Parameter for T.";

    int dim = T3D.nSlcFT(); 
    int dimSize = dim * dim * (dim / 2 + 1); 
	Complex *comT3D = &T3D[0];
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(comT3D[i]);
	}
    
    LOG(INFO) << "Step2: Prepare Paramete for Calculate FSC.";

    int fscMatsize = FSC.size();
    RFLOAT *FSCmat = new RFLOAT[fscMatsize];
    Map<vec>(FSCmat, FSC.rows(), FSC.cols()) = FSC;
    
    LOG(INFO) << "Step3: Start CalculateT...";

    cuthunder::CalculateT(gpuIdx,
                          douT3D,
                          FSCmat,
                          fscMatsize,
                          joinHalf,
                          maxRadius,
                          wienerF,
                          dim,
                          pf);
    
    for(int i = 0; i < dimSize; i++)
	{
        comT3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
    delete[]FSCmat;
}

void ExposeWT2D(int gpuIdx,
                Image& T2D,
                Image& W2D,
                TabFunction& kernelRL,
                int maxRadius,
                int pf,
                RFLOAT a,
                RFLOAT nf,
                RFLOAT alpha,
                int maxIter,
                int minIter,
                int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for InitialW.";
    
    int dim = T2D.nRowFT(); 
    int dimSize = dim * (dim / 2 + 1); 
    Complex *comW2D = &W2D[0];
    RFLOAT *comW2DR = new RFLOAT[dimSize];
    for(int i = 0; i < dimSize; i++)
    {
        comW2DR[i] = REAL(comW2D[i]);
    }

    LOG(INFO) << "Step2: Prepare Paramete for Calculate C.";

    int r = (maxRadius * pf) * (maxRadius * pf);

    Complex *comT2D = &T2D[0];
    RFLOAT *comT2DR = new RFLOAT[dimSize];
    for(int i = 0; i < dimSize; i++)
    {
        comT2DR[i] = REAL(comT2D[i]);
    }
    
    RFLOAT *tabdata = kernelRL.getData();
    RFLOAT step = kernelRL.getStep();
    int padSize = pf * size; 
    
    LOG(INFO) << "Step3: Start Calculate C...";
      
    cuthunder::CalculateW2D(gpuIdx,
                            comT2DR,
                            comW2DR,
                            tabdata,
                            0,
                            1,
                            step,
                            1e5,
                            dim,
                            r,
                            nf,
                            maxIter,
                            minIter,
                            padSize);
    
    for(int i = 0; i < dimSize; i++)
    {
        comW2D[i] = COMPLEX(comW2DR[i], 0);
    }

    delete[]comW2DR;
    delete[]comT2DR;
}

void ExposeWT(int gpuIdx,
              Volume& T3D,
              Volume& W3D,
              TabFunction& kernelRL,
              int maxRadius,
              int pf,
              RFLOAT a,
              RFLOAT nf,
              RFLOAT alpha,
              int maxIter,
              int minIter,
              int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for InitialW.";
    
    int dim = T3D.nSlcFT(); 
    int dimSize = dim * dim * (dim / 2 + 1); 
    Complex *comW3D = &W3D[0];
    RFLOAT *comW3DR = new RFLOAT[dimSize];
    for(int i = 0; i < dimSize; i++)
    {
        comW3DR[i] = REAL(comW3D[i]);
    }

    LOG(INFO) << "Step2: Prepare Paramete for Calculate C.";

    int r = (maxRadius * pf) * (maxRadius * pf);

    Complex *comT3D = &T3D[0];
    RFLOAT *comT3DR = new RFLOAT[dimSize];
    for(int i = 0; i < dimSize; i++)
    {
        comT3DR[i] = REAL(comT3D[i]);
    }
    
    RFLOAT *tabdata = kernelRL.getData();
    RFLOAT step = kernelRL.getStep();
    int padSize = pf * size; 
    
    LOG(INFO) << "Step3: Start Calculate C...";
      
    cuthunder::CalculateW(gpuIdx,
                          comT3DR,
                          comW3DR,
                          tabdata,
                          0,
                          1,
                          step,
                          1e5,
                          dim,
                          r,
                          nf,
                          maxIter,
                          minIter,
                          padSize);
    
    for(int i = 0; i < dimSize; i++)
    {
        comW3D[i] = COMPLEX(comW3DR[i], 0);
    }

    delete[]comW3DR;
    delete[]comT3DR;
}

void ExposeWT2D(int gpuIdx,
                Image& T2D,
                Image& W2D,
                int maxRadius,
                int pf)
{
    LOG(INFO) << "Step1: Prepare Parameter for InitialW.";
    
    int r = (maxRadius * pf) * (maxRadius * pf);

    int dim = T2D.nRowFT(); 
    int dimSize = dim * (dim / 2 + 1); 
    Complex *comW2D = &W2D[0];
    RFLOAT *comW2DR = new RFLOAT[dimSize];
    for(int i = 0; i < dimSize; i++)
    {
        comW2DR[i] = REAL(comW2D[i]);
    }

    Complex *comT2D = &T2D[0];
    RFLOAT *comT2DR = new RFLOAT[dimSize];
    for(int i = 0; i < dimSize; i++)
    {
        comT2DR[i] = REAL(comT2D[i]);
    }
    
    LOG(INFO) << "Step2: Start Calculate W...";
      
    cuthunder::CalculateW2D(gpuIdx,
                            comT2DR,
                            comW2DR,
                            dim,
                            r);
    
    for(int i = 0; i < dimSize; i++)
    {
        comW2D[i] = COMPLEX(comW2DR[i], 0);
    }

    delete[]comW2DR;
    delete[]comT2DR;
}

void ExposeWT(int gpuIdx,
              Volume& T3D,
              Volume& W3D,
              int maxRadius,
              int pf)
{
    LOG(INFO) << "Step1: Prepare Parameter for InitialW.";
    
    int r = (maxRadius * pf) * (maxRadius * pf);

    int dim = T3D.nSlcFT(); 
    int dimSize = dim * dim * (dim / 2 + 1); 
    Complex *comW3D = &W3D[0];
    RFLOAT *comW3DR = new RFLOAT[dimSize];
    for(int i = 0; i < dimSize; i++)
    {
        comW3DR[i] = REAL(comW3D[i]);
    }

    Complex *comT3D = &T3D[0];
    RFLOAT *comT3DR = new RFLOAT[dimSize];
    for(int i = 0; i < dimSize; i++)
    {
        comT3DR[i] = REAL(comT3D[i]);
    }
    
    LOG(INFO) << "Step2: Start Calculate W...";
      
    cuthunder::CalculateW(gpuIdx,
                          comT3DR,
                          comW3DR,
                          dim,
                          r);
    
    for(int i = 0; i < dimSize; i++)
    {
        comW3D[i] = COMPLEX(comW3DR[i], 0);
    }

    delete[]comW3DR;
    delete[]comT3DR;
}

void ExposePF2D(int gpuIdx,
                Image& padDst,
                Image& padDstR,
                Image& F2D,
                Image& W2D,
                int maxRadius,
                int pf)
{
    LOG(INFO) << "Step1: Prepare Parameter for pad.";

    Complex *comPAD = &padDst[0];
    Complex *comF2D = &F2D[0];
    RFLOAT *comPADR = &padDstR(0);
    
    LOG(INFO) << "Step2: Prepare Paramete for CalculateFW.";

    int dim = F2D.nRowFT(); 
    int pdim = padDst.nRowFT(); 
    int dimSize = dim * (dim / 2 + 1); 
    Complex *comW2D = &W2D[0];
	RFLOAT *comW2DR = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        comW2DR[i] = REAL(comW2D[i]);
	}
	
    int r = (maxRadius * pf) * (maxRadius * pf);

    LOG(INFO) << "Step4: Start PrepareF...";
    
    cuthunder::CalculateF2D(gpuIdx,
                            reinterpret_cast<cuthunder::Complex*>(comPAD),
                            reinterpret_cast<cuthunder::Complex*>(comF2D),
                            comPADR,
                            comW2DR,
                            r,
                            pdim,
                            dim);
    
    delete[]comW2DR;
}

void ExposePF(int gpuIdx,
              Volume& padDst,
              Volume& padDstR,
              Volume& F3D,
              Volume& W3D,
              int maxRadius,
              int pf)
{
    LOG(INFO) << "Step1: Prepare Parameter for pad.";

    Complex *comPAD = &padDst[0];
    Complex *comF3D = &F3D[0];
    RFLOAT *comPADR = &padDstR(0);
    
    LOG(INFO) << "Step2: Prepare Paramete for CalculateFW.";

    int dim = F3D.nSlcFT(); 
    int pdim = padDst.nSlcFT(); 
    int dimSize = dim * dim * (dim / 2 + 1); 
    Complex *comW3D = &W3D[0];
	RFLOAT *comW3DR = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        comW3DR[i] = REAL(comW3D[i]);
	}
	
    int r = (maxRadius * pf) * (maxRadius * pf);

    LOG(INFO) << "Step4: Start PrepareF...";
    
    cuthunder::CalculateF(gpuIdx,
                          reinterpret_cast<cuthunder::Complex*>(comPAD),
                          reinterpret_cast<cuthunder::Complex*>(comF3D),
                          comPADR,
                          comW3DR,
                          r,
                          pdim,
                          dim);
    
    delete[]comW3DR;
}

void ExposeCorrF2D(int gpuIdx,
                   Image& imgDst,
                   Volume& dst,
                   RFLOAT* mkbRL,
                   RFLOAT nf)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";
    
    RFLOAT *comIDst = &imgDst(0);
    Complex *comDst = &dst[0];
    int dim = imgDst.nRowRL();
    
    LOG(INFO) << "Step2: Start CorrSoftMaskF...";
      
    cuthunder::CorrSoftMaskF2D(gpuIdx,
                               comIDst,
                               reinterpret_cast<cuthunder::Complex*>(comDst),
                               mkbRL,
                               nf,
                               dim);

}

void ExposeCorrF(int gpuIdx,
                 Volume& dstN,
                 Volume& dst,
                 RFLOAT* mkbRL,
                 RFLOAT nf)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";

    RFLOAT *comDstN = &dstN(0);
    Complex *comDst = &dst[0];
    int dim = dstN.nSlcRL();
    
    LOG(INFO) << "Step2: Start CorrSoftMaskF...";
      
    cuthunder::CorrSoftMaskF(gpuIdx,
                             reinterpret_cast<cuthunder::Complex*>(comDst),
                             comDstN,
                             mkbRL,
                             nf,
                             dim);
   
}

void TranslateI2D(int gpuIdx,
                  Image& img,
                  double ox,
                  double oy,
                  int r)
{
    LOG(INFO) << "Step1: Prepare Parameter for TransImg.";

    Complex *comImg = &img[0];
    int dim = img.nRowRL();

    LOG(INFO) << "Step4: Start PrepareF...";
    
    cuthunder::TranslateI2D(gpuIdx,
                            reinterpret_cast<cuthunder::Complex*>(comImg),
                            (RFLOAT)ox,
                            (RFLOAT)oy,
                            r,
                            dim);
}

void TranslateI(int gpuIdx,
                Volume& ref,
                double ox,
                double oy,
                double oz,
                int r)
{
    LOG(INFO) << "Step1: Prepare Parameter for TransImg.";

    Complex *comRef = &ref[0];
    int dim = ref.nSlcRL();

    LOG(INFO) << "Step4: Start PrepareF...";
    
    cuthunder::TranslateI(gpuIdx,
                          reinterpret_cast<cuthunder::Complex*>(comRef),
                          (RFLOAT)ox,
                          (RFLOAT)oy,
                          (RFLOAT)oz,
                          r,
                          dim);
}

void ReMask(vector<Image>& img,
            RFLOAT maskRadius,
            RFLOAT pixelSize,
            RFLOAT ew,
            int idim,
            int imgNum)
{
    LOG(INFO) << "Step1: Prepare Parameter for Remask.";

    std::vector<cuthunder::Complex*> imgData;
    for (int i = 0; i < imgNum; i++)
    {
        imgData.push_back(reinterpret_cast<cuthunder::Complex*>(&img[i][0])); 
    }

    cuthunder::reMask(imgData,
                      maskRadius,
                      pixelSize,
                      ew,
                      idim,
                      imgNum);
}

