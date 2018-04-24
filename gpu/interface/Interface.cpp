#include "Interface.h"

#include "cuthunder.h"

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

void ExpectGlobal3D(Complex* vol,
                    Complex* datP,
                    RFLOAT* ctfP,
                    RFLOAT* sigRcpP,
                    double* trans,
                    RFLOAT* wC,
                    RFLOAT* wR,
                    RFLOAT* wT,
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
    
    cuthunder::expectGlobal3D(reinterpret_cast<cuthunder::Complex*>(vol),
                              reinterpret_cast<cuthunder::Complex*>(datP),
                              ctfP,
                              sigRcpP,
                              trans,
                              wC,
                              wR,
                              wT,
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
                   int idim,
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
                             idim,
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
                    RFLOAT* baseL,
                    int kIdx,
                    int nK,
                    int nR,
                    int nT,
                    int idim,
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
                              baseL,
                              kIdx,
                              nK,
                              nR,
                              nT,
                              idim,
                              npxl,
                              imgNum);
}

void InsertI2D(Complex* F2D,
               RFLOAT* T2D,
               MPI_Comm& hemi,
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
                         hemi,
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
              MPI_Comm& hemi,
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
              int imgNum)
{
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    Complex *comF3D = &F3D[0];
    int dimSize = T3D.sizeFT();
    
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(T3D[i]);
	}

    cuthunder::InsertFT(reinterpret_cast<cuthunder::Complex*>(comF3D),
                        douT3D,
                        hemi,
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

    for(size_t i = 0; i < T3D.sizeFT(); i++)
	{
        T3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
}

void InsertFT(Volume& F3D,
              Volume& T3D,
              MPI_Comm& hemi,
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
              int imgNum)
{
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    Complex *comF3D = &F3D[0];
    int dimSize = T3D.sizeFT();
    
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(T3D[i]);
	}

    cuthunder::InsertFT(reinterpret_cast<cuthunder::Complex*>(comF3D),
                        douT3D,
                        hemi,
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

    for(size_t i = 0; i < T3D.sizeFT(); i++)
	{
        T3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
}

void InsertF(Volume& F3D,
             Volume& T3D,
             MPI_Comm& hemi,
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
             int rSize,
             int tSize,
             int dSize,
             int npxl,
             int mReco,
             int idim,
             int imgNum)
{
    LOG(INFO) << "Prepare Parameter for Tranlate and Insert.";

    Complex *comF3D = &F3D[0];
    int dimSize = T3D.sizeFT();
    
    RFLOAT *douT3D = new RFLOAT[dimSize];
	for(int i = 0; i < dimSize; i++)
	{
        douT3D[i] = REAL(T3D[i]);
	}

    cuthunder::InsertF(reinterpret_cast<cuthunder::Complex*>(comF3D),
                       douT3D,
                       hemi,
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
                       rSize,
                       tSize,
                       dSize,
                       mReco,
                       imgNum,
                       idim,
                       F3D.nSlcFT());

    for(size_t i = 0; i < T3D.sizeFT(); i++)
	{
        T3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
}

void PrepareTF(int gpuIdx,
               Volume& F3D,
	           Volume& T3D,
               const Symmetry& sym,
	           int maxRadius,
	           int pf)
{
	LOG(INFO) << "Step1: Prepare Parameter for NormalizeT.";

	RFLOAT sf = 1.0 / REAL(T3D[0]);

	Complex *comT3D = &T3D[0];
    RFLOAT *douT3D = new RFLOAT[T3D.sizeFT()];
	for(size_t i = 0; i < T3D.sizeFT(); i++)
	{
        douT3D[i] = REAL(comT3D[i]);
	}
    
    Complex *comF3D = &F3D[0];
	
    LOG(INFO) << "Step2: Prepare Paramete for SymmetrizeT.";

	int nSymmetryElement = sym.nSymmetryElement();
    double *symMat = new double[nSymmetryElement * 9];

    dmat33 L, R;   
    
	for(int i = 0; i < nSymmetryElement; i++)
	{
        sym.get(L, R, i);
        Map<dmat33>(symMat + i * 9, 3, 3) = R;
	}
   
    int r = (maxRadius * pf + 1) * (maxRadius * pf + 1);

    LOG(INFO) << "Step3: Start PrepareTF...";

    cuthunder::PrepareTF(gpuIdx,
                         reinterpret_cast<cuthunder::Complex*>(comF3D),
                         douT3D,
                         symMat,
                         sf,
                         nSymmetryElement, 
                         LINEAR_INTERP,
                         T3D.nSlcFT(),
                         r);
    
    for(size_t i = 0; i < T3D.sizeFT(); i++)
	{
        comT3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
    delete[]symMat;
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

	Complex *comT2D = &T2D[0];
    RFLOAT *douT2D = new RFLOAT[T2D.sizeFT()];
	for(size_t i = 0; i < T2D.sizeFT(); i++)
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
                            T2D.nRowFT(),
                            pf);
    
    for(size_t i = 0; i < T2D.sizeFT(); i++)
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

	Complex *comT3D = &T3D[0];
    RFLOAT *douT3D = new RFLOAT[T3D.sizeFT()];
	for(size_t i = 0; i < T3D.sizeFT(); i++)
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
                          T3D.nSlcFT(),
                          pf);
    
    for(size_t i = 0; i < T3D.sizeFT(); i++)
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
                RFLOAT alpha,
                int maxIter,
                int minIter,
                int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for InitialW.";
    
    Complex *comW2D = &W2D[0];
    RFLOAT *comW2DR = new RFLOAT[W2D.sizeFT()];
    for(size_t i = 0; i < W2D.sizeFT(); i++)
    {
        comW2DR[i] = REAL(comW2D[i]);
    }

    LOG(INFO) << "Step2: Prepare Paramete for Calculate C.";

#ifdef RECONSTRUCTOR_KERNEL_PADDING
    RFLOAT nf = MKB_RL(0, a * pf, alpha);
#else
    RFLOAT nf = MKB_RL(0, a, alpha);
#endif
    
    int r = (maxRadius * pf) * (maxRadius * pf);

    Complex *comT2D = &T2D[0];
    RFLOAT *comT2DR = new RFLOAT[T2D.sizeFT()];
    for(size_t i = 0; i < T2D.sizeFT(); i++)
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
                            T2D.nRowFT(),
                            r,
                            nf,
                            maxIter,
                            minIter,
                            padSize);
    
    for(size_t i = 0; i < W2D.sizeFT(); i++)
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
              RFLOAT alpha,
              int maxIter,
              int minIter,
              int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for InitialW.";
    
    Complex *comW3D = &W3D[0];
    RFLOAT *comW3DR = new RFLOAT[W3D.sizeFT()];
    for(size_t i = 0; i < W3D.sizeFT(); i++)
    {
        comW3DR[i] = REAL(comW3D[i]);
    }

    LOG(INFO) << "Step2: Prepare Paramete for Calculate C.";

#ifdef RECONSTRUCTOR_KERNEL_PADDING
    RFLOAT nf = MKB_RL(0, a * pf, alpha);
#else
    RFLOAT nf = MKB_RL(0, a, alpha);
#endif
    
    int r = (maxRadius * pf) * (maxRadius * pf);

    Complex *comT3D = &T3D[0];
    RFLOAT *comT3DR = new RFLOAT[T3D.sizeFT()];
    for(size_t i = 0; i < T3D.sizeFT(); i++)
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
                          T3D.nSlcFT(),
                          r,
                          nf,
                          maxIter,
                          minIter,
                          padSize);
    
    for(size_t i = 0; i < W3D.sizeFT(); i++)
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

    Complex *comW2D = &W2D[0];
    RFLOAT *comW2DR = new RFLOAT[W2D.sizeFT()];
    for(size_t i = 0; i < W2D.sizeFT(); i++)
    {
        comW2DR[i] = REAL(comW2D[i]);
    }

    Complex *comT2D = &T2D[0];
    RFLOAT *comT2DR = new RFLOAT[T2D.sizeFT()];
    for(size_t i = 0; i < T2D.sizeFT(); i++)
    {
        comT2DR[i] = REAL(comT2D[i]);
    }
    
    LOG(INFO) << "Step2: Start Calculate W...";
      
    cuthunder::CalculateW2D(gpuIdx,
                            comT2DR,
                            comW2DR,
                            T2D.nRowFT(),
                            r);
    
    for(size_t i = 0; i < W2D.sizeFT(); i++)
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

    Complex *comW3D = &W3D[0];
    RFLOAT *comW3DR = new RFLOAT[W3D.sizeFT()];
    for(size_t i = 0; i < W3D.sizeFT(); i++)
    {
        comW3DR[i] = REAL(comW3D[i]);
    }

    Complex *comT3D = &T3D[0];
    RFLOAT *comT3DR = new RFLOAT[T3D.sizeFT()];
    for(size_t i = 0; i < T3D.sizeFT(); i++)
    {
        comT3DR[i] = REAL(comT3D[i]);
    }
    
    LOG(INFO) << "Step2: Start Calculate W...";
      
    cuthunder::CalculateW(gpuIdx,
                          comT3DR,
                          comW3DR,
                          T3D.nSlcFT(),
                          r);
    
    for(size_t i = 0; i < W3D.sizeFT(); i++)
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

    Complex *comW2D = &W2D[0];
	RFLOAT *comW2DR = new RFLOAT[W2D.sizeFT()];
	for(size_t i = 0; i < W2D.sizeFT(); i++)
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
                            padDst.nRowFT(),
                            F2D.nRowFT());
    
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

    Complex *comW3D = &W3D[0];
	RFLOAT *comW3DR = new RFLOAT[W3D.sizeFT()];
	for(size_t i = 0; i < W3D.sizeFT(); i++)
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
                          padDst.nSlcFT(),
                          F3D.nSlcFT());
    
    delete[]comW3DR;
}

void ExposeCorrF2D(int gpuIdx,
                   Image& imgDst,
                   Volume& dst,
                   RFLOAT nf,
                   RFLOAT a,
                   RFLOAT alpha,
                   int pf,
                   int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";
    
    RFLOAT *comIDst = &imgDst(0);
    Complex *comDst = &dst[0];

    int padSize = pf * size;
    int dim = imgDst.nRowRL();
    RFLOAT *mkbRL = new RFLOAT[(dim / 2 + 1) * (dim / 2 + 1)];
    
        for (int j = 0; j <= dim / 2; j++) 
            for (int i = 0; i <= dim / 2; i++) 
            {
                size_t index = j * (dim / 2 + 1) + i;
        
#ifdef RECONSTRUCTOR_MKB_KERNEL
                mkbRL[index] = MKB_RL(NORM(i, j) / padSize,
                                  a * pf,
                                  alpha);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                mkbRL[index] = TIK_RL(NORM(i, j) / padSize);
#endif
            }
    
    
    LOG(INFO) << "Step2: Start CorrSoftMaskF...";
      
    cuthunder::CorrSoftMaskF2D(gpuIdx,
                               comIDst,
                               reinterpret_cast<cuthunder::Complex*>(comDst),
                               mkbRL,
                               nf,
                               dim);

    delete[] mkbRL;
}

void ExposeCorrF(int gpuIdx,
                 Volume& dstN,
                 Volume& dst,
                 RFLOAT nf,
                 RFLOAT a,
                 RFLOAT alpha,
                 int pf,
                 int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";

    RFLOAT *comDstN = &dstN(0);
    Complex *comDst = &dst[0];

    int padSize = pf * size;
    int dim = dstN.nSlcRL();
    int slcSize = (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *mkbRL = new RFLOAT[slcSize * (dim / 2 + 1)];
    
    for (int k = 0; k <= dim / 2; k++) 
        for (int j = 0; j <= dim / 2; j++) 
            for (int i = 0; i <= dim / 2; i++) 
            {
                size_t index = k * slcSize + j * (dim / 2 + 1) + i;
        
#ifdef RECONSTRUCTOR_MKB_KERNEL
                mkbRL[index] = MKB_RL(NORM_3(i, j, k) / padSize,
                                  a * pf,
                                  alpha);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                mkbRL[index] = TIK_RL(NORM_3(i, j, k) / padSize);
#endif
            }
    
    
    LOG(INFO) << "Step2: Start CorrSoftMaskF...";
      
    cuthunder::CorrSoftMaskF(gpuIdx,
                             reinterpret_cast<cuthunder::Complex*>(comDst),
                             comDstN,
                             mkbRL,
                             nf,
                             dim);
   
    delete[] mkbRL;
}

