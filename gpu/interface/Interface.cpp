#include "Interface.h"

#include "cuthunder.h"

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

void PrepareTF(Volume& F3D,
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

    cuthunder::PrepareTF(reinterpret_cast<cuthunder::Complex*>(comF3D),
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

void ExposePT(Volume& T3D,
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

    cuthunder::CalculateT(douT3D,
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

void ExposeWT(Volume& C3D,
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
    
    Complex *comC3D = &C3D[0];
    
    RFLOAT *tabdata = kernelRL.getData();
    RFLOAT step = kernelRL.getStep();
    int padSize = pf * size; 
    
    LOG(INFO) << "Step3: Start Calculate C...";
      
    cuthunder::CalculateW(reinterpret_cast<cuthunder::Complex*>(comC3D),
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

void ExposeWT(Volume& T3D,
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
      
    cuthunder::CalculateW(comT3DR,
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

void ExposePF(Volume& padDst,
              Volume& F3D,
              Volume& W3D,
              int maxRadius,
              int pf)
{
    LOG(INFO) << "Step1: Prepare Parameter for pad.";

    Complex *comPAD = &padDst[0];
    Complex *comF3D = &F3D[0];
    
    LOG(INFO) << "Step2: Prepare Paramete for CalculateFW.";

    Complex *comW3D = &W3D[0];
	RFLOAT *comW3DR = new RFLOAT[W3D.sizeFT()];
	for(size_t i = 0; i < W3D.sizeFT(); i++)
	{
        comW3DR[i] = REAL(comW3D[i]);
	}
	
    int r = (maxRadius * pf) * (maxRadius * pf);

    LOG(INFO) << "Step4: Start PrepareF...";
    
    cuthunder::CalculateF(reinterpret_cast<cuthunder::Complex*>(comPAD),
                          reinterpret_cast<cuthunder::Complex*>(comF3D),
                          comW3DR,
                          r,
                          padDst.nSlcFT(),
                          F3D.nSlcFT());
    
    delete[]comW3DR;
}

void ExposeCorrF(Volume& padDst,
                 Volume& dst,
                 RFLOAT nf,
                 RFLOAT a,
                 RFLOAT alpha,
                 int pf,
                 int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";

    VOL_EXTRACT_RL(dst, padDst, 1.0 / pf);
   
    RFLOAT *comDst = &dst(0);

    int padSize = pf * size;
    int dim = dst.nSlcRL();
    int slcSize = (dim / 2 + 1) * (dim / 2 + 1);
    RFLOAT *mkbRL = new RFLOAT[slcSize * (dim / 2 + 1)];
    
    #pragma omp parallel for schedule(dynamic)
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
      
    cuthunder::CorrSoftMaskF(comDst,
                             mkbRL,
                             nf,
                             dim);
   
    delete[] mkbRL;
}

