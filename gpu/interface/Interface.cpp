#include "Interface.h"

#include "cuthunder.h"

void ExpectPrecal(vector<CTFAttr>& ctfAttr,
                  double* def,
                  double* k1,
                  double* k2,
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

void InsertF(Volume& F3D,
             Volume& T3D,
             MPI_Comm& hemi,
             Complex* datP,
             double* ctfP,
             double* sigRcpP,
             CTFAttr* ctfaData,
             double *offS,
             double *w,
             double *nR,
             double *nT,
             double *nD,
             const int *iCol,
             const int *iRow, 
             double pixelSize,
             bool cSearch,
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
    
    double *douT3D = new double[dimSize];
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

void ExposePT(Volume& T3D,
	          const Symmetry& sym,
	          int maxRadius,
	          int pf,
	          vec FSC,
	          bool joinHalf,
              const int wienerF)
{
	LOG(INFO) << "Step1: Prepare Parameter for NormalizeT.";

	double sf = 1.0 / REAL(T3D[0]);

	Complex *comT3D = &T3D[0];
    double *douT3D = new double[T3D.sizeFT()];
	for(size_t i = 0; i < T3D.sizeFT(); i++)
	{
        douT3D[i] = REAL(comT3D[i]);
	}
    
	
    LOG(INFO) << "Step2: Prepare Paramete for SymmetrizeT.";

	int nSymmetryElement = sym.nSymmetryElement();
    double *symMat = new double[nSymmetryElement * 9];

    mat33 L, R;   
    
	for(int i = 0; i < nSymmetryElement; i++)
	{
        sym.get(L, R, i);
        Map<mat33>(symMat + i * 9, 3, 3) = R;
	}
    
    
    LOG(INFO) << "Step3: Prepare Paramete for Calculate FSC.";

    int fscMatsize = FSC.size();
    double *FSCmat = new double[fscMatsize];
    Map<vec>(FSCmat, FSC.rows(), FSC.cols()) = FSC;
    
    LOG(INFO) << "Step4: Start PrepareT...";

    cuthunder::PrepareT(douT3D,
                        T3D.nSlcFT(),
                        FSCmat,
                        fscMatsize,
                        symMat,
                        nSymmetryElement, 
                        LINEAR_INTERP,
                        joinHalf,
                        maxRadius,
                        pf,
                        wienerF,
                        sf);
    
    for(size_t i = 0; i < T3D.sizeFT(); i++)
	{
        comT3D[i] = COMPLEX(douT3D[i], 0);
	}
    
    delete[]douT3D;
    delete[]symMat;
    delete[]FSCmat;
}

void ExposeWT(Volume& C3D,
              Volume& T3D,
              Volume& W3D,
              TabFunction& kernelRL,
              int maxRadius,
              int pf,
              double a,
              double alpha,
              int maxIter,
              int minIter,
              int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for InitialW.";
    
    Complex *comW3D = &W3D[0];
    double *comW3DR = new double[W3D.sizeFT()];
    for(size_t i = 0; i < W3D.sizeFT(); i++)
    {
        comW3DR[i] = REAL(comW3D[i]);
    }

    LOG(INFO) << "Step2: Prepare Paramete for Calculate C.";

#ifdef RECONSTRUCTOR_KERNEL_PADDING
    double nf = MKB_RL(0, a * pf, alpha);
#else
    double nf = MKB_RL(0, a, alpha);
#endif

    int r = (maxRadius * pf) * (maxRadius * pf);

    Complex *comT3D = &T3D[0];
    double *comT3DR = new double[T3D.sizeFT()];
    for(size_t i = 0; i < T3D.sizeFT(); i++)
    {
        comT3DR[i] = REAL(comT3D[i]);
    }
    
    Complex *comC3D = &C3D[0];
    
    double *tabdata = kernelRL.getData();
    double step = kernelRL.getStep();
    int padSize = pf * size; 
    
    printf("padSize:%d\n", padSize);
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
    double *comW3DR = new double[W3D.sizeFT()];
    for(size_t i = 0; i < W3D.sizeFT(); i++)
    {
        comW3DR[i] = REAL(comW3D[i]);
    }

    Complex *comT3D = &T3D[0];
    double *comT3DR = new double[T3D.sizeFT()];
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

void ExposePF(Volume& F3D,
              Volume& W3D,
              const Symmetry& sym,
              int maxRadius,
              int pf,
              double sf,
              int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for NormalizeF.";

    Complex *comF3D = &F3D[0];
    
    LOG(INFO) << "Step2: Prepare Paramete for SymmetrizeF.";

    int nSymmetryElement = sym.nSymmetryElement();
    double *symMat = new double[nSymmetryElement * 9];

    mat33 L, R;   
    
    for(int i = 0; i < nSymmetryElement; i++)
    {
        sym.get(L, R, i);
        Map<mat33>(symMat + i * 9, 3, 3) = R;
    }

    LOG(INFO) << "Step3: Prepare Paramete for CalculateFW.";

    Complex *comW3D = &W3D[0];
	double *comW3DR = new double[W3D.sizeFT()];
	for(size_t i = 0; i < W3D.sizeFT(); i++)
	{
        comW3DR[i] = REAL(comW3D[i]);
	}
	
    LOG(INFO) << "Step4: Start PrepareF...";
    
    cuthunder::PrepareF(reinterpret_cast<cuthunder::Complex*>(comF3D),
                        comW3DR,
                        sf,
                        nSymmetryElement,
                        symMat,
                        LINEAR_INTERP,
                        maxRadius,
                        EDGE_WIDTH_FT,
                        pf,
                        F3D.nSlcFT(),
                        size);
    
    delete[]comW3DR;
    delete[]symMat; 
}

void ExposeCorrF(Volume& padDst,
                 Volume& dst,
                 double a,
                 double alpha,
                 int pf,
                 int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";

    VOL_EXTRACT_RL(dst, padDst, 1.0 / pf);
   
    double *comDst = &dst(0);

    int padSize = pf * size;
    int dim = dst.nSlcRL();
    int slcSize = (dim / 2 + 1) * (dim / 2 + 1);
    double *mkbRL = new double[slcSize * (dim / 2 + 1)];
    
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
                             dim);
   
    delete[] mkbRL;
}

void ExposeCorrF(Volume& F3D,
                 Volume& dst,
                 double nf,
                 double a,
                 double alpha,
                 int pf,
                 int size)
{
    LOG(INFO) << "Step1: Prepare Parameter for CorrectingF.";

    VOL_EXTRACT_RL(dst, F3D, 1.0 / pf);
   
    double *comDst = &dst(0);

    int padSize = pf * size;
    int dim = dst.nSlcRL();
    int slcSize = (dim / 2 + 1) * (dim / 2 + 1);
    double *mkbRL = new double[slcSize * (dim / 2 + 1)];
    
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
                             dim,
                             size,
                             EDGE_WIDTH_RL);
   
    delete[] mkbRL;
}
