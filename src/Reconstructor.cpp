/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Reconstructor.h"

Reconstructor::Reconstructor()
{
    defaultInit();
}

Reconstructor::Reconstructor(const int mode,
                             const int size,
                             const int N,
                             const int pf,
                             const Symmetry* sym,
                             const RFLOAT a,
                             const RFLOAT alpha)
{
    defaultInit();

    init(mode, size, N, pf, sym, a, alpha);
}

Reconstructor::~Reconstructor()
{
#ifdef GPU_RECONSTRUCT
    if (_mode == MODE_3D)
    {
        _fft.fwDestroyPlanMT();
        _fft.bwDestroyPlanMT();
    }
#else
    _fft.fwDestroyPlanMT();
    _fft.bwDestroyPlanMT();
#endif
}

void Reconstructor::init(const int mode,
                         const int size,
                         const int N,
                         const int pf,
                         const Symmetry* sym,
                         const RFLOAT a,
                         const RFLOAT alpha)
{
    _mode = mode;
    _size = size;
    _N = N;
    _pf = pf;
    _sym = sym;

    _a = a;
    _alpha = alpha;

    // initialise the interpolation kernel
    
    ALOG(INFO, "LOGGER_RECO") << "Initialising Kernels";
    BLOG(INFO, "LOGGER_RECO") << "Initialising Kernels";

    _kernelFT.init(boost::bind(MKB_FT_R2,
                               boost::placeholders::_1,
#ifdef RECONSTRUCTOR_KERNEL_PADDING
                               _pf * _a,
#else
                               _a,
#endif
                               _alpha),
                   0,
                   TSGSL_pow_2(_pf * _a),
                   1e5);

    _kernelRL.init(boost::bind(MKB_RL_R2,
                               boost::placeholders::_1,
#ifdef RECONSTRUCTOR_KERNEL_PADDING
                               _pf * _a,
#else
                               _a,
#endif
                               _alpha),
                   0,
                   1,
                   1e5);

    _maxRadius = (_size / 2 - CEIL(a));
}

void Reconstructor::allocSpace()
{
    if (_mode == MODE_2D)
    {
        // Create Fourier Plans First, Then Allocate Space
        // For Save Memory Space

#ifndef GPU_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Creating Fourier Transform Plans";
        BLOG(INFO, "LOGGER_RECO") << "Creating Fourier Transform Plans";

        _fft.fwCreatePlanMT(PAD_SIZE, PAD_SIZE);
        _fft.bwCreatePlanMT(PAD_SIZE, PAD_SIZE);
#endif

        ALOG(INFO, "LOGGER_RECO") << "Allocating Spaces";
        BLOG(INFO, "LOGGER_RECO") << "Allocating Spaces";

        _F2D.alloc(PAD_SIZE, PAD_SIZE, FT_SPACE);
        _W2D.alloc(PAD_SIZE, PAD_SIZE, FT_SPACE);
        _C2D.alloc(PAD_SIZE, PAD_SIZE, FT_SPACE);
        _T2D.alloc(PAD_SIZE, PAD_SIZE, FT_SPACE);
    }
    else if (_mode == MODE_3D)
    {
        // Create Fourier Plans First, Then Allocate Space
        // For Save Memory Space

        ALOG(INFO, "LOGGER_RECO") << "Creating Fourier Transform Plans";
        BLOG(INFO, "LOGGER_RECO") << "Creating Fourier Transform Plans";

        _fft.fwCreatePlanMT(PAD_SIZE, PAD_SIZE, PAD_SIZE);
        _fft.bwCreatePlanMT(PAD_SIZE, PAD_SIZE, PAD_SIZE);

        ALOG(INFO, "LOGGER_RECO") << "Allocating Spaces";
        BLOG(INFO, "LOGGER_RECO") << "Allocating Spaces";

        _F3D.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
        _W3D.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
        _C3D.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
        _T3D.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);

    }
    else 
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    reset();
}

void Reconstructor::freeSpace()
{
    if (_mode == MODE_2D)
    {
        ALOG(INFO, "LOGGER_RECO") << "Freeing Spaces";
        BLOG(INFO, "LOGGER_RECO") << "Freeing Spaces";

#ifndef GPU_RECONSTRUCT
        _fft.fwDestroyPlanMT();
        _fft.bwDestroyPlanMT();
#endif

        _F2D.clear();
        _W2D.clear();
        _C2D.clear();
        _T2D.clear();
    }
    else if (_mode == MODE_3D)
    {
        ALOG(INFO, "LOGGER_RECO") << "Freeing Spaces";
        BLOG(INFO, "LOGGER_RECO") << "Freeing Spaces";

        _fft.fwDestroyPlanMT();
        _fft.bwDestroyPlanMT();
        
        _F3D.clear();
        _W3D.clear();
        _C3D.clear();
        _T3D.clear();

    }
    else 
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
}

void Reconstructor::resizeSpace(const int size)
{
#ifdef GPU_RECONSTRUCT
    if (_mode == MODE_3D)
    {
        _fft.fwDestroyPlanMT();
        _fft.bwDestroyPlanMT();
    }
#else
    _fft.fwDestroyPlanMT();
    _fft.bwDestroyPlanMT();
#endif

    _size = size;
}

void Reconstructor::reset()
{
    _iCol = NULL;
    _iRow = NULL;
    _iPxl = NULL;
    _iSig = NULL;

    _calMode = POST_CAL_MODE;

    _MAP = true;

    _gridCorr = true;

    _joinHalf = false;

    if (_mode == MODE_2D)
    {
        #pragma omp parallel for
        SET_0_FT(_F2D);

        #pragma omp parallel for
        SET_1_FT(_W2D);

        #pragma omp parallel for
        SET_0_FT(_C2D);

        #pragma omp parallel for
        SET_0_FT(_T2D);
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for
        SET_0_FT(_F3D);

        #pragma omp parallel for
        SET_1_FT(_W3D);

        #pragma omp parallel for
        SET_0_FT(_C3D);

        #pragma omp parallel for
        SET_0_FT(_T3D);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    _ox = 0;
    _oy = 0;
    _oz = 0;

    _counter = 0;
}

int Reconstructor::mode() const
{
    return _mode;
}

void Reconstructor::setMode(const int mode)
{
    _mode = mode;
}

bool Reconstructor::MAP() const
{
    return _MAP;
}

void Reconstructor::setMAP(const bool MAP)
{
    _MAP = MAP;
}

bool Reconstructor::gridCorr() const
{
    return _gridCorr;
}

void Reconstructor::setGridCorr(const bool gridCorr)
{
    _gridCorr = gridCorr;
}

bool Reconstructor::joinHalf() const
{
    return _joinHalf;
}

void Reconstructor::setJoinHalf(const bool joinHalf)
{
    _joinHalf = joinHalf;
}

void Reconstructor::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Reconstructor::setFSC(const vec& FSC)
{
    _FSC = FSC;
}

void Reconstructor::setTau(const vec& tau)
{
    _tau = tau;
}

void Reconstructor::setSig(const vec& sig)
{
    _sig = sig;
}

void Reconstructor::setOx(const double ox)
{
    _ox = ox;
}

void Reconstructor::setOy(const double oy)
{
    _oy = oy;
}

void Reconstructor::setOz(const double oz)
{
    _oz = oz;
}

void Reconstructor::setCounter(const int counter)
{
    _counter = counter;
}

double Reconstructor::ox() const
{
    return _ox;
}

double Reconstructor::oy() const
{
    return _oy;
}

double Reconstructor::oz() const
{
    return _oz;
}

int Reconstructor::counter() const
{
    return _counter;
}

int Reconstructor::maxRadius() const
{
    return _maxRadius;
}

void Reconstructor::setMaxRadius(const int maxRadius)
{
    _maxRadius = maxRadius;
}

void Reconstructor::preCal(int& nPxl,
                           const int* iCol,
                           const int* iRow,
                           const int* iPxl,
                           const int* iSig) const
{
    nPxl = _nPxl;

    iCol = _iCol;
    iRow = _iRow;
    iPxl = _iPxl;
    iSig = _iSig;
}

void Reconstructor::setPreCal(const int nPxl,
                              const int* iCol,
                              const int* iRow,
                              const int* iPxl,
                              const int* iSig)
{
    _calMode = PRE_CAL_MODE;

    _nPxl = nPxl;

    _iCol = iCol;
    _iRow = iRow;
    _iPxl = iPxl;
    _iSig = iSig;
}

void Reconstructor::insertDir(const dvec2& dir)
{
    insertDir(dir(0), dir(1), 0);
}

void Reconstructor::insertDir(const dvec3& dir)
{
    insertDir(dir(0), dir(1), dir(2));
}

void Reconstructor::insertDir(const double ox,
                              const double oy,
                              const double oz)
{
    #pragma omp atomic
    _ox += ox;

    #pragma omp atomic
    _oy += oy;

    #pragma omp atomic
    _oz += oz;

    #pragma omp atomic
    _counter +=1;
}

void Reconstructor::insert(const Image& src,
                           const Image& ctf,
                           const dmat22& rot,
                           const RFLOAT w)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_2D REPORT_ERROR("WRONG MODE");

    if (_calMode != POST_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");

    if ((src.nColRL() != _size) ||
        (src.nRowRL() != _size) ||
        (ctf.nColRL() != _size) ||
        (ctf.nRowRL() != _size))
        REPORT_ERROR("INCORRECT SIZE OF INSERTING IMAGE");
#endif

    IMAGE_FOR_EACH_PIXEL_FT(src)
    {
        if (QUAD(i, j) < gsl_pow_2(_maxRadius))
        {
            dvec2 newCor((double)(i * _pf), (double)(j * _pf));
            dvec2 oldCor = rot * newCor;

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _F2D.addFT(src.getFTHalf(i, j)
                     * REAL(ctf.getFTHalf(i, j))
                     * w, 
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a, 
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _F2D.addFT(src.getFTHalf(i, j)
                     * REAL(ctf.getFTHalf(i, j))
                     * w, 
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _T2D.addFT(TSGSL_pow_2(REAL(ctf.getFTHalf(i, j)))
                     * w, 
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a,
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
             _T2D.addFT(TSGSL_pow_2(REAL(ctf.getFTHalf(i, j)))
                      * w, 
                        (RFLOAT)oldCor(0), 
                        (RFLOAT)oldCor(1));
#endif

#endif
        }
    }
}

void Reconstructor::insert(const Image& src,
                           const Image& ctf,
                           const dmat33& rot,
                           const RFLOAT w)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_3D REPORT_ERROR("WRONG MODE");

    if (_calMode != POST_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");

    if ((src.nColRL() != _size) ||
        (src.nRowRL() != _size) ||
        (ctf.nColRL() != _size) ||
        (ctf.nRowRL() != _size))
        REPORT_ERROR("INCORRECT SIZE OF INSERTING IMAGE");
#endif

        IMAGE_FOR_EACH_PIXEL_FT(src)
        {
            if (QUAD(i, j) < gsl_pow_2(_maxRadius))
            {
                const double* ptr = rot.data();
                double oldCor[3];
                oldCor[0] = (ptr[0] * i + ptr[3] * j) * _pf;
                oldCor[1] = (ptr[1] * i + ptr[4] * j) * _pf;
                oldCor[2] = (ptr[2] * i + ptr[5] * j) * _pf;

#ifdef RECONSTRUCTOR_MKB_KERNEL
                _F3D.addFT(src.getFTHalf(i, j)
                         * REAL(ctf.getFTHalf(i, j))
                         * w, 
                           (RFLOAT)oldCor[0], 
                           (RFLOAT)oldCor[1], 
                           (RFLOAT)oldCor[2], 
                           _pf * _a, 
                           _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                _F3D.addFT(src.getFTHalf(i, j)
                         * REAL(ctf.getFTHalf(i, j))
                         * w, 
                           (RFLOAT)oldCor[0], 
                           (RFLOAT)oldCor[1], 
                           (RFLOAT)oldCor[2]);
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
                _T3D.addFT(TSGSL_pow_2(REAL(ctf.getFTHalf(i, j)))
                         * w, 
                           (RFLOAT)oldCor[0], 
                           (RFLOAT)oldCor[1], 
                           (RFLOAT)oldCor[2],
                           _pf * _a,
                           _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                _T3D.addFT(TSGSL_pow_2(REAL(ctf.getFTHalf(i, j)))
                         * w, 
                           (RFLOAT)oldCor[0], 
                           (RFLOAT)oldCor[1], 
                           (RFLOAT)oldCor[2]);
#endif

#endif
            }
        }
}

void Reconstructor::insertP(const Image& src,
                            const Image& ctf,
                            const dmat22& rot,
                            const RFLOAT w,
                            const vec* sig)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_2D REPORT_ERROR("WRONG MODE");

    if (_calMode != PRE_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");
#endif

        for (int i = 0; i < _nPxl; i++)
        {
            dvec2 newCor((double)(_iCol[i]), (double)(_iRow[i]));
            dvec2 oldCor = rot * newCor;

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _F2D.addFT(src.iGetFT(_iPxl[i])
                     * REAL(ctf.iGetFT(_iPxl[i]))
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a, 
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _F2D.addFT(src.iGetFT(_iPxl[i])
                     * REAL(ctf.iGetFT(_iPxl[i]))
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _T2D.addFT(TSGSL_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a,
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _T2D.addFT(TSGSL_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#endif
        }
}

void Reconstructor::insertP(const Image& src,
                            const Image& ctf,
                            const dmat33& rot,
                            const RFLOAT w,
                            const vec* sig)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_3D REPORT_ERROR("WRONG MODE");

    if (_calMode != PRE_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");
#endif

    for (int i = 0; i < _nPxl; i++)
    {
        const double* ptr = rot.data();
        double oldCor[3];
        int iCol = _iCol[i];
        int iRow = _iRow[i];
        oldCor[0] = ptr[0] * iCol + ptr[3] * iRow;
        oldCor[1] = ptr[1] * iCol + ptr[4] * iRow;
        oldCor[2] = ptr[2] * iCol + ptr[5] * iRow;

#ifdef RECONSTRUCTOR_MKB_KERNEL
        _F3D.addFT(src.iGetFT(_iPxl[i])
                 * REAL(ctf.iGetFT(_iPxl[i]))
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0],
                   (RFLOAT)oldCor[1],
                   (RFLOAT)oldCor[2],
                   _pf * _a, 
                   _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        _F3D.addFT(src.iGetFT(_iPxl[i])
                 * REAL(ctf.iGetFT(_iPxl[i]))
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0],
                   (RFLOAT)oldCor[1],
                   (RFLOAT)oldCor[2]);
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
        _T3D.addFT(TSGSL_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0],
                   (RFLOAT)oldCor[1],
                   (RFLOAT)oldCor[2],
                   _pf * _a,
                   _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        _T3D.addFT(TSGSL_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0],
                   (RFLOAT)oldCor[1],
                   (RFLOAT)oldCor[2]);
#endif

#endif
    }
}

void Reconstructor::insertP(const Complex* src,
                            const RFLOAT* ctf,
                            const dmat22& rot,
                            const RFLOAT w,
                            const vec* sig)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_2D REPORT_ERROR("WRONG MODE");

    if (_calMode != PRE_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");
#endif

        for (int i = 0; i < _nPxl; i++)
        {
            dvec2 newCor((double)(_iCol[i]), (double)(_iRow[i]));
            dvec2 oldCor = rot * newCor;

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _F2D.addFT(src[i]
                     * ctf[i]
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a, 
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _F2D.addFT(src[i]
                     * ctf[i]
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
            _T2D.addFT(TSGSL_pow_2(ctf[i])
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1), 
                       _pf * _a,
                       _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            _T2D.addFT(TSGSL_pow_2(ctf[i])
                     * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                     * w,
                       (RFLOAT)oldCor(0), 
                       (RFLOAT)oldCor(1));
#endif

#endif
        }
}

void Reconstructor::insertP(const Complex* src,
                            const RFLOAT* ctf,
                            const dmat33& rot,
                            const RFLOAT w,
                            const vec* sig)
{
#ifdef RECONSTRUCTOR_ASSERT_CHECK
    IF_MASTER
        REPORT_ERROR("INSERTING IMAGES INTO RECONSTRUCTOR IN MASTER");

    NT_MODE_3D REPORT_ERROR("WRONG MODE");

    if (_calMode != PRE_CAL_MODE)
        REPORT_ERROR("WRONG PRE(POST) CALCULATION MODE IN RECONSTRUCTOR");
#endif

    for (int i = 0; i < _nPxl; i++)
    {
        const double* ptr = rot.data();
        double oldCor[3];
        int iCol = _iCol[i];
        int iRow = _iRow[i];
        oldCor[0] = ptr[0] * iCol + ptr[3] * iRow;
        oldCor[1] = ptr[1] * iCol + ptr[4] * iRow;
        oldCor[2] = ptr[2] * iCol + ptr[5] * iRow;

#ifdef RECONSTRUCTOR_MKB_KERNEL
        _F3D.addFT(src[i]
                 * ctf[i]
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0], 
                   (RFLOAT)oldCor[1], 
                   (RFLOAT)oldCor[2], 
                   _pf * _a, 
                   _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        _F3D.addFT(src[i]
                 * ctf[i]
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0], 
                   (RFLOAT)oldCor[1], 
                   (RFLOAT)oldCor[2]);
#endif

#ifdef RECONSTRUCTOR_ADD_T_DURING_INSERT

#ifdef RECONSTRUCTOR_MKB_KERNEL
        _T3D.addFT(TSGSL_pow_2(ctf[i])
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0], 
                   (RFLOAT)oldCor[1], 
                   (RFLOAT)oldCor[2],
                   _pf * _a,
                   _kernelFT);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
        _T3D.addFT(TSGSL_pow_2(ctf[i])
                 * (sig == NULL ? 1 : (*sig)(_iSig[i]))
                 * w,
                   (RFLOAT)oldCor[0], 
                   (RFLOAT)oldCor[1], 
                   (RFLOAT)oldCor[2]);
#endif

#endif
    }
}

#ifdef GPU_INSERT

void Reconstructor::insertI(Complex* datP,
                            RFLOAT* ctfP,
                            RFLOAT* sigP,
                            RFLOAT* w,
                            double* offS,
                            double* nr,
                            double* nt,
                            double* nd,
                            int* nc,
                            CTFAttr* ctfaData, 
                            RFLOAT pixelSize,
                            bool cSearch,
                            int opf,
                            int mReco,
                            int idim,
                            int imgNum)
{   
    double* O3D = new double[3];
    int* counter = new int[1];
    O3D[0] = _ox;
    O3D[1] = _oy;
    O3D[2] = _oz;
    counter[0] = _counter;
    int dimSize = _T3D.sizeFT();
    
    InsertFT(_F3D,
             _T3D,
             O3D,
             counter,
             _hemi,
             _slav,
             datP,
             ctfP,
             sigP,
             ctfaData,
             offS,
             w,
             nr,
             nt,
             nd,
             nc,
             _iCol,
             _iRow,
             pixelSize,
             cSearch,
             opf,
             _nPxl,
             mReco,
             idim,
             dimSize,
             imgNum);
    
    _ox = O3D[0];
    _oy = O3D[1];
    _oz = O3D[2];
    _counter = counter[0];
    
    delete[] O3D;
    delete[] counter;
}

void Reconstructor::insertI(Complex* datP,
                            RFLOAT* ctfP,
                            RFLOAT* sigP,
                            RFLOAT* w,
                            double* offS,
                            double* nr,
                            double* nt,
                            double* nd,
                            CTFAttr* ctfaData, 
                            RFLOAT pixelSize,
                            bool cSearch,
                            int opf,
                            int mReco,
                            int idim,
                            int imgNum)
{   
    double* O3D = new double[3];
    int* counter = new int[1];
    O3D[0] = _ox;
    O3D[1] = _oy;
    O3D[2] = _oz;
    counter[0] = _counter;
    int dimSize = _T3D.sizeFT();
    
    InsertFT(_F3D,
             _T3D,
             O3D,
             counter,
             _hemi,
             _slav,
             datP,
             ctfP,
             sigP,
             ctfaData,
             offS,
             w,
             nr,
             nt,
             nd,
             _iCol,
             _iRow,
             pixelSize,
             cSearch,
             opf,
             _nPxl,
             mReco,
             idim,
             dimSize,
             imgNum);
    
    _ox = O3D[0];
    _oy = O3D[1];
    _oz = O3D[2];
    _counter = counter[0];
    
    delete[] O3D;
    delete[] counter;
}

int Reconstructor::getModelDim()
{
    return _F2D.nRowFT();
}

int Reconstructor::getModelSize()
{
    return _F2D.sizeFT();
}

void Reconstructor::getF(Complex* modelF)
{
    for(size_t i = 0; i < _F2D.sizeFT(); i++)
        modelF[i] = _F2D[i];
}

void Reconstructor::getT(RFLOAT* modelT)
{
    for(size_t i = 0; i < _T2D.sizeFT(); i++)
        modelT[i] = REAL(_T2D[i]);
}

void Reconstructor::resetF(Complex* modelF)
{
    for(size_t i = 0; i < _F2D.sizeFT(); i++)
        _F2D[i] = modelF[i];
}

void Reconstructor::resetT(RFLOAT* modelT)
{
    for(size_t i = 0; i < _T2D.sizeFT(); i++)
        _T2D[i] = COMPLEX(modelT[i], 0);
}

void Reconstructor::prepareTFG(int gpuIdx)
{
    IF_MASTER return;

    // only in 3D mode, symmetry should be considered
    IF_MODE_3D
    {
	    int nSymmetryElement = _sym->nSymmetryElement();
        double *symMat = new double[nSymmetryElement * 9];

#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Prepare param for Symmetrizing TF";
        BLOG(INFO, "LOGGER_RECO") << "Prepare param for Symmetrizing TF";
        dmat33 L, R;   
        
	    for(int i = 0; i < nSymmetryElement; i++)
	    {
            _sym->get(L, R, i);
            Map<dmat33>(symMat + i * 9, 3, 3) = R;
	    }
#endif
        PrepareTF(gpuIdx,
                  _F3D,
                  _T3D,
                  symMat,
                  nSymmetryElement,
                  _maxRadius,
                  _pf);
        
        delete[]symMat;
    }
}

#endif // GPU_INSERT

void Reconstructor::prepareTF()
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_RECO") << "Allreducing T";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing T";

    allReduceT();

    // only in 3D mode, symmetry should be considered
    IF_MODE_3D
    {
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Symmetrizing T";
        BLOG(INFO, "LOGGER_RECO") << "Symmetrizing T";

        symmetrizeT();
#endif
    }

    ALOG(INFO, "LOGGER_RECO") << "Allreducing F";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing F";

    allReduceF();

    // only in 3D mode, symmetry should be considered
    IF_MODE_3D
    {
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Symmetrizing F";
        BLOG(INFO, "LOGGER_RECO") << "Symmetrizing F";

        symmetrizeF();
#endif
    }
}

void Reconstructor::reconstruct(Image& dst)
{
    Volume tmp;

    reconstruct(tmp);

    dst.alloc(PAD_SIZE, PAD_SIZE, RL_SPACE);

    SLC_EXTRACT_RL(dst, tmp, 0);
}

void Reconstructor::prepareO()
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_RECO") << "Allreducing O";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing O";

    allReduceO();

    IF_MODE_3D
    {
#ifdef RECONSTRUCTOR_SYMMETRIZE_DURING_RECONSTRUCT
        ALOG(INFO, "LOGGER_RECO") << "Symmetrizing O";
        BLOG(INFO, "LOGGER_RECO") << "Symmetrizing O";

        symmetrizeO();
#endif
    }

    _ox /= _counter;
    _oy /= _counter;
    _oz /= _counter;
}

void Reconstructor::reconstruct(Volume& dst)
{
    IF_MASTER return;

#ifdef VERBOSE_LEVEL_2

    IF_MODE_2D
    {
        ALOG(INFO, "LOGGER_RECO") << "Reconstructing Under 2D Mode";
        BLOG(INFO, "LOGGER_RECO") << "Reconstructing Under 2D Mode";
    }

    IF_MODE_3D
    {
        ALOG(INFO, "LOGGER_RECO") << "Reconstructing Under 3D Mode";
        BLOG(INFO, "LOGGER_RECO") << "Reconstructing Under 3D Mode";
    }

#endif

    // only in 3D mode, the MAP method is appropriate
    //if (_MAP && (_mode == MODE_3D))
    if (_MAP)
    {
        // Obviously, wiener_filter with FSC can be wrong when dealing with
        // preferrable orienation problem
#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
        vec avg = vec::Zero(_maxRadius * _pf + 1);

        if (_mode == MODE_2D)
        {
            ringAverage(avg,
                        _T2D,
                        gsl_real,
                        _maxRadius * _pf - 1);
        }
        else if (_mode == MODE_3D)
        {
            shellAverage(avg,
                         _T3D,
                         gsl_real,
                         _maxRadius * _pf - 1);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }

        // the last two elements have low fidelity
        avg(_maxRadius * _pf - 1) = avg(_maxRadius * _pf - 2);
        avg(_maxRadius * _pf) = avg(_maxRadius * _pf - 2);

#ifdef VERBOSE_LEVEL_2
        ALOG(INFO, "LOGGER_SYS") << "End of Avg = "
                                 << avg(avg.size() - 5) << ", "
                                 << avg(avg.size() - 4) << ", "
                                 << avg(avg.size() - 3) << ", "
                                 << avg(avg.size() - 2) << ", "
                                 << avg(avg.size() - 1);
        BLOG(INFO, "LOGGER_SYS") << "End of Avg = "
                                 << avg(avg.size() - 5) << ", "
                                 << avg(avg.size() - 4) << ", "
                                 << avg(avg.size() - 3) << ", "
                                 << avg(avg.size() - 2) << ", "
                                 << avg(avg.size() - 1);
#endif

#endif

#ifdef VERBOSE_LEVEL_2
        ALOG(INFO, "LOGGER_SYS") << "End of FSC = " << _FSC(_FSC.size() - 1);
        BLOG(INFO, "LOGGER_SYS") << "End of FSC = " << _FSC(_FSC.size() - 1);
#endif

        if (_mode == MODE_2D)
        {
            #pragma omp parallel for schedule(dynamic)
            IMAGE_FOR_EACH_PIXEL_FT(_T2D)
                if ((QUAD(i, j) >= TSGSL_pow_2(WIENER_FACTOR_MIN_R * _pf)) &&
                    (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf)))
                {
                    int u = AROUND(NORM(i, j));

                    RFLOAT FSC = (u / _pf >= _FSC.size())
                               ? 0
                               : _FSC(u / _pf);

                    FSC = TSGSL_MAX_RFLOAT(FSC_BASE_L, TSGSL_MIN_RFLOAT(FSC_BASE_H, FSC));

#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
                    FSC = sqrt(2 * FSC / (1 + FSC));
#else
                    if (_joinHalf) FSC = sqrt(2 * FSC / (1 + FSC));
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
                    _T2D.setFT(_T2D.getFT(i, j)
                             + COMPLEX((1 - FSC) / FSC * avg(u), 0),
                               i,
                               j);
#else
                    _T2D.setFT(_T2D.getFT(i, j) / FSC, i, j);
#endif
                }
        }
        else if (_mode == MODE_3D)
        {
            #pragma omp parallel for schedule(dynamic)
            VOLUME_FOR_EACH_PIXEL_FT(_T3D)
                if ((QUAD_3(i, j, k) >= TSGSL_pow_2(WIENER_FACTOR_MIN_R * _pf)) &&
                    (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf)))
                {
                    int u = AROUND(NORM_3(i, j, k));

                    RFLOAT FSC = (u / _pf >= _FSC.size())
                               ? 0
                               : _FSC(u / _pf);

                    FSC = TSGSL_MAX_RFLOAT(FSC_BASE_L, TSGSL_MIN_RFLOAT(FSC_BASE_H, FSC));

#ifdef RECONSTRUCTOR_ALWAYS_JOIN_HALF
                    FSC = sqrt(2 * FSC / (1 + FSC));
#else
                    if (_joinHalf) FSC = sqrt(2 * FSC / (1 + FSC));
#endif

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC_FREQ_AVG
                    _T3D.setFT(_T3D.getFT(i, j, k)
                             + COMPLEX((1 - FSC) / FSC * avg(u), 0),
                               i,
                               j,
                               k);
#else
                    _T3D.setFT(_T3D.getFT(i, j, k) / FSC, i, j, k);
#endif
                }
        }
        else
        {
            REPORT_ERROR("INEXISTENT_MODE");

            abort();
        }
#endif
    }

#ifdef VERBOSE_LEVEL_2

    ALOG(INFO, "LOGGER_RECO") << "Initialising W";
    BLOG(INFO, "LOGGER_RECO") << "Initialising W";

#endif

    if (_mode == MODE_2D)
    {
        #pragma omp parallel for
        IMAGE_FOR_EACH_PIXEL_FT(_W2D)
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
                _W2D.setFTHalf(COMPLEX(1, 0), i, j);
            else
                _W2D.setFTHalf(COMPLEX(0, 0), i, j);
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for
        VOLUME_FOR_EACH_PIXEL_FT(_W3D)
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
                _W3D.setFTHalf(COMPLEX(1, 0), i, j, k);
            else
                _W3D.setFTHalf(COMPLEX(0, 0), i, j, k);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    if (_gridCorr)
    {
        RFLOAT diffC = TS_MAX_RFLOAT_VALUE;
        RFLOAT diffCPrev = TS_MAX_RFLOAT_VALUE;

        int m = 0;

        int nDiffCNoDecrease = 0;

        for (m = 0; m < MAX_N_ITER_BALANCE; m++)
        {
#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_RECO") << "Balancing Weights Round " << m;
            BLOG(INFO, "LOGGER_RECO") << "Balancing Weights Round " << m;

            ALOG(INFO, "LOGGER_RECO") << "Determining C";
            BLOG(INFO, "LOGGER_RECO") << "Determining C";

#endif
        
            if (_mode == MODE_2D)
            {
                #pragma omp parallel for
                FOR_EACH_PIXEL_FT(_C2D)
                    _C2D[i] = _T2D[i] * REAL(_W2D[i]);
            }
            else if (_mode == MODE_3D)
            {
                #pragma omp parallel for
                FOR_EACH_PIXEL_FT(_C3D)
                    _C3D[i] = _T3D[i] * REAL(_W3D[i]);
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }

#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_RECO") << "Convoluting C";
            BLOG(INFO, "LOGGER_RECO") << "Convoluting C";

#endif

            convoluteC();

#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_RECO") << "Re-Calculating W";
            BLOG(INFO, "LOGGER_RECO") << "Re-Calculating W";

#endif

            if (_mode == MODE_2D)
            {
                #pragma omp parallel for schedule(dynamic)
                IMAGE_FOR_EACH_PIXEL_FT(_W2D)
                    if (QUAD(i, j) < gsl_pow_2(_maxRadius * _pf))
                        _W2D.setFTHalf(_W2D.getFTHalf(i, j)
                                     / TSGSL_MAX_RFLOAT(ABS(_C2D.getFTHalf(i, j)),
                                                   1e-6),
                                       i,
                                       j);
            }
            else if (_mode == MODE_3D)
            {
                #pragma omp parallel for schedule(dynamic)
                VOLUME_FOR_EACH_PIXEL_FT(_W3D)
                    if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
                        _W3D.setFTHalf(_W3D.getFTHalf(i, j, k)
                                     / TSGSL_MAX_RFLOAT(ABS(_C3D.getFTHalf(i, j, k)),
                                                   1e-6),
                                       i,
                                       j,
                                       k);
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }

#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_RECO") << "Calculating Distance to Total Balanced";
            BLOG(INFO, "LOGGER_RECO") << "Calculating Distance to Total Balanced";

#endif

            diffCPrev = diffC;

            diffC = checkC();
 
#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_SYS") << "After "
                                     << m
                                     << " Iterations, Distance to Total Balanced: "
                                     << diffC;
            BLOG(INFO, "LOGGER_SYS") << "After "
                                     << m
                                     << " Iterations, Distance to Total Balanced: "
                                     << diffC;

#endif

#ifdef VERBOSE_LEVEL_2

            ALOG(INFO, "LOGGER_RECO") << "Distance to Total Balanced: " << diffC;
            BLOG(INFO, "LOGGER_RECO") << "Distance to Total Balanced: " << diffC;

#endif

            if (diffC > diffCPrev * DIFF_C_DECREASE_THRES)
                nDiffCNoDecrease += 1;
            else
                nDiffCNoDecrease = 0;

            if ((diffC < DIFF_C_THRES) ||
                ((m >= MIN_N_ITER_BALANCE) &&
                (nDiffCNoDecrease == N_DIFF_C_NO_DECREASE))) break;
        }
    }
    else
    {
        // no grid correction
        if (_mode == MODE_2D)
        {
            #pragma omp parallel for schedule(dynamic)
            IMAGE_FOR_EACH_PIXEL_FT(_W2D)
                if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
                    _W2D.setFTHalf(COMPLEX(1.0
                                         / TSGSL_MAX_RFLOAT(ABS(_T2D.getFTHalf(i, j)),
                                                       1e-6),
                                           0),
                                   i,
                                   j);
        }
        else if (_mode == MODE_3D)
        {
            #pragma omp parallel for schedule(dynamic)
            VOLUME_FOR_EACH_PIXEL_FT(_W3D)
                if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
                    _W3D.setFTHalf(COMPLEX(1.0
                                         / TSGSL_MAX_RFLOAT(ABS(_T3D.getFTHalf(i, j, k)),
                                                       1e-6),
                                           0),
                                   i,
                                   j,
                                   k);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }

    if (_mode == MODE_2D)
    {
#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Image";
        BLOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Image";

#endif

        Image padDst(_N * _pf, _N * _pf, FT_SPACE);

        #pragma omp parallel
        SET_0_FT(padDst);

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";

#endif

        #pragma omp parallel for schedule(dynamic)
        IMAGE_FOR_EACH_PIXEL_FT(_F2D)
        {
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
            {
                padDst.setFTHalf(_F2D.getFTHalf(i, j)
                               * _W2D.getFTHalf(i, j),
                                 i,
                                 j);
            }
        }

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming Padded Destination Image";
        BLOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming Padded Destination Image";

#endif

        FFT fft;
        fft.bwMT(padDst);

        Image imgDst;

        IMG_EXTRACT_RL(imgDst, padDst, 1.0 / _pf);

        dst.alloc(_N, _N, 1, RL_SPACE);

        #pragma omp parallel
        IMAGE_FOR_EACH_PIXEL_RL(imgDst)
            dst.setRL(imgDst.getRL(i, j), i, j, 0);
    }
    else if (_mode == MODE_3D)
    {
#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Volume";

#endif

        Volume padDst(_N * _pf, _N * _pf, _N * _pf, FT_SPACE);

        #pragma omp parallel
        SET_0_FT(padDst);

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";

#endif

        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_FT(_F3D)
        {
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
            {
                padDst.setFTHalf(_F3D.getFTHalf(i, j, k)
                               * _W3D.getFTHalf(i, j ,k),
                                 i,
                                 j,
                                 k);
            }
        }

#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming Padded Destination Volume";

#endif

        FFT fft;
        fft.bwMT(padDst);
        
#ifdef VERBOSE_LEVEL_2

        ALOG(INFO, "LOGGER_RECO") << "Extracting Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Extracting Destination Volume";

#endif

        VOL_EXTRACT_RL(dst, padDst, 1.0 / _pf);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

#ifdef RECONSTRUCTOR_CORRECT_CONVOLUTION_KERNEL

#ifdef VERBOSE_LEVEL_2

    ALOG(INFO, "LOGGER_RECO") << "Correcting Convolution Kernel";
    BLOG(INFO, "LOGGER_RECO") << "Correcting Convolution Kernel";

#endif

#ifdef RECONSTRUCTOR_MKB_KERNEL
    RFLOAT nf = MKB_RL(0, _a * _pf, _alpha);
#endif

    if (_mode == MODE_2D)
    {
        Image imgDst(_N, _N, RL_SPACE);

        SLC_EXTRACT_RL(imgDst, dst, 0);

        #pragma omp parallel for schedule(dynamic)
        IMAGE_FOR_EACH_PIXEL_RL(imgDst)
        {
#ifdef RECONSTRUCTOR_MKB_KERNEL
            imgDst.setRL(imgDst.getRL(i, j)
                       / MKB_RL(NORM(i, j) / (_pf * _N),
                                _a * _pf,
                                _alpha)
                       * nf,
                         i,
                         j);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            imgDst.setRL(imgDst.getRL(i, j)
                       / TIK_RL(NORM(i, j) / (_pf * _N)),
                         i,
                         j);
#endif
        }

        SLC_REPLACE_RL(dst, imgDst, 0);
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_RL(dst)
        {
#ifdef RECONSTRUCTOR_MKB_KERNEL
            dst.setRL(dst.getRL(i, j, k)
                     / MKB_RL(NORM_3(i, j, k) / (_pf * _N),
                              _a * _pf,
                              _alpha)
                     * nf,
                       i,
                       j,
                       k);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
            dst.setRL(dst.getRL(i, j, k)
                     / TIK_RL(NORM_3(i, j, k) / (_pf * _N)),
                       i,
                       j,
                       k);
#endif
        }
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

#ifdef VERBOSE_LEVEL_2

    ALOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";
    BLOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";

#endif

#endif

#ifdef RECONSTRUCTOR_REMOVE_NEG
    ALOG(INFO, "LOGGER_RECO") << "Removing Negative Values";
    BLOG(INFO, "LOGGER_RECO") << "Removing Negative Values";

    #pragma omp parallel for
    REMOVE_NEG(dst);
#endif
}

#ifdef GPU_RECONSTRUCT

void Reconstructor::reconstructG(Volume& dst,
                                 int gpuIdx)
{
    IF_MASTER return;

#ifdef VERBOSE_LEVEL_2
    IF_MODE_2D
    {
        ALOG(INFO, "LOGGER_RECO") << "Reconstructing Under 2D Mode";
        BLOG(INFO, "LOGGER_RECO") << "Reconstructing Under 2D Mode";
    }

    IF_MODE_3D
    {
        ALOG(INFO, "LOGGER_RECO") << "Reconstructing Under 3D Mode";
        BLOG(INFO, "LOGGER_RECO") << "Reconstructing Under 3D Mode";
    }
#endif
    
    RFLOAT* volumeT;
    RFLOAT* volumeW;

    if (_mode == MODE_2D)
    {
        size_t dimSize = _T2D.sizeFT();
        volumeT = (RFLOAT*)malloc(dimSize * sizeof(RFLOAT));
        volumeW = (RFLOAT*)malloc(dimSize * sizeof(RFLOAT));
	    
        for(size_t i = 0; i < dimSize; i++)
	    {
            volumeT[i] = REAL(_T2D[i]);
            volumeW[i] = REAL(_W2D[i]);
	    }
    }
    else if (_mode == MODE_3D)
    {
        size_t dimSize = _T3D.sizeFT();
        volumeT = (RFLOAT*)malloc(dimSize * sizeof(RFLOAT));
        volumeW = (RFLOAT*)malloc(dimSize * sizeof(RFLOAT));
	    
        for(size_t i = 0; i < dimSize; i++)
	    {
            volumeT[i] = REAL(_T3D[i]);
            volumeW[i] = REAL(_W3D[i]);
	    }
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

#ifdef RECONSTRUCTOR_WIENER_FILTER_FSC
    // only in 3D mode, the MAP method is appropriate
    if (_MAP)
    {
        if (_mode == MODE_2D)
        {
            ExposePT2D(gpuIdx,
                       volumeT,
                       _maxRadius,
                       _pf,
                       _T2D.nRowFT(),
                       _FSC,
                       _joinHalf,
                       WIENER_FACTOR_MIN_R);
        }
        else if (_mode == MODE_3D)
        {
            ExposePT(gpuIdx,
                     volumeT,
                     _maxRadius,
                     _pf,
                     _T3D.nSlcFT(),
                     _FSC,
                     _joinHalf,
                     WIENER_FACTOR_MIN_R);
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }
#endif
 
    if (_mode == MODE_2D)
    {
        for (size_t i = 0; i < _T2D.sizeFT(); i++)
	    {
            _T2D[i] = COMPLEX(volumeT[i], 0);
	    }
    }
    else if (_mode == MODE_3D)
    {
        for (size_t i = 0; i < _T3D.sizeFT(); i++)
	    {
            _T3D[i] = COMPLEX(volumeT[i], 0);
	    }
    }
    
    if (_gridCorr)
    {
#ifdef RECONSTRUCTOR_KERNEL_PADDING
        RFLOAT nf = MKB_RL(0, _a * _pf, _alpha);
#else
        RFLOAT nf = MKB_RL(0, _a, _alpha);
#endif
        if (_mode == MODE_2D)
        {
            ExposeWT2D(gpuIdx,
                       volumeT,
                       volumeW,
                       _kernelRL,
                       nf,
                       _maxRadius,
                       _pf,
                       _T2D.nRowFT(),
                       MAX_N_ITER_BALANCE,
                       MIN_N_ITER_BALANCE,
                       _N);
        }
        else if (_mode == MODE_3D)
        {
            int tabSize = 1e5;
            int streamNum = 3;
            RFLOAT *diff;
            RFLOAT *cmax;
            int *counter;
            
            Complex *dev_C;
            RFLOAT *dev_W;
            RFLOAT *dev_T;
            RFLOAT *dev_tab;
            void *stream[streamNum];
            RFLOAT *devDiff;
            RFLOAT *devMax;
            int *devCount;
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
            diff = new RFLOAT[_C3D.nSlcFT()];
            counter = new int[_C3D.nSlcFT()];
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
            cmax = new RFLOAT[_C3D.nSlcFT()];
#endif

            AllocDevicePoint(gpuIdx,
                             &dev_C,
                             &dev_W,
                             &dev_T,
                             &dev_tab,
                             &devDiff,
                             &devMax,
                             &devCount,
                             stream,
                             streamNum,
                             tabSize,
                             _C3D.nSlcFT());
            
            HostDeviceInit(gpuIdx,
                           _C3D,
                           volumeW,
                           volumeT,
                           _kernelRL.getData(),
                           dev_W,
                           dev_T,
                           dev_tab,
                           stream,
                           streamNum,
                           tabSize,
                           _maxRadius,
                           _pf,
                           _C3D.nSlcFT());
            
            RFLOAT diffC = TS_MAX_RFLOAT_VALUE;
            RFLOAT diffCPrev = TS_MAX_RFLOAT_VALUE;

            int m = 0;
            int nDiffCNoDecrease = 0;

            for (m = 0; m < MAX_N_ITER_BALANCE; m++)
            {
                ExposeC(gpuIdx,
                        _C3D,
                        dev_C,
                        dev_T,
                        dev_W,
                        stream,
                        streamNum,
                        _C3D.nSlcFT());
                
                _fft.bwExecutePlanMT(_C3D);
                
                ExposeForConvC(gpuIdx,
                               _C3D,
                               dev_C,
                               dev_tab,
                               stream,
                               _kernelRL,
                               nf,
                               streamNum,
                               tabSize,
                               _pf,
                               _N);
                
                _fft.fwExecutePlanMT(_C3D);
                
                diffCPrev = diffC;
                
                ExposeWC(gpuIdx,
                         _C3D,
                         dev_C,
                         diff,
                         cmax,
                         dev_W,
                         devDiff,
                         devMax,
                         devCount,
                         counter,
                         stream,
                         diffC,
                         streamNum,
                         _maxRadius,
                         _pf);
 
                if (diffC > diffCPrev * DIFF_C_DECREASE_THRES)
                    nDiffCNoDecrease += 1;
                else
                    nDiffCNoDecrease = 0;

                if ((diffC < DIFF_C_THRES) ||
                    ((m >= MIN_N_ITER_BALANCE) &&
                    (nDiffCNoDecrease == N_DIFF_C_NO_DECREASE))) break;
            }
            
            FreeDevHostPoint(gpuIdx,
                             &dev_C,
                             &dev_W,
                             &dev_T,
                             &dev_tab,
                             &devDiff,
                             &devMax,
                             &devCount,
                             stream,
                             _C3D,
                             volumeW,
                             volumeT,
                             streamNum,
                             _C3D.nSlcFT());
            
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE        
            delete[] diff;
            delete[] counter;
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
            delete[] cmax;
#endif
        }
        //else if (_mode == MODE_3D)
        //{
        //    ExposeWT(gpuIdx,
        //             volumeT,
        //             volumeW,
        //             _kernelRL,
        //             _maxRadius,
        //             _pf,
        //             _T3D.nSlcFT(),
        //             nf,
        //             MAX_N_ITER_BALANCE,
        //             MIN_N_ITER_BALANCE,
        //             _N);
        //}
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }
    else
    {
        if (_mode == MODE_2D)
        {
            ExposeWT2D(gpuIdx,
                       volumeT,
                       volumeW,
                       _maxRadius,
                       _pf,
                       _T2D.nRowFT());
        }
        else if (_mode == MODE_3D)
        {
            ExposeWT(gpuIdx,
                     volumeT,
                     volumeW,
                     _maxRadius,
                     _pf,
                     _T3D.nSlcFT());
        }
        else
        {
            REPORT_ERROR("INEXISTENT MODE");

            abort();
        }
    }

    free(volumeT);
    
    if (_mode == MODE_2D)
    {
#ifdef VERBOSE_LEVEL_2
        ALOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Image";
        BLOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Image";
#endif

        Image padDst(_N * _pf, _N * _pf, FT_SPACE);
        Image padDstR(_N * _pf, _N * _pf, RL_SPACE);

#ifdef VERBOSE_LEVEL_2
        ALOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";
#endif

        ExposePF2D(gpuIdx,
                   padDst,
                   padDstR,
                   _F2D,
                   volumeW,
                   _maxRadius,
                   _pf);

        padDst.clearFT();

        Image img;
        IMG_EXTRACT_RL(img, padDstR, 1.0 / _pf);
        
        Volume dstT(_N, _N, 1, RL_SPACE);
        
        IMAGE_FOR_EACH_PIXEL_RL(img)
            dstT.setRL(img.getRL(i, j), i, j, 0);  
        img.clearRL();

        Image imgDst(_N, _N, RL_SPACE);
        SLC_EXTRACT_RL(imgDst, dstT, 0);
        dstT.clearRL();
        
        RFLOAT nf = 0;
#ifdef RECONSTRUCTOR_MKB_KERNEL
        nf = MKB_RL(0, _a * _pf, _alpha);
#endif

        dst.alloc(_N, _N, 1, FT_SPACE);

        int dim = imgDst.nRowRL();
        RFLOAT *mkbRL = new RFLOAT[(dim / 2 + 1) * (dim / 2 + 1)];
        int padSize = _pf * _N;
    
        for (int j = 0; j <= dim / 2; j++) 
            for (int i = 0; i <= dim / 2; i++) 
            {
                size_t index = j * (dim / 2 + 1) + i;
        
#ifdef RECONSTRUCTOR_MKB_KERNEL
                mkbRL[index] = MKB_RL(NORM(i, j) / padSize,
                                  _a * _pf,
                                  _alpha);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                mkbRL[index] = TIK_RL(NORM(i, j) / padSize);
#endif
            }
    
        ExposeCorrF2D(gpuIdx,
                      imgDst,
                      dst,
                      mkbRL,
                      nf);
        
        delete[] mkbRL;
        imgDst.clearRL();
    }
    else if (_mode == MODE_3D)
    {
#ifdef VERBOSE_LEVEL_2
        ALOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Setting Up Padded Destination Volume";

        ALOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";
        BLOG(INFO, "LOGGER_RECO") << "Placing F into Padded Destination Volume";

#endif

        Volume padDst(_N * _pf, _N * _pf, _N * _pf, FT_SPACE);
        //Volume padDstR(_N * _pf, _N * _pf, _N * _pf, RL_SPACE);

        ExposePFW(gpuIdx,
                  padDst,
                  _F3D,
                  volumeW,
                  _maxRadius,
                  _pf);
        
        FFT fft;
        fft.bw(padDst);
        
        //ExposePF(gpuIdx,
        //         padDst,
        //         padDstR,
        //         _F3D,
        //         volumeW,
        //         _maxRadius,
        //         _pf);
        
        //padDst.clearFT();

        //dst.alloc(AROUND((1.0 / _pf) * padDstR.nColRL()), 
        //          AROUND((1.0 / _pf) * padDstR.nRowRL()), 
        //          AROUND((1.0 / _pf) * padDstR.nSlcRL()), 
        //          FT_SPACE);

        //Volume dstN;
        //dstN.alloc(AROUND((1.0 / _pf) * padDstR.nColRL()), 
        //           AROUND((1.0 / _pf) * padDstR.nRowRL()), 
        //           AROUND((1.0 / _pf) * padDstR.nSlcRL()), 
        //           RL_SPACE);
        //
        //VOLUME_FOR_EACH_PIXEL_RL(dstN) 
        //    dstN.setRL(padDstR.getRL(i, j, k), i, j, k);

        //padDstR.clearRL();
        
        dst.alloc(AROUND((1.0 / _pf) * padDst.nColRL()), 
                  AROUND((1.0 / _pf) * padDst.nRowRL()), 
                  AROUND((1.0 / _pf) * padDst.nSlcRL()), 
                  RL_SPACE);

        VOLUME_FOR_EACH_PIXEL_RL(dst) 
            dst.setRL(padDst.getRL(i, j, k), i, j, k);
        
        padDst.clearRL();

        RFLOAT nf = 0;
#ifdef RECONSTRUCTOR_MKB_KERNEL
        nf = MKB_RL(0, _a * _pf, _alpha);
#endif

        int padSize = _pf * _N;
        //int dim = dstN.nSlcRL();
        int dim = dst.nSlcRL();
        int slcSize = (dim / 2 + 1) * (dim / 2 + 1);
        RFLOAT *mkbRL = new RFLOAT[slcSize * (dim / 2 + 1)];
        
        for (int k = 0; k <= dim / 2; k++) 
            for (int j = 0; j <= dim / 2; j++) 
                for (int i = 0; i <= dim / 2; i++) 
                {
                    size_t index = k * slcSize + j * (dim / 2 + 1) + i;
        
#ifdef RECONSTRUCTOR_MKB_KERNEL
                    mkbRL[index] = MKB_RL(NORM_3(i, j, k) / padSize,
                                      _a * _pf,
                                      _alpha);
#endif

#ifdef RECONSTRUCTOR_TRILINEAR_KERNEL
                    mkbRL[index] = TIK_RL(NORM_3(i, j, k) / padSize);
#endif
                }
    
        ExposeCorrF(gpuIdx,
                    dst,
                    mkbRL,
                    nf);
       
#ifdef RECONSTRUCTOR_REMOVE_NEG
        REMOVE_NEG(dst);
#endif
        fft.fw(dst);
        
        //ExposeCorrF(gpuIdx,
        //            dstN,
        //            dst,
        //            mkbRL,
        //            nf);
        
        delete[] mkbRL;
        //dstN.clearRL();
        dst.clearRL();
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    free(volumeW);

#ifdef VERBOSE_LEVEL_2
    ALOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";
    BLOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";
#endif
}

#endif // GPU_RECONSTRUCT

void Reconstructor::allReduceF()
{

    ALOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere A";
    BLOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere B";

    MPI_Barrier(_hemi);

    if (_mode == MODE_2D)
        MPI_Allreduce_Large(&_F2D[0],
                            _F2D.sizeFT(),
                            TS_MPI_DOUBLE_COMPLEX,
                            MPI_SUM,
                            _hemi);
    else if (_mode == MODE_3D)
        MPI_Allreduce_Large(&_F3D[0],
                            _F3D.sizeFT(),
                            TS_MPI_DOUBLE_COMPLEX,
                            MPI_SUM,
                            _hemi);
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    MPI_Barrier(_hemi);
}

void Reconstructor::allReduceT()
{
    ALOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere A";
    BLOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere B";

    MPI_Barrier(_hemi);

    if (_mode == MODE_2D)
        MPI_Allreduce_Large(&_T2D[0],
                            _T2D.sizeFT(),
                            TS_MPI_DOUBLE_COMPLEX,
                            MPI_SUM,
                            _hemi);
    else if (_mode == MODE_3D)
        MPI_Allreduce_Large(&_T3D[0],
                            _T3D.sizeFT(),
                            TS_MPI_DOUBLE_COMPLEX,
                            MPI_SUM,
                            _hemi);
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    MPI_Barrier(_hemi);

#ifdef RECONSTRUCTOR_NORMALISE_T_F
    ALOG(INFO, "LOGGER_RECO") << "Normalising T and F";
    BLOG(INFO, "LOGGER_RECO") << "Normalising T and F";

    if (_mode == MODE_2D)
    {
        RFLOAT sf = 1.0 / REAL(_T2D[0]);

        #pragma omp parallel for
        SCALE_FT(_T2D, sf);
        #pragma omp parallel for
        SCALE_FT(_F2D, sf);
    }
    else if (_mode == MODE_3D)
    {
        RFLOAT sf = 1.0 / REAL(_T3D[0]);

        #pragma omp parallel for
        SCALE_FT(_T3D, sf);
        #pragma omp parallel for
        SCALE_FT(_F3D, sf);
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
#endif
}

void Reconstructor::allReduceO()
{
    ALOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere A";
    BLOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere B";

    MPI_Barrier(_slav);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_ox,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _slav);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_oy,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _slav);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_oz,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _slav);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_counter,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  _slav);
}

RFLOAT Reconstructor::checkC() const
{
#ifdef RECONSTRUCTOR_CHECK_C_AVERAGE
    RFLOAT diff = 0;

    int counter = 0;

    if (_mode == MODE_2D)
    {
        #pragma omp parallel for schedule(dynamic)
        IMAGE_FOR_EACH_PIXEL_FT(_C2D)
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
            {
                #pragma omp atomic
                diff += fabs(ABS(_C2D.getFT(i, j)) - 1);
                #pragma omp atomic
                counter += 1;
            }
    }
    else if (_mode == MODE_3D)
    {
        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_FT(_C3D)
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
            {
                #pragma omp atomic
                diff += fabs(ABS(_C3D.getFT(i, j, k)) - 1);
                #pragma omp atomic
                counter += 1;
            }
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }

    return diff / counter;
#endif

#ifdef RECONSTRUCTOR_CHECK_C_MAX
    if (_mode == MODE_2D)
    {
        vector<RFLOAT> diff(_C2D.sizeFT(), 0);
        
        #pragma omp parallel for schedule(dynamic)
        IMAGE_FOR_EACH_PIXEL_FT(_C2D)
            if (QUAD(i, j) < TSGSL_pow_2(_maxRadius * _pf))
                diff[_C2D.iFTHalf(i, j)] = fabs(ABS(_C2D.getFTHalf(i, j)) - 1);

        return *std::max_element(diff.begin(), diff.end());
    }
    else if (_mode == MODE_3D)
    {
        vector<RFLOAT> diff(_C3D.sizeFT(), 0);

        #pragma omp parallel for schedule(dynamic)
        VOLUME_FOR_EACH_PIXEL_FT(_C3D)
            if (QUAD_3(i, j, k) < TSGSL_pow_2(_maxRadius * _pf))
                diff[_C3D.iFTHalf(i, j, k)] = fabs(ABS(_C3D.getFTHalf(i, j, k)) - 1);

        return *std::max_element(diff.begin(), diff.end());
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
#endif
}

void Reconstructor::convoluteC()
{
#ifdef RECONSTRUCTOR_KERNEL_PADDING
    RFLOAT nf = MKB_RL(0, _a * _pf, _alpha);
#else
    RFLOAT nf = MKB_RL(0, _a, _alpha);
#endif

    if (_mode == MODE_2D)
    {
        _fft.bwExecutePlanMT(_C2D);

        #pragma omp parallel for
        IMAGE_FOR_EACH_PIXEL_RL(_C2D)
            _C2D.setRL(_C2D.getRL(i, j)
                     * _kernelRL(QUAD(i, j) / TSGSL_pow_2(_N * _pf))
                     / nf,
                       i,
                       j);

        _fft.fwExecutePlanMT(_C2D);

        _C2D.clearRL();
    }
    else if (_mode == MODE_3D)
    {
        _fft.bwExecutePlanMT(_C3D);

        #pragma omp parallel for
        VOLUME_FOR_EACH_PIXEL_RL(_C3D)
            _C3D.setRL(_C3D.getRL(i, j, k)
                     * _kernelRL(QUAD_3(i, j, k) / TSGSL_pow_2(_N * _pf))
                     / nf,
                       i,
                       j,
                       k);

        _fft.fwExecutePlanMT(_C3D);

        _C3D.clearRL();
    }
    else
    {
        REPORT_ERROR("INEXISTENT MODE");

        abort();
    }
}

void Reconstructor::symmetrizeF()
{
    if (_sym != NULL)
        SYMMETRIZE_FT(_F3D, _F3D, *_sym, _maxRadius * _pf + 1, LINEAR_INTERP);
    else
        CLOG(WARNING, "LOGGER_SYS") << "Symmetry Information Not Assigned in Reconstructor";
}

void Reconstructor::symmetrizeT()
{
    if (_sym != NULL)
        SYMMETRIZE_FT(_T3D, _T3D, *_sym, _maxRadius * _pf + 1, LINEAR_INTERP);
    else
        CLOG(WARNING, "LOGGER_SYS") << "Symmetry Information Not Assigned in Reconstructor";
}

void Reconstructor::symmetrizeO()
{
    if (_sym != NULL)
    {
        dmat33 L, R;

        dvec3 result = dvec3(_ox, _oy, _oz);

        for (int i = 0; i < _sym->nSymmetryElement(); i++)
        {
            _sym->get(L, R, i);

            result += R * dvec3(_ox, _oy, _oz);
        }

        _counter *= (1 + _sym->nSymmetryElement());

        _ox = result(0);
        _oy = result(1);
        _oz = result(2);
    }
    else
        CLOG(WARNING, "LOGGER_SYS") << "Symmetry Information Not Assigned in Reconstructor";
}
