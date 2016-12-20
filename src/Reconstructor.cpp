/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Reconstructor.h"

Reconstructor::Reconstructor() {}

Reconstructor::Reconstructor(const int size,
                             const int pf,
                             const Symmetry* sym,
                             const double a,
                             const double alpha)
{
    init(size, pf, sym, a, alpha);
}

Reconstructor::~Reconstructor() {}

void Reconstructor::init(const int size,
                         const int pf,
                         const Symmetry* sym,
                         const double a,
                         const double alpha)
{
    _size = size;
    _pf = pf;
    _sym = sym;

    /***
    _FSC = vec::Constant(1, 1);

    _sig = vec::Zero(1);

    _tau = vec::Constant(1, 1);
    ***/

    _a = a;
    _alpha = alpha;

    // initialise the interpolation kernel
    _kernel.init(bind(MKB_FT_R2, _1, _pf * _a, _alpha),
                 0,
                 gsl_pow_2(_pf * _a),
                 1e5);

    _maxRadius = (_size / 2 - a);

    _F.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
    _W.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
    _C.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
    _T.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);

    reset();
}

void Reconstructor::reset()
{
    _rot.clear();
    _w.clear();
    _ctf.clear();

    _iCol = NULL;
    _iRow = NULL;
    _iPxl = NULL;
    _iSig = NULL;

    _calMode = POST_CAL_MODE;

    #pragma omp parallel for
    SET_0_FT(_F);

    #pragma omp parallel for
    SET_1_FT(_W);

    #pragma omp parallel for
    SET_0_FT(_C);

    #pragma omp parallel for
    SET_0_FT(_T);
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

void Reconstructor::insert(const Image& src,
                           const Image& ctf,
                           const mat33& rot,
                           const vec2& t,
                           const double w)
{
    IF_MASTER
    {
        CLOG(FATAL, "LOGGER_SYS") << "Inserting Images into Reconstructor in MASTER";
    }

    if (_calMode != POST_CAL_MODE)
    {
        CLOG(FATAL, "LOGGER_SYS") << "Wrong Pre(Post) Calculation Mode in Reconstructor";
    }

    if ((src.nColRL() != _size) ||
        (src.nRowRL() != _size) ||
        (ctf.nColRL() != _size) ||
        (ctf.nRowRL() != _size))
        CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of Inserting Image"
                                  << ": _size = " << _size
                                  << ", nCol = " << src.nColRL()
                                  << ", nRow = " << src.nRowRL();

    Image transSrc(_size, _size, FT_SPACE);
    translateMT(transSrc, src, _maxRadius, -t(0), -t(1));

    vector<mat33> sr;
    symmetryRotation(sr, rot, _sym);

    for (int k = 0; k < int(sr.size()); k++)
    {
        _rot.push_back(sr[k]);
        _w.push_back(w);
        _ctf.push_back(&ctf);

        #pragma omp parallel for schedule(dynamic)
        IMAGE_FOR_EACH_PIXEL_FT(transSrc)
        {
            if (QUAD(i, j) < gsl_pow_2(_maxRadius))
            {
                vec3 newCor = {(double)(i * _pf), (double)(j * _pf), 0};
                vec3 oldCor = sr[k] * newCor;

#ifdef MKB_KERNEL
                _F.addFT(transSrc.getFTHalf(i, j)
                       * REAL(ctf.getFTHalf(i, j))
                       * w, 
                         oldCor(0), 
                         oldCor(1), 
                         oldCor(2), 
                         _pf * _a, 
                         _kernel);
#endif

#ifdef TRILINEAR_KERNEL
                _F.addFT(transSrc.getFTHalf(i, j)
                       * REAL(ctf.getFTHalf(i, j))
                       * w, 
                         oldCor(0), 
                         oldCor(1), 
                         oldCor(2));
#endif

#ifdef ADD_T_DURING_INSERT

#ifdef MKB_KERNEL
                _T.addFT(gsl_pow_2(REAL(ctf.getFTHalf(i, j)))
                       * w, 
                         oldCor(0), 
                         oldCor(1), 
                         oldCor(2),
                         _pf * _a,
                         _kernel);
#endif

#ifdef TRILINEAR_KERNEL
                _T.addFT(gsl_pow_2(REAL(ctf.getFTHalf(i, j)))
                       * w, 
                         oldCor(0), 
                         oldCor(1), 
                         oldCor(2));
#endif

#endif
            }
        }
    }
}

void Reconstructor::insertP(const Image& src,
                            const Image& ctf,
                            const mat33& rot,
                            const vec2& t,
                            const double w)
{
    IF_MASTER
    {
        CLOG(FATAL, "LOGGER_SYS") << "Inserting Images into Reconstructor in MASTER";
    }

    if (_calMode != PRE_CAL_MODE)
    {
        CLOG(FATAL, "LOGGER_SYS") << "Wrong Pre(Post) Calculation Mode in Reconstructor";
    }

    Image transSrc(_size, _size, FT_SPACE);
    translateMT(transSrc, src, -t(0), -t(1), _iCol, _iRow, _iPxl, _nPxl);

    vector<mat33> sr;
    symmetryRotation(sr, rot, _sym);

    for (int k = 0; k < int(sr.size()); k++)
    {
        _rot.push_back(sr[k]);
        _w.push_back(w);
        _ctf.push_back(&ctf);

        #pragma omp parallel for
        for (int i = 0; i < _nPxl; i++)
        {
            vec3 newCor = {(double)(_iCol[i] * _pf), (double)(_iRow[i] * _pf), 0};
            vec3 oldCor = sr[k] * newCor;

#ifdef MKB_KERNEL
            _F.addFT(transSrc[_iPxl[i]]
                   * REAL(ctf.iGetFT(_iPxl[i]))
                   * w,
                     oldCor(0), 
                     oldCor(1), 
                     oldCor(2), 
                     _pf * _a, 
                     _kernel);
#endif

#ifdef TRILINEAR_KERNEL
            _F.addFT(transSrc[_iPxl[i]]
                   * REAL(ctf.iGetFT(_iPxl[i]))
                   * w,
                     oldCor(0), 
                     oldCor(1), 
                     oldCor(2));
#endif

#ifdef ADD_T_DURING_INSERT

#ifdef MKB_KERNEL
            _T.addFT(gsl_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                   * w,
                     oldCor(0), 
                     oldCor(1), 
                     oldCor(2),
                     _pf * _a,
                     _kernel);
#endif

#ifdef TRILINEAR_KERNEL
            _T.addFT(gsl_pow_2(REAL(ctf.iGetFT(_iPxl[i])))
                   * w,
                     oldCor(0), 
                     oldCor(1), 
                     oldCor(2));
#endif

#endif
        }
    }
}

void Reconstructor::insert(const Image& src,
                           const Image& ctf,
                           const Coordinate5D coord,
                           const double w)
{
    mat33 rot;
    rotate3D(rot, coord.phi, coord.theta, coord.psi);

    vec2 t = {(double)coord.x, (double)coord.y};

    insert(src, ctf, rot, t, w);
}

void Reconstructor::reconstruct(Volume& dst)
{
    IF_MASTER return;

    ALOG(INFO, "LOGGER_RECO") << "Allreducing T";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing T";

    allReduceT();

    /***
    #pragma omp parallel for
    FOR_EACH_PIXEL_FT(_T)
        _T[i] += COMPLEX(1, 0);
    ***/

    vec avg = vec::Zero(_maxRadius * _pf + 1);
    shellAverage(avg,
                 _T,
                 [](const Complex x){ return REAL(x); },
                 _maxRadius * _pf - 1);
    // the last two elements have low fidelity
    avg(_maxRadius * _pf - 1) = avg(_maxRadius * _pf - 2);
    avg(_maxRadius * _pf) = avg(_maxRadius * _pf - 2);

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

    ALOG(INFO, "LOGGER_SYS") << "End of FSC = " << _FSC(_FSC.size() - 1);
    BLOG(INFO, "LOGGER_SYS") << "End of FSC = " << _FSC(_FSC.size() - 1);

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(_T)
        if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
        {
            int u = AROUND(NORM_3(i, j, k));

            /***
            double FSC = (u >= _FSC.size())
                       ? _FSC(_FSC.size() - 1)
                       : _FSC(u);
                       ***/

            double FSC = GSL_MAX_DBL((u >= _FSC.size())
                                   ? _FSC(_FSC.size() - 1)
                                   : _FSC(u),
                                     0.001);

            /***
            double FSC = GSL_MAX_DBL((u >= _FSC.size())
                                   ? _FSC(_FSC.size() - 1)
                                   : _FSC(u),
                                     0.01);
                                     ***/

            /***
            _T[i] += COMPLEX((1 - FSC) / FSC
                           * MKB_BLOB_VOL(_pf * _a, _alpha)
                           / gsl_pow_3(_pf),
                             0);
                             ***/

            _T[i] += COMPLEX((1 - FSC) / FSC
                           * avg(u),
                           /// gsl_pow_3(_pf),
                             0);
        }

    ALOG(INFO, "LOGGER_RECO") << "Initialising W";
    BLOG(INFO, "LOGGER_RECO") << "Initialising W";

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_FT(_W)
        if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
            _W.setFTHalf(COMPLEX(1, 0), i, j, k);
        else
            _W.setFTHalf(COMPLEX(0, 0), i, j, k);

    double diffC = DBL_MAX;
    double diffCPrev = DBL_MAX;

    for (int m = 0; ; m++)
    {
        ALOG(INFO, "LOGGER_RECO") << "Balancing Weights Round " << m;
        BLOG(INFO, "LOGGER_RECO") << "Balancing Weights Round " << m;

        /***
        ALOG(INFO, "LOGGER_RECO") << "Allreducing W";
        BLOG(INFO, "LOGGER_RECO") << "Allreducing W";

        allReduceW();
        ***/

        ALOG(INFO, "LOGGER_RECO") << "Determining C";
        BLOG(INFO, "LOGGER_RECO") << "Determining C";
        
        #pragma omp parallel for
        FOR_EACH_PIXEL_FT(_C)
            _C[i] = _T[i] * _W[i];

        convoluteC();

        ALOG(INFO, "LOGGER_RECO") << "Calculating Distance to Total Balanced";
        BLOG(INFO, "LOGGER_RECO") << "Calculating Distance to Total Balanced";
        
        diffCPrev = diffC;

        diffC = checkC();

        ALOG(INFO, "LOGGER_RECO") << "Distance to Total Balanced: " << diffC;
        BLOG(INFO, "LOGGER_RECO") << "Distance to Total Balanced: " << diffC;

        if ((m >= N_ITER_BALANCE) &&
            (diffC > diffCPrev * DIFF_C_DECREASE_THRES)) break;
        else
        {
            #pragma omp parallel for schedule(dynamic)
            VOLUME_FOR_EACH_PIXEL_FT(_W)
                if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
                    _W.setFTHalf(_W.getFTHalf(i, j, k)
                               / GSL_MAX_DBL(ABS(_C.getFTHalf(i, j, k)),
                                             1e-6),
                                 i,
                                 j,
                                 k);
        }
    }

    ALOG(INFO, "LOGGER_RECO") << "Allreducing F";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing F";

    allReduceF();

    ALOG(INFO, "LOGGER_RECO") << "Copying F";
    BLOG(INFO, "LOGGER_RECO") << "Copying F";

    dst = _F.copyVolume();

    MPI_Barrier(_hemi);

    /***
    ALOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming F";
    BLOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming F";
    ***/

    ILOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming F";

    FFT fft;
    fft.bwMT(dst);

    ILOG(INFO, "LOGGER_RECO") << "Inverse Fourier Transforming F Accomplished";

    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_RECO") << "Correcting Convolution Kernel";
    BLOG(INFO, "LOGGER_RECO") << "Correcting Convolution Kernel";

#ifdef MKB_KERNEL
    double nf = MKB_RL(0, _a * _pf, _alpha);
#endif

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_RL(dst)
    {
        double r = NORM_3(i, j, k) / PAD_SIZE;

        /***
        if (r >= 0.5 / _pf * RECO_LOOSE_FACTOR)
            dst.setRL(0, i, j, k);
        ***/

#ifdef MKB_KERNEL
        if (r < 0.5 / _pf * RECO_LOOSE_FACTOR)
            dst.setRL(dst.getRL(i, j, k)
                    / MKB_RL(r, _a * _pf, _alpha)
                    * nf,
                      i,
                      j,
                      k);
        else
            dst.setRL(0, i, j, k);
#endif

#ifdef TRILINEAR_KERNEL
        if (r < 0.5 / _pf * RECO_LOOSE_FACTOR)
            dst.setRL(dst.getRL(i, j, k)
                    / TIK_RL(r),
                      i,
                      j,
                      k);
        else
            dst.setRL(0, i, j, k);
#endif
    }

    ALOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";
    BLOG(INFO, "LOGGER_RECO") << "Convolution Kernel Corrected";

    /***
    CLOG(INFO, "LOGGER_SYS") << "sum(dst)" << dst.sizeRL()
                                            * gsl_stats_mean(&dst(0), 1, dst.sizeRL());
    ***/
}

void Reconstructor::allReduceW()
{
    #pragma omp parallel for
    SET_0_FT(_C);

    ALOG(INFO, "LOGGER_RECO") << "Re-calculating C";
    BLOG(INFO, "LOGGER_RECO") << "Re-calculating C";

    if (_calMode == POST_CAL_MODE)
    {
        #pragma omp parallel for
        for (int k = 0; k < int(_rot.size()); k++)
            for (int j = -_size / 2; j < _size / 2; j++)
                for (int i = 0; i <= _size / 2; i++)
                {
                    if (QUAD(i, j) < gsl_pow_2(_maxRadius))
                    {
                        vec3 newCor = {(double)(i * _pf), (double)(j * _pf), 0};
                        vec3 oldCor = _rot[k] * newCor;

#ifdef MKB_KERNEL
                        _C.addFT(REAL(_W.getByInterpolationFT(oldCor[0],
                                                              oldCor[1],
                                                              oldCor[2],
                                                              LINEAR_INTERP))
                               * gsl_pow_2(REAL(_ctf[k]->getFTHalf(i, j)))
                               * _w[k],
                                 oldCor[0],
                                 oldCor[1],
                                 oldCor[2],
                                 _pf * _a,
                                 _kernel);
#endif

#ifdef TRILINEAR_KERNEL
                        _C.addFT(REAL(_W.getByInterpolationFT(oldCor[0],
                                                              oldCor[1],
                                                              oldCor[2],
                                                              LINEAR_INTERP))
                               * gsl_pow_2(REAL(_ctf[k]->getFTHalf(i, j)))
                               * _w[k],
                                 oldCor[0],
                                 oldCor[1],
                                 oldCor[2]);
#endif
                    }
                }
    }
    else if (_calMode == PRE_CAL_MODE)
    {
        #pragma omp parallel for
        for (int k = 0; k < int(_rot.size()); k++)
            for (int i = 0; i < _nPxl; i++)
            {
                vec3 newCor = {(double)(_iCol[i] * _pf), (double)(_iRow[i] * _pf), 0};
                vec3 oldCor = _rot[k] * newCor;

#ifdef MKB_KERNEL
                _C.addFT(REAL(_W.getByInterpolationFT(oldCor[0],
                                                      oldCor[1],
                                                      oldCor[2],
                                                      LINEAR_INTERP))
                       * gsl_pow_2(REAL(_ctf[k]->iGetFT(_iPxl[i])))
                       * _w[k],
                         oldCor[0],
                         oldCor[1],
                         oldCor[2],
                         _pf * _a,
                         _kernel);
#endif

#ifdef TRILINEAR_KERNEL
                _C.addFT(REAL(_W.getByInterpolationFT(oldCor[0],
                                                      oldCor[1],
                                                      oldCor[2],
                                                      LINEAR_INTERP))
                       * gsl_pow_2(REAL(_ctf[k]->iGetFT(_iPxl[i])))
                       * _w[k],
                         oldCor[0],
                         oldCor[1],
                         oldCor[2]);
#endif
            }
    }
    else
    {
        CLOG(FATAL, "LOGGER_SYS") << "Invalid Pre(Post) Calculation Mode in Reconstructor";
    }
    
    ALOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere A";
    BLOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere B";

    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_RECO") << "Allreducing C";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing C";

    MPI_Allreduce_Large(&_C[0],
                        _C.sizeFT(),
                        MPI_DOUBLE_COMPLEX,
                        MPI_SUM,
                        _hemi);

    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_RECO") << "Correcting Convolution Correction of C";
    BLOG(INFO, "LOGGER_RECO") << "Correcting Convolution Correction of C";

    /***
    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(_C)
        if (QUAD_3(i, j, k) >= gsl_pow_2(_maxRadius * _pf))
            _C.setFTHalf(COMPLEX(0, 0), i, j, k);
    ***/

    FFT fft;
    fft.bwMT(_C);

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(_C)
    {
        _C.setRL(_C.getRL(i, j, k)
               / TIK_RL(NORM_3(i, j, k) / PAD_SIZE),
                 i,
                 j,
                 k);
    }

    fft.fwMT(_C);
    _C.clearRL();

    ALOG(INFO, "LOGGER_RECO") << "Adding Wiener Factor to C";
    BLOG(INFO, "LOGGER_RECO") << "Adding Wiener Factor to C";

    /***
    #pragma omp parallel for
    SET_0_FT(_T);

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(_T)
        if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
        {
            if ((i % _pf == 0) &&
                (j % _pf == 0) &&
                (k % _pf == 0))
                _T.addFT(_W.getFTHalf(i, j, k),
                         i,
                         j,
                         k,
                         _pf * _a,
                         _kernel);
        }
    ***/

    _T = _W.copyVolume();

    fft.bwMT(_T);

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(_T)
    {
        _T.setRL(_T.getRL(i, j, k)
               * MKB_RL(NORM_3(i, j, k) / PAD_SIZE,
                        _a * _pf,
                        _alpha),
                 i,
                 j,
                 k);
    }

    fft.fwMT(_T);
    _T.clearRL();

    #pragma omp parallel for
    FOR_EACH_PIXEL_FT(_C)
        _C[i] += _T[i];

    /***
    double blobVol = MKB_BLOB_VOL(_a * _pf, _alpha);

    #pragma omp parallel for
    FOR_EACH_PIXEL_FT(_C)
        _C[i] += blobVol * _W[i];
    ***/

    /***
    #pragma omp parallel for
    FOR_EACH_PIXEL_FT(_C)
        _C[i] += _W[i] * _T[i];
        ***/

    ALOG(INFO, "LOGGER_RECO") << "Re-calculating W";
    BLOG(INFO, "LOGGER_RECO") << "Re-calculating W";

    /***
    #pragma omp parallel for
    FOR_EACH_PIXEL_FT(_W)
        _W[i] /= REAL(_C[i]);
    ***/

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(_W)
        if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
        {
            _W.setFTHalf(_W.getFTHalf(i, j, k)
                       / REAL(_C.getFTHalf(i, j, k)),
                         i,
                         j,
                         k);
        }

            /***
            if (REAL(_C.getFTHalf(i, j, k)) > 1)
            {
                _W.setFTHalf(_W.getFTHalf(i, j, k)
                           / REAL(_C.getFTHalf(i, j, k)),
                             i,
                             j,
                             k);
            }
            ***/
    /***
        }
        else
            _W.setFTHalf(COMPLEX(0, 0), i, j, k);
            ***/
}

void Reconstructor::allReduceF()
{
    MUL_FT(_F, _W);

    ALOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere A";
    BLOG(INFO, "LOGGER_RECO") << "Waiting for Synchronizing all Processes in Hemisphere B";

    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_RECO") << "Allreducing F";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing F";

    MPI_Allreduce_Large(&_F[0],
                        _F.sizeFT(),
                        MPI_DOUBLE_COMPLEX,
                        MPI_SUM,
                        _hemi);

    MPI_Barrier(_hemi);
}

void Reconstructor::allReduceT()
{
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_RECO") << "Allreducing T";
    BLOG(INFO, "LOGGER_RECO") << "Allreducing T";

    MPI_Allreduce_Large(&_T[0],
                        _T.sizeFT(),
                        MPI_DOUBLE_COMPLEX,
                        MPI_SUM,
                        _hemi);

    MPI_Barrier(_hemi);
}

double Reconstructor::checkC() const
{
    double diff = 0;

#ifdef CHECK_C_AVERAGE
    int counter = 0;

    #pragma omp parallel for schedule(dynamic)
    VOLUME_FOR_EACH_PIXEL_FT(_C)
        if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
        {
            #pragma omp critical
            diff += abs(ABS(_C.getFT(i, j, k)) - 1);
            #pragma omp critical
            counter += 1;
        }

    return diff / counter;
#endif

#ifdef CHECK_C_MAX
    #pragma omp parallel for schedule(dynamic) reduction(max:diff)
    VOLUME_FOR_EACH_PIXEL_FT(_C)
        if (QUAD_3(i, j, k) < gsl_pow_2(_maxRadius * _pf))
            diff = GSL_MAX_DBL(diff, abs(ABS(_C.getFT(i, j, k)) - 1));

    return diff;
#endif
}

void Reconstructor::convoluteC()
{
    FFT fft;

    fft.bwMT(_C);
    _C.clearFT();

    double nf = MKB_RL(0, _a * _pf, _alpha);

    #pragma omp parallel for
    VOLUME_FOR_EACH_PIXEL_RL(_C)
    {
        double r = NORM_3(i, j, k) / PAD_SIZE;

        _C.setRL(_C.getRL(i, j, k)
               * MKB_RL(r, _a * _pf, _alpha)
               / nf,
                 i,
                 j,
                 k);
    }

    fft.fwMT(_C);
    _C.clearRL();
}
