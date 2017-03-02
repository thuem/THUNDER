/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/ 
#include "MLModel.h"

MLModel::~MLModel()
{
    clear();
}

void MLModel::init(const int mode,
                   const bool refine,
                   const bool ctfRefine,
                   const int k,
                   const int size,
                   const int r,
                   const int pf,
                   const double pixelSize,
                   const double a,
                   const double alpha,
                   const Symmetry* sym)
{
    _mode = mode;
    _refine = refine;
    _ctfRefine = ctfRefine;

    _k = k;
    _size = size;
    _r = r;
    _pf = pf;
    _pixelSize = pixelSize;
    _a = a;
    _alpha = alpha;
    _sym = sym;

    _FSC = mat::Constant(1, _k, 1);
    _tau = mat::Constant(1, _k, DBL_MAX);
}

int MLModel::mode() const
{
    return _mode;
}

void MLModel::setMode(const int mode)
{
    _mode = mode;
}

bool MLModel::refine() const
{
    return _refine;
}

void MLModel::setRefine(const bool refine)
{
    _refine = refine;
}

void MLModel::initProjReco()
{
    ALOG(INFO, "LOGGER_INIT") << "Appending Projectors and Reconstructors";
    BLOG(INFO, "LOGGER_INIT") << "Appending Projectors and Reconstructors";

    FOR_EACH_CLASS
    {
        ALOG(INFO, "LOGGER_INIT") << "Appending Projector of Reference " << l;
        BLOG(INFO, "LOGGER_INIT") << "Appending Projector of Reference " << l;

        _proj.push_back(Projector());

        ALOG(INFO, "LOGGER_INIT") << "Appending Reconstructor of Reference " << l;
        BLOG(INFO, "LOGGER_INIT") << "Appending Reconstructor of Reference " << l;

        _reco.push_back(boost::movelib::unique_ptr<Reconstructor>(new Reconstructor()));
    }

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Projectors and Reconstructors Appended";
    BLOG(INFO, "LOGGER_INIT") << "Projectors and Reconstructors Appended";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Setting Up MPI Environment of Reconstructors";
    BLOG(INFO, "LOGGER_INIT") << "Setting Up MPI Environment of Reconstructors";

    FOR_EACH_CLASS
        _reco[l]->setMPIEnv(_commSize, _commRank, _hemi);

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "MPI Environment of Reconstructors Set Up";
    BLOG(INFO, "LOGGER_INIT") << "MPI Environment of Reconstructors Set Up";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Refreshing Projectors";
    BLOG(INFO, "LOGGER_INIT") << "Refreshing Projectors";

    refreshProj();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Projectors Refreshed";
    BLOG(INFO, "LOGGER_INIT") << "Projectors Refreshed";
#endif

    ALOG(INFO, "LOGGER_INIT") << "Refreshing Reconstructors";
    BLOG(INFO, "LOGGER_INIT") << "Refreshing Reconstructors";

    refreshReco();

#ifdef VERBOSE_LEVEL_1
    MPI_Barrier(_hemi);

    ALOG(INFO, "LOGGER_INIT") << "Reconstructors Refreshed";
    BLOG(INFO, "LOGGER_INIT") << "Reconstructors Refreshed";
#endif
}

Volume& MLModel::ref(const int i)
{
    return _ref[i];
}

void MLModel::appendRef(Volume ref)
{
    _ref.push_back(boost::move(ref));
}

int MLModel::k() const
{
    return _k;
}

int MLModel::size() const
{
    return _size;
}

int MLModel::maxR() const
{
    return _size / 2 - _a;
}

int MLModel::r() const
{
    return _r;
}

void MLModel::setR(const int r)
{
    _rPrev = _r;

    _r = r;

    updateRU();
}

int MLModel::rU() const
{
    return _rU;
}

void MLModel::setRU(const int rU)
{
    _rUPrev = _rU;

    _rU = rU;
}

int MLModel::rPrev() const
{
    return _rPrev;
}

int MLModel::rUPrev() const
{
    return _rUPrev;
}

int MLModel::rT() const
{
    return _rT;
}

void MLModel::setRT(const int rT)
{
    _rT = rT;
}

int MLModel::res() const
{
    return _res;
}

void MLModel::setRes(const int res)
{
    _res = res;
}

int MLModel::resT() const
{
    return _resT;
}

void MLModel::setResT(const int resT)
{
    _resT = resT;
}

int MLModel::rGlobal() const
{
    return _rGlobal;
}

void MLModel::setRGlobal(const int rGlobal)
{
    _rGlobal = rGlobal;
}

Projector& MLModel::proj(const int i)
{
    return _proj[i];
}

Reconstructor& MLModel::reco(const int i)
{
    return *_reco[i];
}

void MLModel::BcastFSC()
{
    MLOG(INFO, "LOGGER_COMPARE") << "Setting Size of _FSC";

    _FSC.resize(_rU, _k);

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_COMPARE") << "Gathering References from Hemisphere A and Hemisphere B";

    FOR_EACH_CLASS
    {
        IF_MASTER
        {
            Volume A, B;

            if (_mode == MODE_2D)
            {
                MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                              << _size 
                                              << " X "
                                              << _size;

                A.alloc(_size, _size, 1, FT_SPACE);
                B.alloc(_size, _size, 1, FT_SPACE);
            }
            else if (_mode == MODE_3D)
            {
                MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                             << _size
                                             << " X "
                                             << _size
                                             << " X "
                                             << _size;

                A.alloc(_size, _size, _size, FT_SPACE);
                B.alloc(_size, _size, _size, FT_SPACE);
            }
            else
                REPORT_ERROR("INEXISTENT MODE");

#ifdef VERBOSE_LEVEL_1
            MLOG(INFO, "LOGGER_COMPARE") << "Size of Hemisphere A of Reference "
                                         << l
                                         << " to be Received: "
                                         << A.sizeFT();
            MLOG(INFO, "LOGGER_COMPARE") << "Size of Hemisphere A of Reference "
                                         << l
                                         << " to be Received: "
                                         << B.sizeFT();
#endif

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from Hemisphere A";

            MPI_Recv_Large(&A[0],
                           A.sizeFT(),
                           MPI_DOUBLE_COMPLEX,
                           HEMI_A_LEAD,
                           l,
                           MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from Hemisphere B";

            MPI_Recv_Large(&B[0],
                           B.sizeFT(),
                           MPI_DOUBLE_COMPLEX,
                           HEMI_B_LEAD,
                           l,
                           MPI_COMM_WORLD);

#ifdef VERBOSE_LEVEL_1
                MLOG(INFO, "LOGGER_COMPARE") << "Zero, REAL = "
                                             << REAL(A[0])
                                             << ", IMAG = "
                                             << IMAG(A[0]);
                MLOG(INFO, "LOGGER_COMPARE") << "Zero, REAL = "
                                             << REAL(B[0])
                                             << ", IMAG = "
                                             << IMAG(B[0]);
#endif

            vec fsc(_rU);

            if (_mode == MODE_2D)
            {
                MLOG(INFO, "LOGGER_COMPARE") << "Calculating FRC of Reference " << l;

                FRC(fsc, A, B, 0);

                _FSC.col(l) = fsc;
            }
            else if (_mode == MODE_3D)
            {
                MLOG(INFO, "LOGGER_COMPARE") << "Calculating FSC of Reference " << l;

                /***
                FFT fft;

                fft.bwMT(A);
                fft.bwMT(B);

                Volume tmpA(_size, _size, _size, RL_SPACE);
                Volume tmpB(_size, _size, _size, RL_SPACE);

                VOL_EXTRACT_RL(tmpA, A, 1.0 / _pf);
                VOL_EXTRACT_RL(tmpB, B, 1.0 / _pf);

                fft.fwMT(tmpA);
                fft.fwMT(tmpB);

                FSC(fsc, tmpA, tmpB);
                ***/

                FSC(fsc, A, B);

                /***
                for (int i = 0; i < _rU * _pf; i++)
                    _FSC(i, l) = fsc(i / _pf);

                fft.fwMT(A);
                fft.fwMT(B);
                A.clearRL();
                B.clearRL();
                ***/
                _FSC.col(l) = fsc;
            }
            else
                REPORT_ERROR("INEXISTENT MODE");

            MLOG(INFO, "LOGGER_COMPARE") << "Averaging A and B";

            /***
            double r = GSL_MIN_DBL(resA2P(1.0 / A_B_AVERAGE_THRES,
                                          _size,
                                          _pixelSize) * _pf,
                                   (_r - 1) * _pf);
                                   ***/

            //double r = (_r - 1) * _pf;

            //double r = _rU * _pf;

            /***
            double r = GSL_MIN_DBL((resA2P(1.0 / A_B_AVERAGE_THRES,
                                           _size,
                                           _pixelSize) + 1) * _pf,
                                   //_rU * _pf);

            MLOG(INFO, "LOGGER_COMPARE") << "Averaging A and B Belower Resolution "
                                         << 1.0 / resP2A(r, _size * _pf, _pixelSize)
                                         << "(Angstrom)";

            #pragma omp parallel for schedule(dynamic)
            VOLUME_FOR_EACH_PIXEL_FT(A)
                if (QUAD_3(i, j, k) < gsl_pow_2(r))
                {
                    Complex avg = (A.getFT(i, j, k) + B.getFT(i, j, k)) / 2;
                    A.setFT(avg, i, j, k);
                    B.setFT(avg, i, j, k);
                }
            ***/

            /***
            #pragma omp parallel for
            FOR_EACH_PIXEL_FT(A)
            {
                Complex avg = (A[i] + B[i]) / 2;
                A[i] = avg;
                B[i] = avg;
            }
            ***/

            /***
            int r = GSL_MIN_INT(AROUND(resA2P(1.0 / A_B_AVERAGE_THRES,
                                             _size,
                                             _pixelSize)),
                                _r);

            if (_mode == MODE_2D)
            {
                //TODO
            }
            else if (_mode == MODE_3D)
            {
                #pragma omp parallel for
                VOLUME_FOR_EACH_PIXEL_FT(A)
                    if (QUAD_3(i, j, k) < gsl_pow_2(r))
                    {
                        Complex avg = (A.getFTHalf(i, j, k)
                                     + B.getFTHalf(i, j, k))
                                    / 2;
                        A.setFTHalf(avg, i, j, k);
                        B.setFTHalf(avg, i, j, k);
                    }
            }
            ***/

            /***
            vec tau(_rU * _pf);
            Image tmp(_size * _pf, _size * _pf, FT_SPACE);
            SLC_EXTRACT_FT(tmp, A, 0);
            powerSpectrum(tau, tmp, _rU * _pf);
            _tau.col(l) = tau;
            ***/

            MLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                         << l
                                         << " to Hemisphere A";
            
            MPI_Ssend_Large(&A[0],
                            A.sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            HEMI_A_LEAD,
                            l,
                            MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Reference "
                                         << l
                                         << " Sent to Hemisphere A";

            MLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                         << l
                                         << " to Hemisphere B";

            MPI_Ssend_Large(&B[0],
                            B.sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            HEMI_B_LEAD,
                            l,
                            MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Reference "
                                         << l
                                         << " Sent to Hemisphere B";
        }
        else
        {
            if ((_commRank == HEMI_A_LEAD) ||
                (_commRank == HEMI_B_LEAD))
            {
                ALOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                             << l
                                             << " from Hemisphere A";

                BLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                             << l
                                             << " from Hemisphere B";

#ifdef VERBOSE_LEVEL_1
                ALOG(INFO, "LOGGER_COMPARE") << "Size of Reference "
                                             << l
                                             << " to be Sent: "
                                             << _ref[l].sizeFT();
                BLOG(INFO, "LOGGER_COMPARE") << "Size of Reference "
                                             << l
                                             << " to be Sent: "
                                             << _ref[l].sizeFT();

                ALOG(INFO, "LOGGER_COMPARE") << "Zero, REAL = "
                                             << REAL(_ref[l][0])
                                             << ", IMAG = "
                                             << IMAG(_ref[l][0]);
                BLOG(INFO, "LOGGER_COMPARE") << "Zero, REAL = "
                                             << REAL(_ref[l][0])
                                             << ", IMAG = "
                                             << IMAG(_ref[l][0]);
#endif

                MPI_Ssend_Large(&_ref[l][0],
                                _ref[l].sizeFT(),
                                MPI_DOUBLE_COMPLEX,
                                MASTER_ID,
                                l,
                                MPI_COMM_WORLD);

                ALOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from MASTER";
                BLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from MASTER";

                MPI_Recv_Large(&_ref[l][0],
                               _ref[l].sizeFT(),
                               MPI_DOUBLE_COMPLEX,
                               MASTER_ID,
                               l,
                               MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (isA())
        {
            ALOG(INFO, "LOGGER_COMPARE") << "Broadcasting Reference " << l << " from A_LEAD";
            MPI_Bcast_Large(&_ref[l][0],
                            _ref[l].sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            0,
                            _hemi);
        }

        if (isB())
        {
            BLOG(INFO, "LOGGER_COMPARE") << "Broadcasting Reference " << l << " from B_LEAD";
            MPI_Bcast_Large(&_ref[l][0],
                            _ref[l].sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            0,
                            _hemi);
        }

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef VERBOSE_LEVEL_1
        MLOG(INFO, "LOGGER_COMPARE") << "Reference " << l << " Broadcasted from A_LEAD and B_LEAD";
#endif
    }

    MLOG(INFO, "LOGGER_COMPARE") << "Broadcasting FSC from MASTER";

    MPI_Bcast(_FSC.data(),
              _FSC.size(),
              MPI_DOUBLE,
              MASTER_ID,
              MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_COMPARE") << "FSC Broadcasted from MASTER";
}

void MLModel::lowPassRef(const double thres,
                         const double ew)
{
    FOR_EACH_CLASS
    {
        if (_mode == MODE_2D)
        {
            //TODO
        }
        else if (_mode == MODE_3D)
            lowPassFilter(_ref[l], _ref[l], thres, ew);
        else
            REPORT_ERROR("INEXISTENT MODE");
    }
}

mat MLModel::fsc() const
{
    return _FSC;
}

mat MLModel::snr() const
{
    return _SNR;
}

vec MLModel::fsc(const int i) const
{
    return _FSC.col(i);
}

vec MLModel::snr(const int i) const
{
    return _SNR.col(i);
}

void MLModel::refreshSNR()
{
    _SNR.resize(_FSC.rows(), _FSC.cols());

    FOR_EACH_CLASS
        _SNR.col(l) = _FSC.col(l).array() / (1 - _FSC.col(l).array());
}

void MLModel::refreshTau()
{
    // TODO

    /***
    _tau.resize(_rU * _pf - 1, _k);

    FOR_EACH_CLASS
    {
        vec ps(_rU * _pf - 1);
        powerSpectrum(ps, _ref[l], _rU * _pf - 1);
        _tau.col(l) = ps / 2;
    }
    ***/

    /***
    _tau.resize(maxR() * _pf - 1, _k);

    FOR_EACH_CLASS
    {
        vec ps(maxR() * _pf - 1);
        powerSpectrum(ps, _ref[l], maxR() * _pf - 1);
        _tau.col(l) = ps / 2;
    }
    ***/
}

void MLModel::refreshSig(const vec& sig)
{
    _sig = sig;
}

/***
void MLModel::resetTau()
{
    _tau = mat::Constant(_tau.rows(), _tau.rows(), DBL_MAX);
}

void MLModel::resetTau(const vec tau)
{
    FOR_EACH_CLASS
        _tau.col(l) = tau;
}
***/

vec MLModel::tau(const int i) const
{
    return _tau.col(i);
}

int MLModel::resolutionP(const int i,
                         const double thres,
                         const bool inverse) const
{
    return resP(_FSC.col(i), thres, 1, 1, inverse);
}

int MLModel::resolutionP(const double thres,
                         const bool inverse) const
{
    int result = 0;

    FOR_EACH_CLASS
        if (result < resolutionP(l, thres, inverse))
            result = resolutionP(l, thres, inverse);

    return result;
}

double MLModel::resolutionA(const int i,
                            const double thres) const
{
    return resP2A(resolutionP(i, thres), _size, _pixelSize);
}

double MLModel::resolutionA(const double thres) const
{
    return resP2A(resolutionP(thres), _size, _pixelSize);
}

void MLModel::setProjMaxRadius(const int maxRadius)
{
    FOR_EACH_CLASS
        _proj[l].setMaxRadius(maxRadius);
}

void MLModel::refreshProj()
{
    FOR_EACH_CLASS
    {
        if (_mode == MODE_2D)
        {
            _proj[l].setMode(MODE_2D);

            Image tmp(_size, _size, FT_SPACE);
            SLC_EXTRACT_FT(tmp, _ref[l], 0);

            _proj[l].setProjectee(tmp.copyImage());
        }
        else if (_mode == MODE_3D)
        {
            _proj[l].setMode(MODE_3D);

            _proj[l].setProjectee(_ref[l].copyVolume());
        }
        else
            REPORT_ERROR("INEXISTENT MODE");

        _proj[l].setMaxRadius(_r);
        _proj[l].setPf(_pf);

        if (_searchType == SEARCH_TYPE_GLOBAL)
            _proj[l].setInterp(INTERP_TYPE_GLOBAL);
        else
            _proj[l].setInterp(INTERP_TYPE_LOCAL);
    }
}

void MLModel::refreshReco()
{
    ALOG(INFO, "LOGGER_SYS") << "Refreshing Reconstructor(s) with Frequency Upper Boundary : "
                             << _rU;
    BLOG(INFO, "LOGGER_SYS") << "Refreshing Reconstructor(s) with Frequency Upper Boundary : "
                             << _rU;

    FOR_EACH_CLASS
    {
        ALOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Initialising";
        BLOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Initialising";

        _reco[l]->init(_mode,
                       _size,
                       _pf,
                       _sym,
                       _a,
                       _alpha);

        ALOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Setting Up Max Radius";
        BLOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Setting Up Max Radius";

        _reco[l]->setMaxRadius(_rU);
    }
}

void MLModel::resetReco()
{
    ALOG(INFO, "LOGGER_SYS") << "Resetting Reconstructor(s) with Frequency Upper Boundary : "
                             << _rU;
    BLOG(INFO, "LOGGER_SYS") << "Resetting Reconstructor(s) with Frequency Upper Boundary : "
                             << _rU;

#ifdef VERBOSE_LEVEL_3
    ILOG(INFO, "LOGGER_SYS") << "Resetting Reconstructor(s)";
#endif

    FOR_EACH_CLASS
    {
        _reco[l]->reset();

        _reco[l]->setFSC(_FSC.col(l).head(_res));

        _reco[l]->setMaxRadius(_rU);
    }

#ifdef VERBOSE_LEVEL_3
    ILOG(INFO, "LOGGER_SYS") << "Reconstructor(s) Reset";
#endif
}

/***
void MLModel::refreshRecoSigTau(const int rSig,
                                const int rTau)
{
    FOR_EACH_CLASS
    {
        _reco[l]->setSig(_sig.head(rSig));

        //_reco[l]->setTau(_tau.col(l).head(rTau * _pf - 1));
        // the last value of _tau can be inaccurate
        _reco[l]->setTau(_tau.col(l).head((rTau - 1) * _pf));
    }
}
***/

void MLModel::updateR(const double thres)
{
    // record the frequency
    _rPrev = _r;

    bool elevate = false;

    MLOG(INFO, "LOGGER_SYS") << "_r = " << _r;
    MLOG(INFO, "LOGGER_SYS") << "_searchType = " << _searchType;

    if ((_r == _rGlobal) &&
        (_searchType == SEARCH_TYPE_GLOBAL))
    {
        MLOG(INFO, "LOGGER_SYS") << "Using rChangeDecreaseFactor "
                                 << R_CHANGE_DECREASE_STUN;
        elevate = determineIncreaseR(R_CHANGE_DECREASE_STUN);
    }
    else if (_searchType == SEARCH_TYPE_GLOBAL)
    {
        MLOG(INFO, "LOGGER_SYS") << "Using rChangeDecreaseFactor "
                                 << R_CHANGE_DECREASE_GLOBAL;

        elevate = determineIncreaseR(R_CHANGE_DECREASE_GLOBAL);
    }
    else
    {
        MLOG(INFO, "LOGGER_SYS") << "Using rChangeDecreaseFactor "
                                 << R_CHANGE_DECREASE_LOCAL;

        elevate = determineIncreaseR(R_CHANGE_DECREASE_LOCAL);
    }

    if (elevate)
    {
        MLOG(INFO, "LOGGER_SYS") << "Elevating Cutoff Frequency";

        elevateR(thres);
    }
}

void MLModel::elevateR(const double thres)
{
    if (_searchType == SEARCH_TYPE_GLOBAL)
        _r = GSL_MAX_INT(_r,
                         GSL_MIN_INT(resolutionP(thres, false) + 1 + CUTOFF_BEYOND_RES,
                                     _r + AROUND((double)_rGlobal / 3)));
    else
        _r = GSL_MAX_INT(_r,
                         GSL_MIN_INT(resolutionP(thres, false) + 1 + CUTOFF_BEYOND_RES,
                                     _r + AROUND((double)maxR() / 6)));

    if (_searchType == SEARCH_TYPE_GLOBAL)
        _r = GSL_MIN_INT(_rGlobal, _r);
}

double MLModel::rVari() const
{
    return _rVari;
}

double MLModel::tVariS0() const
{
    return _tVariS0;
}

double MLModel::tVariS1() const
{
    return _tVariS1;
}

double MLModel::stdRVari() const
{
    return _stdRVari;
}

double MLModel::stdTVariS0() const
{
    return _stdTVariS0;
}

double MLModel::stdTVariS1() const
{
    return _stdTVariS1;
}

void MLModel::setRVari(const double rVari)
{
    _rVari = rVari;
}

void MLModel::setTVariS0(const double tVariS0)
{
    _tVariS0 = tVariS0;
}

void MLModel::setTVariS1(const double tVariS1)
{
    _tVariS1 = tVariS1;
}

void MLModel::setStdRVari(const double stdRVari)
{
    _stdRVari = stdRVari;
}

void MLModel::setStdTVariS0(const double stdTVariS0)
{
    _stdTVariS0 = stdTVariS0;
}

void MLModel::setStdTVariS1(const double stdTVariS1)
{
    _stdTVariS1 = stdTVariS1;
}

double MLModel::rChange() const
{
    return _rChange;
}

double MLModel::rChangePrev() const
{
    return _rChangePrev;
}

double MLModel::stdRChange() const
{
    return _stdRChange;
}

void MLModel::setRChange(const double rChange)
{
    _rChangePrev = _rChange;

    _rChange = rChange;
}

void MLModel::resetRChange()
{
    _rChangePrev = 1;
    
    _rChange = 1;

    _stdRChange = 0;
}

void MLModel::setStdRChange(const double stdRChange)
{
    _stdRChangePrev = _stdRChange;

    _stdRChange = stdRChange;
}

int MLModel::nRChangeNoDecrease() const
{
    return _nRChangeNoDecrease;
}

void MLModel::setNRChangeNoDecrease(const int nRChangeNoDecrease)
{
    _nRChangeNoDecrease = nRChangeNoDecrease;
}

int MLModel::nTopResNoImprove() const
{
    return _nTopResNoImprove;
}

void MLModel::setNTopResNoImprove(const int nTopResNoImprove)
{
    _nTopResNoImprove = nTopResNoImprove;
}

int MLModel::searchType()
{
    // record search type
    _searchTypePrev = _searchType;

    // If the searching needs to stop, return the stop signal.
    if (_searchType == SEARCH_TYPE_STOP) return SEARCH_TYPE_STOP;

    if ((_searchType == SEARCH_TYPE_LOCAL) ||
        (_searchType == SEARCH_TYPE_CTF))
    {
        // If it is local search, check whether there is no space for
        // improvement or not. If there is, perform further local search, if
        // there is not, stop the search.
        IF_MASTER
        {
            if (_increaseR)
            {
                /***
                if ((_r > _rT) ||
                    (_res > _resT))
                ***/
                if (_res > _resT)
                    _nTopResNoImprove = 0;
                else
                {
                    _nTopResNoImprove += 1;

                    MLOG(INFO, "LOGGER_SYS") << "Top Resolution (Pixel): "
                                             << _resT
                                             << ", Current Resolution (Pixel): "
                                             << _res;

                    MLOG(INFO, "LOGGER_SYS") << "Number of Iterations without Top Resolution Elevating : "
                                             << _nTopResNoImprove;
                }

                if (_nTopResNoImprove >= MAX_ITER_RES_NO_IMPROVE)
                {
                    if ((_searchType == SEARCH_TYPE_LOCAL) &&
                        _ctfRefine)
                    {
                        _searchType = SEARCH_TYPE_CTF;
                        _nTopResNoImprove = 0;

                        resetRChange();
                        setNRChangeNoDecrease(0);
                        setIncreaseR(false);
                    }
                    else
                        _searchType = SEARCH_TYPE_STOP;
                }
            }
        }
    }
    else
    {
        // If it is global search now, make sure the change of rotations
        // beteween iterations still gets room for improvement.
        IF_MASTER
            if ((_r == _rGlobal) && _increaseR)
            {
                if (_refine)
                    _searchType = SEARCH_TYPE_LOCAL;
                else
                    _searchType = SEARCH_TYPE_STOP;
            }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&_searchType,
              1,
              MPI_INT,
              MASTER_ID,
              MPI_COMM_WORLD);

    return _searchType;
}

int MLModel::searchTypePrev() const
{
    return _searchTypePrev;
}

bool MLModel::increaseR() const
{
    return _increaseR;
}

void MLModel::setIncreaseR(const bool increaseR)
{
    _increaseR = increaseR;
}

void MLModel::updateRU()
{
    _rUPrev = _rU;

    _rU = GSL_MIN_INT(_r
                    + ((_searchType == SEARCH_TYPE_GLOBAL)
                     ? AROUND((double)_rGlobal / 3)
                     : AROUND((double)maxR() / 3)),
                      maxR());

    MLOG(INFO, "LOGGER_SYS") << "Resetting Frequency Boundary of Reconstructor to "
                             << _rU;
}

void MLModel::clear()
{
    _ref.clear();

    _proj.clear();
    _reco.clear();
}

bool MLModel::determineIncreaseR(const double rChangeDecreaseFactor)
{
    IF_MASTER
    {
        /***
        if (_rChange > _rChangePrev
                     - rChangeDecreaseFactor
                     * _stdRChangePrev)
                     ***/
        if (_rChange > (1 - rChangeDecreaseFactor) * _rChangePrev)
        {
            // When the frequency remains the same as the last iteration, check
            // whether there is a decrease of rotation change.
            _nRChangeNoDecrease += 1;
        }
        else
            _nRChangeNoDecrease = 0;

        switch (_searchType)
        {
            case SEARCH_TYPE_STOP:
                _increaseR = false;
                break;

            case SEARCH_TYPE_GLOBAL:
                _increaseR = (_nRChangeNoDecrease
                           >= MAX_ITER_R_CHANGE_NO_DECREASE_GLOBAL);
                break;

            case SEARCH_TYPE_LOCAL:
            case SEARCH_TYPE_CTF:
                _increaseR = (_nRChangeNoDecrease
                           >= MAX_ITER_R_CHANGE_NO_DECREASE_LOCAL);
                break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&_increaseR,
              1,
              MPI_C_BOOL,
              MASTER_ID,
              MPI_COMM_WORLD);

    return _increaseR;
}

void MLModel::avgHemi()
{
    FOR_EACH_CLASS
    {
        IF_MASTER
        {
            Volume A, B;
            if (_mode == MODE_2D)
            {
                MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                             << _size
                                             << " X "
                                             << _size;

                A.alloc(_size, _size, 1, FT_SPACE);
                B.alloc(_size, _size, 1, FT_SPACE);
            }
            else if (_mode == MODE_3D)
            {
                MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                             << _size
                                             << " X "
                                             << _size
                                             << " X "
                                             << _size;

                Volume A(_size, _size, _size, FT_SPACE);
                Volume B(_size, _size, _size, FT_SPACE);
            }

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from Hemisphere A";

            MPI_Recv_Large(&A[0],
                           A.sizeFT(),
                           MPI_DOUBLE_COMPLEX,
                           HEMI_A_LEAD,
                           l,
                           MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from Hemisphere B";

            MPI_Recv_Large(&B[0],
                           B.sizeFT(),
                           MPI_DOUBLE_COMPLEX,
                           HEMI_B_LEAD,
                           l,
                           MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Averaging Two Hemispheres";
            FOR_EACH_PIXEL_FT(_ref[l])
                _ref[l][i] = (A[i] + B[i]) / 2;
        }
        else
        {
            if ((_commRank == HEMI_A_LEAD) ||
                (_commRank == HEMI_B_LEAD))
            {
                ALOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                             << l
                                             << " from Hemisphere A";
                BLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                             << l
                                             << " from Hemisphere B";

                MPI_Ssend_Large(&_ref[l][0],
                                _ref[l].sizeFT(),
                                MPI_DOUBLE_COMPLEX,
                                MASTER_ID,
                                l,
                                MPI_COMM_WORLD);
            }
        }
    }
}
