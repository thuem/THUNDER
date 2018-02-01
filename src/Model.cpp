//This header file is add by huabin
#include "huabin.h"
/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/ 
#include "Model.h"

Model::~Model()
{
    clear();
}

void Model::init(const int mode,
                   const bool gSearch,
                   const bool lSearch,
                   const bool cSearch,
                   const bool coreFSC,
                   const int coreR,
                   const bool maskFSC,
                   const Volume* mask,
                   const bool goldenStandard,
                   const int k,
                   const int size,
                   const int r,
                   const int pf,
                   const RFLOAT pixelSize,
                   const RFLOAT a,
                   const RFLOAT alpha,
                   const Symmetry* sym)
{
    _mode = mode;
    _gSearch = gSearch;
    _lSearch = lSearch;
    _cSearch = cSearch;

    _coreFSC = coreFSC;
    _coreR = coreR;

    _maskFSC = maskFSC;
    _mask = mask;

    _goldenStandard = goldenStandard;

    _k = k;
    _size = size;
    _r = r;
    _pf = pf;
    _pixelSize = pixelSize;
    _a = a;
    _alpha = alpha;
    _sym = sym;

    _FSC = mat::Constant(1, _k, 1);
    _tau = mat::Constant(1, _k, TS_MAX_RFLOAT_VALUE);
}

int Model::mode() const
{
    return _mode;
}

void Model::setMode(const int mode)
{
    _mode = mode;
}

bool Model::gSearch() const
{
    return _gSearch;
}

void Model::setGSearch(const bool gSearch)
{
    _gSearch = gSearch;
}

bool Model::lSearch() const
{
    return _lSearch;
}

void Model::setLSearch(const bool lSearch)
{
    _lSearch = lSearch;
}

/***
bool Model::refine() const
{
    return _refine;
}

void Model::setRefine(const bool refine)
{
    _refine = refine;
}
***/

void Model::initProjReco()
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

Volume& Model::ref(const int i)
{
    return _ref[i];
}

void Model::appendRef(Volume ref)
{
    _ref.push_back(boost::move(ref));
}

void Model::clearRef()
{
    _ref.clear();
}

int Model::k() const
{
    return _k;
}

int Model::size() const
{
    return _size;
}

int Model::maxR() const
{
    return _size / 2 - CEIL(_a);
}

int Model::r() const
{
    return _r;
}

void Model::setR(const int r)
{
    _rPrev = _r;

    _r = r;

    updateRU();
}

int Model::rInit() const
{
    return _rInit;
}

void Model::setRInit(const int rInit)
{
    _rInit = rInit;
}

int Model::rU() const
{
    return _rU;
}

void Model::setRU(const int rU)
{
    _rUPrev = _rU;

    _rU = rU;
}

void Model::setMaxRU()
{
    _rU = maxR();
}

int Model::rPrev() const
{
    return _rPrev;
}

void Model::setRPrev(const int rPrev)
{
    _rPrev = rPrev;
}

int Model::rUPrev() const
{
    return _rUPrev;
}

void Model::setRUPrev(const int rUPrev)
{
    _rUPrev = rUPrev;
}

int Model::rT() const
{
    return _rT;
}

void Model::setRT(const int rT)
{
    _rT = rT;
}

int Model::res() const
{
    return _res;
}

void Model::setRes(const int res)
{
    _res = res;
}

int Model::resT() const
{
    return _resT;
}

void Model::setResT(const int resT)
{
    _resT = resT;
}

int Model::rGlobal() const
{
    return _rGlobal;
}

void Model::setRGlobal(const int rGlobal)
{
    _rGlobal = rGlobal;
}

Projector& Model::proj(const int i)
{
    return _proj[i];
}

Reconstructor& Model::reco(const int i)
{
    return *_reco[i];
}

void Model::compareTwoHemispheres(const bool fscFlag,
                                  const bool avgFlag,
                                  const RFLOAT thres)
{
    if (fscFlag)
    {
        MLOG(INFO, "LOGGER_COMPARE") << "Setting Size of _FSC";

        _FSC.resize(_rU, _k);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_COMPARE") << "Gathering References from Hemisphere A and Hemisphere B";

    FOR_EACH_CLASS
    {
        IF_MASTER
        {
            Volume A, B;

            if (_mode == MODE_2D)
            {
#ifdef VERBOSE_LEVEL_2
                MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                              << _size 
                                              << " X "
                                              << _size;
#endif

                A.alloc(_size, _size, 1, FT_SPACE);
                B.alloc(_size, _size, 1, FT_SPACE);
            }
            else if (_mode == MODE_3D)
            {
#ifdef VERBOSE_LEVEL_2
                MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                             << _size
                                             << " X "
                                             << _size
                                             << " X "
                                             << _size;
#endif

                A.alloc(_size, _size, _size, FT_SPACE);
                B.alloc(_size, _size, _size, FT_SPACE);
            }
            else
            {
                REPORT_ERROR("INEXISTENT MODE");

                abort();
            }

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
                           TS_MPI_DOUBLE_COMPLEX,
                           HEMI_A_LEAD,
                           l,
                           MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from Hemisphere B";

            MPI_Recv_Large(&B[0],
                           B.sizeFT(),
                           TS_MPI_DOUBLE_COMPLEX,
                           HEMI_B_LEAD,
                           l,
                           MPI_COMM_WORLD);

#ifdef VERBOSE_LEVEL_2
            MLOG(INFO, "LOGGER_COMPARE") << "Zero, REAL = "
                                         << REAL(A[0])
                                         << ", IMAG = "
                                         << IMAG(A[0]);
            MLOG(INFO, "LOGGER_COMPARE") << "Zero, REAL = "
                                         << REAL(B[0])
                                         << ", IMAG = "
                                         << IMAG(B[0]);
#endif

            if (fscFlag)
            {
                vec fsc(_rU);

                if ((_maskFSC) && (_mode == MODE_3D))
                {
                    FFT fft;

                    MLOG(INFO, "LOGGER_COMPARE") << "Calculating FSC of Mask Region of Reference " << l;

                    MLOG(INFO, "LOGGER_COMPARE") << "Calculating FSC of Unmasked Reference";

                    vec fscUnmask(_rU);

                    FSC(fscUnmask, A, B);

                    MLOG(INFO, "LOGGER_COMPARE") << "Determining Random Phase Resolution";
                    
                    int randomPhaseThres = resP(fscUnmask, 0.8, 1, 1, false);

                    MLOG(INFO, "LOGGER_COMPARE") << "Performing Random Phase Resolution from "
                                                 << 1.0 / resP2A(randomPhaseThres, _size, _pixelSize);

                    Volume randomPhaseA(_size, _size, _size, FT_SPACE);
                    Volume randomPhaseB(_size, _size, _size, FT_SPACE);

                    MLOG(INFO, "LOGGER_COMPARE") << "Performing Random Phase on Unmask Reference";

                    randomPhase(randomPhaseA, A, randomPhaseThres);
                    randomPhase(randomPhaseB, B, randomPhaseThres);

                    fft.bwMT(randomPhaseA);
                    fft.bwMT(randomPhaseB);

                    MLOG(INFO, "LOGGER_COMPARE") << "Performing Mask on Random Phase Reference";

                    softMask(randomPhaseA, randomPhaseA, *_mask, 0);
                    softMask(randomPhaseB, randomPhaseB, *_mask, 0);

                    fft.fwMT(randomPhaseA);
                    fft.fwMT(randomPhaseB);

                    randomPhaseA.clearRL();
                    randomPhaseB.clearRL();
                    
                    vec fscRFMask(_rU);

                    FSC(fscRFMask, randomPhaseA, randomPhaseB);

                    // randomPhaseA.clearFT();
                    // randomPhaseB.clearFT();

                    MLOG(INFO, "LOGGER_COMPARE") << "Masking Reference";

                    fft.bwMT(A);
                    fft.bwMT(B);

                    Volume maskA(_size, _size, _size, RL_SPACE);
                    Volume maskB(_size, _size, _size, RL_SPACE);

                    softMask(maskA, A, *_mask, 0);
                    softMask(maskB, B, *_mask, 0);

                    fft.fwMT(maskA);
                    fft.fwMT(maskB);

                    maskA.clearRL();
                    maskB.clearRL();

                    fft.fwMT(A);
                    fft.fwMT(B);

                    MLOG(INFO, "LOGGER_COMPARE") << "Calculating FSC of Masked Reference ";

                    vec fscMask(_rU);

                    FSC(fscMask, maskA, maskB);

                    MLOG(INFO, "LOGGER_COMPARE") << "Calculating True FSC";

                    for (int i = 0; i < _rU; i++)
                    {
                        if (i < randomPhaseThres + 2)
                            fsc(i) = fscMask(i);
                        else
                            fsc(i) = (fscMask(i) - fscRFMask(i)) / (1 - fscRFMask(i));
                    }
                    
                    _FSC.col(l) = fsc;
                }
                else if (_coreFSC && (_mode == MODE_3D))
                {
                    MLOG(INFO, "LOGGER_COMPARE") << "Calculating FSC of Core Region of Reference " << l;

                    FFT fft;
                    fft.bwMT(A);
                    fft.bwMT(B);

                    MLOG(INFO, "LOGGER_COMPARE") << "Core Region is "
                                                 << (2 * _coreR)
                                                 << " x "
                                                 << (2 * _coreR)
                                                 << " x "
                                                 << (2 * _coreR);

                    RFLOAT ef = (2.0 * _coreR) / _size;

                    MLOG(INFO, "LOGGER_COMPARE") << "Core Region Extract Factor: " << ef;

                    MLOG(INFO, "LOGGER_COMPARE") << "Extracing Core Region from Reference " << l;

                    Volume coreA, coreB;
                    VOL_EXTRACT_RL(coreA, A, ef);
                    VOL_EXTRACT_RL(coreB, B, ef);

                    fft.fwMT(coreA);
                    fft.fwMT(coreB);

                    coreA.clearRL();
                    coreB.clearRL();

                    int coreRU = FLOOR(_rU * ef);

                    MLOG(INFO, "LOGGER_COMPARE") << "Determining Core Region FSC of Reference " << l;

                    vec coreFSC(coreRU);
                    FSC(coreFSC, coreA, coreB);

                    for (int i = 0; i < _rU; i++)
                        fsc(i) = coreFSC(GSL_MIN_INT(AROUND(i * ef), coreRU - 1));

                    fft.fwMT(A);
                    fft.fwMT(B);

                    A.clearRL();
                    B.clearRL();

                    _FSC.col(l) = fsc;
                }
                else
                {
                    if (_mode == MODE_2D)
                    {
                        if (_coreFSC)
                            MLOG(WARNING, "LOGGER_COMPARE") << "2D MODE DOES NOT SUPPORT CORE REGION FSC";

                        if (_maskFSC)
                            MLOG(WARNING, "LOGGER_COMPARE") << "2D MODE DOES NOT SUPPORT MASK REGION FSC";

                        MLOG(INFO, "LOGGER_COMPARE") << "Calculating FRC of Reference " << l;

                        FRC(fsc, A, B, 0);

                        _FSC.col(l) = fsc;
                    }
                    else if (_mode == MODE_3D)
                    {
                        MLOG(INFO, "LOGGER_COMPARE") << "Calculating FSC of Reference " << l;

                        FSC(fsc, A, B);

                        _FSC.col(l) = fsc;
                    }
                    else
                    {
                        REPORT_ERROR("INEXISTENT MODE");

                        abort();
                    }
                }
            }

            if (avgFlag)
            {
            
                MLOG(INFO, "LOGGER_COMPARE") << "Averaging A and B";

                if ((_k == 1) && (_goldenStandard))
                {
                    // When refining only one reference, use gold standard FSC.

#ifdef MODEL_AVERAGE_TWO_HEMISPHERE
                    #pragma omp parallel for
                    FOR_EACH_PIXEL_FT(A)
                    {
                        Complex avg = (A[i] + B[i]) / 2;
                        A[i] = avg;
                        B[i] = avg;
                    }
#else
#ifdef MODEL_RESOLUTION_BASE_AVERAGE
                    int r = resolutionP(thres, false);
#else
                    int r = GSL_MIN_INT(AROUND(resA2P(1.0 / A_B_AVERAGE_THRES,
                                                      _size,
                                                      _pixelSize)),
                                        _r);
#endif

                    MLOG(INFO, "LOGGER_COMPARE") << "Averaging A and B Below Resolution "
                                                 << 1.0 / resP2A(r, _size, _pixelSize)
                                                 << "(Angstrom)";

                    if (_mode == MODE_2D)
                    {
                        MLOG(FATAL, "LOGGER_COMPARE") << "2D MODE DOES NOT SUPPORT GOLDEN STANDARD AVERAGING";

                        abort();
                    }
                    else if (_mode == MODE_3D)
                    {
                        #pragma omp parallel for
                        VOLUME_FOR_EACH_PIXEL_FT(A)
                            if (QUAD_3(i, j, k) < TSGSL_pow_2(r))
                            {
                                Complex avg = (A.getFTHalf(i, j, k)
                                             + B.getFTHalf(i, j, k))
                                             / 2;
                                A.setFTHalf(avg, i, j, k);
                                B.setFTHalf(avg, i, j, k);
                            }
                    }
                    else
                    {
                        REPORT_ERROR("INEXISTENT MODE");

                        abort();
                    }
#endif
                }
                else
                {
                    // When refining more than 1 references, directly average two half maps.

                    #pragma omp parallel for
                    FOR_EACH_PIXEL_FT(A)
                    {
                        Complex avg = (A[i] + B[i]) / 2;
                        A[i] = avg;
                        B[i] = avg;
                    }
                }

                MLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                             << l
                                             << " to Hemisphere A";
            
#ifdef MODEL_SWAP_HEMISPHERE
                MPI_Ssend_Large(&B[0],
                                A.sizeFT(),
                                TS_MPI_DOUBLE_COMPLEX,
                                HEMI_A_LEAD,
                                l,
                                MPI_COMM_WORLD);
#else
                MPI_Ssend_Large(&A[0],
                                A.sizeFT(),
                                TS_MPI_DOUBLE_COMPLEX,
                                HEMI_A_LEAD,
                                l,
                                MPI_COMM_WORLD);
#endif

                MLOG(INFO, "LOGGER_COMPARE") << "Reference "
                                             << l
                                             << " Sent to Hemisphere A";

                MLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                             << l
                                             << " to Hemisphere B";

#ifdef MODEL_SWAP_HEMISPHERE
                MPI_Ssend_Large(&A[0],
                                B.sizeFT(),
                                TS_MPI_DOUBLE_COMPLEX,
                                HEMI_B_LEAD,
                                l,
                                MPI_COMM_WORLD);
#else
                MPI_Ssend_Large(&B[0],
                                B.sizeFT(),
                                TS_MPI_DOUBLE_COMPLEX,
                                HEMI_B_LEAD,
                                l,
                                MPI_COMM_WORLD);
#endif

                MLOG(INFO, "LOGGER_COMPARE") << "Reference "
                                             << l
                                             << " Sent to Hemisphere B";

            }
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

#ifdef VERBOSE_LEVEL_2
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
                                TS_MPI_DOUBLE_COMPLEX,
                                MASTER_ID,
                                l,
                                MPI_COMM_WORLD);

                if (avgFlag)
                {

                    ALOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from MASTER";
                    BLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from MASTER";

                    MPI_Recv_Large(&_ref[l][0],
                                   _ref[l].sizeFT(),
                                   TS_MPI_DOUBLE_COMPLEX,
                                   MASTER_ID,
                                   l,
                                   MPI_COMM_WORLD);

                }
            }
        }

        if (avgFlag)
        {
            MPI_Barrier(MPI_COMM_WORLD);

            if (isA())
            {
                ALOG(INFO, "LOGGER_COMPARE") << "Broadcasting Reference " << l << " from A_LEAD";
                MPI_Bcast_Large(&_ref[l][0],
                                _ref[l].sizeFT(),
                                TS_MPI_DOUBLE_COMPLEX,
                                0,
                                _hemi);
            }

            if (isB())
            {
                BLOG(INFO, "LOGGER_COMPARE") << "Broadcasting Reference " << l << " from B_LEAD";
                MPI_Bcast_Large(&_ref[l][0],
                                _ref[l].sizeFT(),
                                TS_MPI_DOUBLE_COMPLEX,
                                0,
                                _hemi);
            }

            MPI_Barrier(MPI_COMM_WORLD);

#ifdef VERBOSE_LEVEL_1
            MLOG(INFO, "LOGGER_COMPARE") << "Reference " << l << " Broadcasted from A_LEAD and B_LEAD";
#endif
        }
    }

    if (fscFlag)
    {

        MLOG(INFO, "LOGGER_COMPARE") << "Broadcasting FSC from MASTER";

        MPI_Bcast(_FSC.data(),
                  _FSC.size(),
                  TS_MPI_DOUBLE,
                  MASTER_ID,
                  MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_COMPARE") << "FSC Broadcasted from MASTER";
    }
}

void Model::lowPassRef(const RFLOAT thres,
                         const RFLOAT ew)
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

mat Model::fsc() const
{
    return _FSC;
}

mat Model::snr() const
{
    return _SNR;
}

vec Model::fsc(const int i) const
{
    return _FSC.col(i);
}

vec Model::snr(const int i) const
{
    return _SNR.col(i);
}

void Model::refreshSNR()
{
    _SNR.resize(_FSC.rows(), _FSC.cols());

    FOR_EACH_CLASS
        _SNR.col(l) = _FSC.col(l).array() / (1 - _FSC.col(l).array());
}

void Model::refreshTau()
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

void Model::refreshSig(const vec& sig)
{
    _sig = sig;
}

/***
void Model::resetTau()
{
    _tau = mat::Constant(_tau.rows(), _tau.rows(), TS_MAX_RFLOAT_VALUE);
}

void Model::resetTau(const vec tau)
{
    FOR_EACH_CLASS
        _tau.col(l) = tau;
}
***/

vec Model::tau(const int i) const
{
    return _tau.col(i);
}

int Model::resolutionP(const int i,
                       const RFLOAT thres,
                       const bool inverse) const
{
    return resP(_FSC.col(i), thres, 1, 1, inverse);
}

int Model::resolutionP(const RFLOAT thres,
                       const bool inverse) const
{
    int result = 0;

    FOR_EACH_CLASS
        if (result < resolutionP(l, thres, inverse))
            result = resolutionP(l, thres, inverse);

    return result;
}

RFLOAT Model::resolutionA(const int i,
                          const RFLOAT thres) const
{
    return resP2A(resolutionP(i, thres), _size, _pixelSize);
}

RFLOAT Model::resolutionA(const RFLOAT thres) const
{
    return resP2A(resolutionP(thres), _size, _pixelSize);
}

void Model::setProjMaxRadius(const int maxRadius)
{
    FOR_EACH_CLASS
        _proj[l].setMaxRadius(maxRadius);
}

void Model::refreshProj()
{
    FOR_EACH_CLASS
    {
        _proj[l].setPf(_pf);

        if (_searchType == SEARCH_TYPE_GLOBAL)
            _proj[l].setInterp(INTERP_TYPE_GLOBAL);
        else
            _proj[l].setInterp(INTERP_TYPE_LOCAL);

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
    }
}

void Model::refreshReco()
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
                       _size,
                       _pf,
                       _sym,
                       _a,
                       _alpha);

#ifdef MODEL_RECONSTRUCTOR_RESIZE
        ALOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Resizing";
        BLOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Resizing";

        _reco[l]->resizeSpace((_rU + CEIL(_a)) * 2);
#endif

        ALOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Setting Up FSC";
        BLOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Setting Up FSC";

        _reco[l]->setFSC(vec::Constant(_rU, 1));

        ALOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Setting Up Max Radius";
        BLOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Setting Up Max Radius";


        _reco[l]->setMaxRadius(_rU);
    }
}

void Model::resetReco(const RFLOAT thres)
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
#ifdef MODEL_RECONSTRUCTOR_RESIZE
        _reco[l]->resizeSpace(GSL_MIN_INT(_size, (_rU + CEIL(_a)) * 2));
#else
        _reco[l]->reset();
#endif
    
        ALOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Setting Up FSC";
        BLOG(INFO, "LOGGER_SYS") << "Reconstructor of Class "
                                 << l
                                 << " Setting Up FSC";

        _reco[l]->setFSC(_FSC.col(l));

        _reco[l]->setMaxRadius(_rU);
    }

#ifdef VERBOSE_LEVEL_3
    ILOG(INFO, "LOGGER_SYS") << "Reconstructor(s) Reset";
#endif
}

/***
void Model::refreshRecoSigTau(const int rSig,
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

void Model::updateR(const RFLOAT thres)
{
    // record the frequency
    _rPrev = _r;

    bool elevate = false;

    MLOG(INFO, "LOGGER_SYS") << "_r = " << _r;
    MLOG(INFO, "LOGGER_SYS") << "_searchType = " << _searchType;

    if ((_r == _rGlobal) &&
        (_searchType == SEARCH_TYPE_GLOBAL))
    {

#ifdef MODEL_DETERMINE_INCREASE_R_R_CHANGE
        MLOG(INFO, "LOGGER_SYS") << "Using rChangeDecreaseFactor "
                                 << R_CHANGE_DECREASE_STUN;
        elevate = determineIncreaseR(R_CHANGE_DECREASE_STUN);
#endif

#ifdef MODEL_DETERMINE_INCREASE_R_T_VARI
        MLOG(INFO, "LOGGER_SYS") << "Using rVariDecreaseFactor "
                                 << T_VARI_DECREASE_STUN;
        elevate = determineIncreaseR(T_VARI_DECREASE_STUN);
#endif

#ifdef MODEL_DETERMINE_INCREASE_FSC
        MLOG(INFO, "LOGGER_SYS") << "Using fscIncreaseFactor "
                                 << FSC_INCREASE_STUN;
        elevate = determineIncreaseR(FSC_INCREASE_STUN);
#endif

    }
    else if (_searchType == SEARCH_TYPE_GLOBAL)
    {

#ifdef MODEL_DETERMINE_INCREASE_R_R_CHANGE
        MLOG(INFO, "LOGGER_SYS") << "Using rChangeDecreaseFactor "
                                 << R_CHANGE_DECREASE_GLOBAL;
        elevate = determineIncreaseR(R_CHANGE_DECREASE_GLOBAL);
#endif

#ifdef MODEL_DETERMINE_INCREASE_R_T_VARI
        MLOG(INFO, "LOGGER_SYS") << "Using rVariDecreaseFactor "
                                 << T_VARI_DECREASE_GLOBAL;
        elevate = determineIncreaseR(T_VARI_DECREASE_GLOBAL);
#endif

#ifdef MODEL_DETERMINE_INCREASE_FSC
        MLOG(INFO, "LOGGER_SYS") << "Using fscIncreaseFactor "
                                 << FSC_INCREASE_GLOBAL;
        elevate = determineIncreaseR(FSC_INCREASE_GLOBAL);
#endif

    }
    else
    {
#ifdef MODEL_DETERMINE_INCREASE_R_R_CHANGE
        MLOG(INFO, "LOGGER_SYS") << "Using rChangeDecreaseFactor "
                                 << R_CHANGE_DECREASE_LOCAL;
        elevate = determineIncreaseR(R_CHANGE_DECREASE_LOCAL);
#endif

#ifdef MODEL_DETERMINE_INCREASE_R_T_VARI
        MLOG(INFO, "LOGGER_SYS") << "Using rVariDecreaseFactor "
                                 << T_VARI_DECREASE_LOCAL;
        elevate = determineIncreaseR(T_VARI_DECREASE_LOCAL);
#endif
        
#ifdef MODEL_DETERMINE_INCREASE_FSC
        MLOG(INFO, "LOGGER_SYS") << "Using fscIncreaseFactor "
                                 << FSC_INCREASE_LOCAL;
        elevate = determineIncreaseR(FSC_INCREASE_LOCAL);
#endif
    }

    if (elevate)
    {
        MLOG(INFO, "LOGGER_SYS") << "Elevating Cutoff Frequency";

        elevateR(thres);
    }
}

void Model::elevateR(const RFLOAT thres)
{
    if (_searchType == SEARCH_TYPE_GLOBAL)
        _r = GSL_MAX_INT(_r,
                         GSL_MIN_INT(resolutionP(thres, false) + 1 + CUTOFF_BEYOND_RES,
                                     _r + CEIL((RFLOAT)(_rGlobal - _rInit) / 2)));
    else
        _r = GSL_MAX_INT(_r,
                         GSL_MIN_INT(resolutionP(thres, false) + 1 + CUTOFF_BEYOND_RES,
                                     _r + CEIL(TSGSL_MAX_RFLOAT(_r * M_SQRT2, (RFLOAT)(maxR() - _rGlobal) / 4))));

    if (_searchType == SEARCH_TYPE_GLOBAL)
        _r = GSL_MIN_INT(_rGlobal, _r);
}

RFLOAT Model::rVari() const
{
    return _rVari;
}

RFLOAT Model::tVariS0() const
{
    return _tVariS0;
}

RFLOAT Model::tVariS1() const
{
    return _tVariS1;
}

RFLOAT Model::stdRVari() const
{
    return _stdRVari;
}

RFLOAT Model::stdTVariS0() const
{
    return _stdTVariS0;
}

RFLOAT Model::stdTVariS1() const
{
    return _stdTVariS1;
}

RFLOAT Model::tVariS0Prev() const
{
    return _tVariS0Prev;
}

RFLOAT Model::tVariS1Prev() const
{
    return _tVariS1Prev;
}

void Model::setRVari(const RFLOAT rVari)
{
    _rVari = rVari;
}

void Model::setTVariS0(const RFLOAT tVariS0)
{
    _tVariS0Prev = _tVariS0;

    _tVariS0 = tVariS0;
}

void Model::setTVariS1(const RFLOAT tVariS1)
{
    _tVariS1Prev = _tVariS1;

    _tVariS1 = tVariS1;
}

void Model::resetTVari()
{
    _tVariS0Prev = TS_MAX_RFLOAT_VALUE;
    _tVariS1Prev = TS_MAX_RFLOAT_VALUE;

    _tVariS0 = TS_MAX_RFLOAT_VALUE;
    _tVariS1 = TS_MAX_RFLOAT_VALUE;
}

void Model::setStdRVari(const RFLOAT stdRVari)
{
    _stdRVari = stdRVari;
}

void Model::setStdTVariS0(const RFLOAT stdTVariS0)
{
    _stdTVariS0 = stdTVariS0;
}

void Model::setStdTVariS1(const RFLOAT stdTVariS1)
{
    _stdTVariS1 = stdTVariS1;
}

RFLOAT Model::fscArea() const
{
    return _fscArea;
}

RFLOAT Model::fscAreaPrev() const
{
    return _fscAreaPrev;
}

void Model::setFSCArea(const RFLOAT fscArea)
{
    _fscAreaPrev = _fscArea;

    _fscArea = fscArea;
}

void Model::resetFSCArea()
{
    _fscAreaPrev = 0;
    _fscArea = 0;
}

RFLOAT Model::rChange() const
{
    return _rChange;
}

RFLOAT Model::rChangePrev() const
{
    return _rChangePrev;
}

RFLOAT Model::stdRChange() const
{
    return _stdRChange;
}

void Model::setRChange(const RFLOAT rChange)
{
    _rChangePrev = _rChange;

    _rChange = rChange;
}

void Model::resetRChange()
{
    _rChangePrev = 1;
    
    _rChange = 1;

    _stdRChange = 0;
}

void Model::setStdRChange(const RFLOAT stdRChange)
{
    _stdRChangePrev = _stdRChange;

    _stdRChange = stdRChange;
}

int Model::nRChangeNoDecrease() const
{
    return _nRChangeNoDecrease;
}

void Model::setNRChangeNoDecrease(const int nRChangeNoDecrease)
{
    _nRChangeNoDecrease = nRChangeNoDecrease;
}

int Model::nTopResNoImprove() const
{
    return _nTopResNoImprove;
}

void Model::setNTopResNoImprove(const int nTopResNoImprove)
{
    _nTopResNoImprove = nTopResNoImprove;
}

int Model::searchType()
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
                        _cSearch)
                    {
                        _searchType = SEARCH_TYPE_CTF;
                        _nTopResNoImprove = 0;

                        resetTVari();
                        resetRChange();
                        resetFSCArea();
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
                _searchType = _lSearch
                            ? SEARCH_TYPE_LOCAL
                            : SEARCH_TYPE_STOP;
                /***
                if (_refine)
                    _searchType = SEARCH_TYPE_LOCAL;
                else
                    _searchType = SEARCH_TYPE_STOP;
                ***/
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

void Model::setSearchType(const int searchType)
{
    _searchType = searchType;
}

int Model::searchTypePrev() const
{
    return _searchTypePrev;
}

void Model::setSearchTypePrev(const int searchTypePrev)
{
    _searchTypePrev = _searchTypePrev;
}

bool Model::increaseR() const
{
    return _increaseR;
}

void Model::setIncreaseR(const bool increaseR)
{
    _increaseR = increaseR;
}

void Model::updateRU()
{
    _rUPrev = _rU;

#ifdef MODEL_ALWAYS_MAX_RU

    _rU = maxR();

#else

#ifdef MODEL_ALWAYS_MAX_RU_EXCEPT_GLOBAL

    _rU = (_searchType == SEARCH_TYPE_GLOBAL)
        ? GSL_MIN_INT(_r + AROUND((RFLOAT)maxR() / 3), maxR())
        : maxR();
#else

    _rU = GSL_MIN_INT(_r + AROUND((RFLOAT)maxR() / 3), maxR());

#endif

#endif

    MLOG(INFO, "LOGGER_SYS") << "Resetting Frequency Boundary of Reconstructor to "
                             << _rU;
}

void Model::clear()
{
    _ref.clear();

    _proj.clear();
    _reco.clear();
}

#ifdef MODEL_DETERMINE_INCREASE_R_R_CHANGE

bool Model::determineIncreaseR(const RFLOAT rChangeDecreaseFactor)
{
    IF_MASTER
    {
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

#endif

#ifdef MODEL_DETERMINE_INCREASE_R_T_VARI

bool Model::determineIncreaseR(const RFLOAT tVariDecreaseFactor)
{
    IF_MASTER
    {
        if ((_tVariS0 > (1 - tVariDecreaseFactor) * _tVariS0Prev) &&
            (_tVariS1 > (1 - tVariDecreaseFactor) * _tVariS1Prev))
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
                _increaseR = (_nRChangeNoDecrease
                           >= MAX_ITER_R_CHANGE_NO_DECREASE_LOCAL);
                break;

            case SEARCH_TYPE_CTF:
                _increaseR = (_nRChangeNoDecrease
                           >= MAX_ITER_R_CHANGE_NO_DECREASE_CTF);
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

#endif

#ifdef MODEL_DETERMINE_INCREASE_FSC

bool Model::determineIncreaseR(const RFLOAT fscIncreaseFactor)
{
    IF_MASTER
    {
        if (_fscArea < (1 + fscIncreaseFactor) * _fscAreaPrev)
        {
            // When the frequency remains the same as the last iteration, check
            // whether there is a decrease of rotation change.
            _nRChangeNoDecrease += 1;

            MLOG(INFO, "LOGGER_SYS") << "_fscArea = " << _fscArea;
            MLOG(INFO, "LOGGER_SYS") << "_fscAreaPrev = " << _fscAreaPrev;
            MLOG(INFO, "LOGGER_SYS") << "fscIncreaseFactor = " << fscIncreaseFactor;
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
                _increaseR = (_nRChangeNoDecrease
                           >= MAX_ITER_R_CHANGE_NO_DECREASE_LOCAL);
                break;

            case SEARCH_TYPE_CTF:
                _increaseR = (_nRChangeNoDecrease
                           >= MAX_ITER_R_CHANGE_NO_DECREASE_CTF);
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

#endif

void Model::avgHemi()
{
    FOR_EACH_CLASS
    {
        IF_MASTER
        {
            Volume A, B;

            if (_mode == MODE_2D)
            {

#ifdef VERBOSE_LEVEL_2
                MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                             << _size
                                             << " X "
                                             << _size;
#endif

                A.alloc(_size, _size, 1, FT_SPACE);
                B.alloc(_size, _size, 1, FT_SPACE);
            }
            else if (_mode == MODE_3D)
            {

#ifdef VERBOSE_LEVEL_2
                MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                             << _size
                                             << " X "
                                             << _size
                                             << " X "
                                             << _size;
#endif

                Volume A(_size, _size, _size, FT_SPACE);
                Volume B(_size, _size, _size, FT_SPACE);
            }

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from Hemisphere A";

            MPI_Recv_Large(&A[0],
                           A.sizeFT(),
                           TS_MPI_DOUBLE_COMPLEX,
                           HEMI_A_LEAD,
                           l,
                           MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << l << " from Hemisphere B";

            MPI_Recv_Large(&B[0],
                           B.sizeFT(),
                           TS_MPI_DOUBLE_COMPLEX,
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
                                TS_MPI_DOUBLE_COMPLEX,
                                MASTER_ID,
                                l,
                                MPI_COMM_WORLD);
            }
        }
    }
}
