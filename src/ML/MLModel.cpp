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

MLModel::MLModel() {}

MLModel::~MLModel()
{
    clear();
}

void MLModel::init(const int k,
                   const int size,
                   const int r,
                   const int pf,
                   const double pixelSize,
                   const double a,
                   const double alpha,
                   const Symmetry* sym)
{
    _k = k;
    _size = size;
    _r = r;
    _pf = pf;
    _pixelSize = pixelSize;
    _a = a;
    _alpha = alpha;
    _sym = sym;
}

void MLModel::initProjReco()
{
    ALOG(INFO, "LOGGER_INIT") << "Appending Projectors and Reconstructors";
    BLOG(INFO, "LOGGER_INIT") << "Appending Projectors and Reconstructors";

    FOR_EACH_CLASS
    {
        ALOG(INFO, "LOGGER_INIT") << "Appending Projector of Reference " << i;
        BLOG(INFO, "LOGGER_INIT") << "Appending Projector of Reference " << i;

        _proj.push_back(Projector());

        ALOG(INFO, "LOGGER_INIT") << "Appending Reconstructor of Reference " << i;
        BLOG(INFO, "LOGGER_INIT") << "Appending Reconstructor of Reference " << i;

        _reco.push_back(unique_ptr<Reconstructor>(new Reconstructor()));
    }

    ALOG(INFO, "LOGGER_INIT") << "Setting Up MPI Environment of Reconstructors";
    BLOG(INFO, "LOGGER_INIT") << "Setting Up MPI Environment of Reconstructors";

    FOR_EACH_CLASS
        _reco[i]->setMPIEnv(_commSize, _commRank, _hemi);

    ALOG(INFO, "LOGGER_INIT") << "Refreshing Projectors";
    BLOG(INFO, "LOGGER_INIT") << "Refreshing Projectors";

    refreshProj();

    ALOG(INFO, "LOGGER_INIT") << "Refreshing Reconstructors";
    BLOG(INFO, "LOGGER_INIT") << "Refreshing Reconstructors";

    refreshReco();
}

Volume& MLModel::ref(const int i)
{
    return _ref[i];
}

void MLModel::appendRef(Volume ref)
{
    if (((ref.nColRL() != _size * _pf) && (ref.nColRL() != 0)) ||
        ((ref.nRowRL() != _size * _pf) && (ref.nRowRL() != 0)) ||
        ((ref.nSlcRL() != _size * _pf) && (ref.nSlcRL() != 0)))
        CLOG(FATAL, "LOGGER_SYS") << "Incorrect Size of Appending Reference"
                                  << ": _size = " << _size
                                  << ", nCol = " << ref.nColRL()
                                  << ", nRow = " << ref.nRowRL()
                                  << ", nSlc = " << ref.nSlcRL();

    _ref.push_back(move(ref));
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
    _r = r;

    updateRU();
}

int MLModel::rU() const
{
    return _rU;
}

int MLModel::rPrev() const
{
    return _rPrev;
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

    _FSC.resize(_rU * _pf, _k);

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO, "LOGGER_COMPARE") << "Gathering References from Hemisphere A and Hemisphere B";

    FOR_EACH_CLASS
    {
        IF_MASTER
        {
            MLOG(INFO, "LOGGER_COMPARE") << "Allocating A and B in Fourier Space with Size: "
                                         << _size * _pf
                                         << " X "
                                         << _size * _pf
                                         << " X "
                                         << _size * _pf;

            Volume A(_size * _pf, _size * _pf, _size * _pf, FT_SPACE);
            Volume B(_size * _pf, _size * _pf, _size * _pf, FT_SPACE);

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << i << " from Hemisphere A";

            MPI_Recv_Large(&A[0],
                           A.sizeFT(),
                           MPI_DOUBLE_COMPLEX,
                           HEMI_A_LEAD,
                           i,
                           MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << i << " from Hemisphere B";

            MPI_Recv_Large(&B[0],
                           B.sizeFT(),
                           MPI_DOUBLE_COMPLEX,
                           HEMI_B_LEAD,
                           i,
                           MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Calculating FSC of Reference " << i;
            vec fsc(_rU * _pf);
            FSC(fsc, A, B);
            _FSC.col(i) = fsc;

            MLOG(INFO, "LOGGER_COMPARE") << "Averaging A and B Below a Certain Resolution";

            double r = GSL_MIN_DBL((resA2P(1.0 / A_B_AVERAGE_THRES,
                                           _size,
                                           _pixelSize) + 1) * _pf,
                                   _rU * _pf);
                                   //0.9 * (_r + 1) * _pf);

            MLOG(INFO, "LOGGER_COMPARE") << "Averaging A and B Belower Resolution "
                                         << 1.0 / resP2A(r / _pf - 1, _size, _pixelSize)
                                         << "(Angstrom)";

            #pragma omp parallel for schedule(dynamic)
            VOLUME_FOR_EACH_PIXEL_FT(A)
                if (QUAD_3(i, j, k) < gsl_pow_2(r))
                {
                    Complex avg = (A.getFT(i, j, k) + B.getFT(i, j, k)) / 2;
                    A.setFT(avg, i, j, k);
                    B.setFT(avg, i, j, k);
                }

            MLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                         << i
                                         << " to Hemisphere A";
            
            MPI_Ssend_Large(&A[0],
                            A.sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            HEMI_A_LEAD,
                            i,
                            MPI_COMM_WORLD);

            MLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                         << i
                                         << " to Hemisphere B";

            MPI_Ssend_Large(&B[0],
                            B.sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            HEMI_B_LEAD,
                            i,
                            MPI_COMM_WORLD);
        }
        else
        {
            if ((_commRank == HEMI_A_LEAD) ||
                (_commRank == HEMI_B_LEAD))
            {
                ALOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                             << i
                                             << " from Hemisphere A";
                BLOG(INFO, "LOGGER_COMPARE") << "Snding Reference "
                                             << i
                                             << " from Hemisphere B";

                MPI_Ssend_Large(&_ref[i][0],
                                _ref[i].sizeFT(),
                                MPI_DOUBLE_COMPLEX,
                                MASTER_ID,
                                i,
                                MPI_COMM_WORLD);

                ALOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << i << " from MASTER";
                BLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << i << " from MASTER";

                MPI_Recv_Large(&_ref[i][0],
                               _ref[i].sizeFT(),
                               MPI_DOUBLE_COMPLEX,
                               MASTER_ID,
                               i,
                               MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (isA())
        {
            ALOG(INFO, "LOGGER_COMPARE") << "Broadcasting Reference from A_LEAD";
            MPI_Bcast_Large(&_ref[i][0],
                            _ref[i].sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            0,
                           _hemi);
        }

        if (isB())
        {
            BLOG(INFO, "LOGGER_COMPARE") << "Broadcasting Reference from B_LEAD";
            MPI_Bcast_Large(&_ref[i][0],
                            _ref[i].sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            0,
                            _hemi);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MLOG(INFO, "LOGGER_COMPARE") << "Broadcasting FSC from MASTER";

    MPI_Bcast(_FSC.data(),
              _FSC.size(),
              MPI_DOUBLE,
              MASTER_ID,
              MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
}

void MLModel::lowPassRef(const double thres,
                         const double ew)
{
    FOR_EACH_CLASS
        lowPassFilter(_ref[i], _ref[i], thres, ew);
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
        _SNR.col(i) = _FSC.col(i).array() / (1 - _FSC.col(i).array());
}

void MLModel::refreshTau()
{
    _tau.resize(_size * _pf / 2 - 1, _k);

    FOR_EACH_CLASS
    {
        vec ps(_size * _pf / 2 - 1);
        powerSpectrum(ps, _ref[i], _size * _pf / 2 - 1);
        _tau.col(i) = ps / 2;
    }
}

vec MLModel::tau(const int i) const
{
    return _tau.col(i);
}

int MLModel::resolutionP(const int i,
                         const double thres) const
{
    int result;

    for (result = _SNR.rows() - 1;
         result >= 0;
         result--)
        if (_SNR(result, i) > thres / (1 - thres)) break;

    return result / _pf;
}

int MLModel::resolutionP(const double thres) const
{
    int result = 0;

    FOR_EACH_CLASS
        if (result < resolutionP(i, thres))
            result = resolutionP(i, thres);

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
        _proj[i].setMaxRadius(maxRadius);
}

void MLModel::refreshProj()
{
    FOR_EACH_CLASS
    {
        _proj[i].setProjectee(_ref[i].copyVolume());
        _proj[i].setMaxRadius(_r);
        _proj[i].setPf(_pf);
    }
}

void MLModel::refreshReco()
{
    FOR_EACH_CLASS
    {
        _reco[i]->init(_size,
                       _pf,
                       _sym,
                       _a,
                       _alpha);

        _reco[i]->setMaxRadius(_rU);
    }
}

void MLModel::updateR(const double thres)
{
    // record the frequency
    _rPrev = _r;

    if ([&]()
        {
            if ((_r == _rGlobal) &&
                (_searchType == SEARCH_TYPE_GLOBAL))
            {
                MLOG(INFO, "LOGGER_SYS") << "Using rChangeDecreaseFactor 0.05";

                return determineIncreaseR(0.05);
            }
            else
            {
                MLOG(INFO, "LOGGER_SYS") << "Using rChangeDecreaseFactor 0.5";

                return determineIncreaseR(0.5);
            }
        }())
    {
        FOR_EACH_CLASS
            if (_FSC.col(i)(_pf * _rU - 1) > thres)
            {
                _r = _rU;

                if (_searchType == SEARCH_TYPE_GLOBAL)
                    _r = GSL_MIN_INT(_rGlobal, _r);

                updateRU();

                return;
            }

        _r = resolutionP(thres) + 1;

        if (_searchType == SEARCH_TYPE_GLOBAL)
            _r = GSL_MIN_INT(_rGlobal, _r);
    
        updateRU();
    }

    /***
    _rU += GSL_MAX_INT(MAX_GAP_GLOBAL, MAX_GAP_LOCAL);
    _rU = GSL_MIN_INT(_rU, maxR());
    ***/
    
    /***
    if (_searchType == SEARCH_TYPE_GLOBAL)
    {
        _r += MIN_GAP_GLOBAL;
        _r = GSL_MIN_INT(_rGlobal, _r);
    }
    else
        _r += MIN_GAP_LOCAL;

    _r = GSL_MIN_INT(_r, maxR());
    ***/
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

void MLModel::allReduceVari(const vector<Particle>& par,
                            const int n)
{
    IF_MASTER return;

    _rVari = 0;
    _tVariS0 = 0;
    _tVariS1 = 0;

    double rVari, tVariS0, tVariS1;

    for (size_t i = 0; i < par.size(); i++)
    {
        par[i].vari(rVari,
                    tVariS0,
                    tVariS1);

        _rVari += rVari;
        _tVariS0 += tVariS0;
        _tVariS1 += tVariS1;
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_rVari,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_tVariS0,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE,
                  &_tVariS1,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  _hemi);

    _rVari /= n;
    _tVariS0 /= n;
    _tVariS1 /= n;
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

int MLModel::searchType()
{
    // If the searching needs to stop, return the stop signal.
    if (_searchType == SEARCH_TYPE_STOP) return SEARCH_TYPE_STOP;

    if (_searchType == SEARCH_TYPE_LOCAL)
    {
        // If it is local search, check whether there is no space for
        // improvement or not. If there is, perform further local search, if
        // there is not, stop the search.
        IF_MASTER
        {
            if (_increaseR)
            {
                if (_r > _rT)
                {
                    _rT = _r;
                    _nRNoImprove = 0;
                }
                else
                    _nRNoImprove += 1;

                _searchType = (_nRNoImprove >= MAX_ITER_R_NO_IMPROVE)
                            ? SEARCH_TYPE_STOP
                            : SEARCH_TYPE_LOCAL;
            }
        }
    }
    else
    {
        // If it is global search now, make sure the change of rotations
        // beteween iterations still gets room for improvement.
        IF_MASTER
        {
            /***
            if ((_rChange > _rChangePrev - 0.02 * _stdRChangePrev) &&
                (_r == _rGlobal) &&
                (_rChange < 0.01))
                _nRChangeNoDecrease += 1;
            else
                _nRChangeNoDecrease = 0;
            ***/

            _searchType = ((_r == _rGlobal) && _increaseR)
                        ? SEARCH_TYPE_LOCAL
                        : SEARCH_TYPE_GLOBAL;
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

bool MLModel::increaseR() const
{
    return _increaseR;
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
        if (_rChange > _rChangePrev
                     - rChangeDecreaseFactor
                     * _stdRChangePrev)
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

void MLModel::updateRU()
{
    _rU = GSL_MIN_INT(_r
                    + GSL_MIN_INT(SEARCH_RES_GAP,
                                  AROUND((double)_size / 32)),
                      maxR());
}
