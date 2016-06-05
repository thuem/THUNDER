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
    FOR_EACH_CLASS
    {
        ALOG(INFO, "LOGGER_INIT") << "Appending Projector of Reference " << i;
        _proj.push_back(Projector());

        ALOG(INFO, "LOGGER_INIT") << "Appending Reconstructor of Reference " << i;
        _reco.push_back(unique_ptr<Reconstructor>(new Reconstructor()));
    }

    ALOG(INFO, "LOGGER_INIT") << "Setting Up MPI Environment of Reconstructors";
    FOR_EACH_CLASS
        _reco[i]->setMPIEnv(_commSize, _commRank, _hemi);

    ALOG(INFO, "LOGGER_INIT") << "Refreshing Projectors";
    refreshProj();

    ALOG(INFO, "LOGGER_INIT") << "Refreshing Reconstructors";
    refreshReco();
}

Volume& MLModel::ref(const int i)
{
    return _ref[i];
}

void MLModel::appendRef(Volume ref)
{
    if (((ref.nColRL() != _size) && (ref.nColRL() != 0)) ||
        ((ref.nRowRL() != _size) && (ref.nRowRL() != 0)) ||
        ((ref.nSlcRL() != _size) && (ref.nSlcRL() != 0)))
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

int MLModel::r() const
{
    return _r;
}

void MLModel::setR(const int r)
{
    _r = r;
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

    _FSC.resize(_r * _pf, _k);

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

            /***
            // check the integrity of transporting using MPI_Status
            MPI_Get_count(&stat, MPI_DOUBLE_COMPLEX, &count);
            if (count != A.sizeFT())
                CLOG(FATAL, "LOGGER_SYS") << "Receiving Incomplete Buffer from Hemisphere A";
                ***/

            MLOG(INFO, "LOGGER_COMPARE") << "Receiving Reference " << i << " from Hemisphere B";

            MPI_Recv_Large(&B[0],
                           B.sizeFT(),
                           MPI_DOUBLE_COMPLEX,
                           HEMI_B_LEAD,
                           i,
                           MPI_COMM_WORLD);

            /***
            // check the integrity of transporting using MPI_Status
            MPI_Get_count(&stat, MPI_DOUBLE_COMPLEX, &count);
            if (count != B.sizeFT())
                CLOG(FATAL, "LOGGER_SYS") << "Receiving Incomplete Buffer from Hemisphere B";
                ***/

            MLOG(INFO, "LOGGER_COMPARE") << "Calculating FSC of Reference " << i;
            vec fsc(_r * _pf);
            FSC(fsc, A, B);
            _FSC.col(i) = fsc;

            MLOG(INFO, "LOGGER_COMPARE") << "Averaging A and B";
            ADD_FT(A, B);
            SCALE_FT(A, 0.5);
            _ref[i] = A.copyVolume();
        }
        else if ((_commRank == HEMI_A_LEAD) ||
                 (_commRank == HEMI_B_LEAD))
        {
            ALOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                         << i
                                         << " from Hemisphere A";
            BLOG(INFO, "LOGGER_COMPARE") << "Sending Reference "
                                         << i
                                         << " from Hemisphere B";

            MPI_Ssend_Large(&_ref[i][0],
                            _ref[i].sizeFT(),
                            MPI_DOUBLE_COMPLEX,
                            MASTER_ID,
                            i,
                            MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO, "LOGGER_COMPARE") << "Broadcasting Average Reference from MASTER";

        MPI_Bcast_Large(&_ref[i][0],
                        _ref[i].sizeFT(),
                        MPI_DOUBLE_COMPLEX,
                        MASTER_ID,
                        MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

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

void MLModel::refreshSNR()
{
    _SNR.resize(_FSC.rows(), _FSC.cols());

    FOR_EACH_CLASS
        _SNR.col(i) = _FSC.col(i).array() / (1 - _FSC.col(i).array());
}

int MLModel::resolutionP(const int i) const
{
    int result;

    for (result = _SNR.rows() - 1;
         result >= 0;
         result--)
        if (_SNR(result, i) > 1) break;

    return result / _pf;
}

int MLModel::resolutionP() const
{
    int result = 0;

    FOR_EACH_CLASS
        if (result < resolutionP(i))
            result = resolutionP(i);

    return result;
}

double MLModel::resolutionA(const int i) const
{
    return resP2A(resolutionP(i), _size, _pixelSize);
}

double MLModel::resolutionA() const
{
    return resP2A(resolutionP(), _size, _pixelSize);
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
        // _reco[i].setMaxRadius(_r);
    }
}

void MLModel::updateR()
{
    FOR_EACH_CLASS
        if (_FSC.col(i)(_pf * _r - 1) > 0.5)
        {
            _r += MIN(MAX_GAP, AROUND(double(_size) / 16));
            _r = MIN(_r, _size / 2 - _a);
            return;
        }

    _r = resolutionP();
    _r += MIN_GAP;
    _r = MIN(_r, _size / 2 - _a);
}

void MLModel::clear()
{
    _ref.clear();

    _proj.clear();
    _reco.clear();
}
