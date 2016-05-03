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
    ALOG(INFO) << "Appending Projectors and Reconstructors";
    FOR_EACH_CLASS
    {
        ALOG(INFO) << "Appending Projector of Reference " << i;
        _proj.push_back(Projector());

        ALOG(INFO) << "Appending Reconstructor of Reference " << i;
        _reco.push_back(std::unique_ptr<Reconstructor>(new Reconstructor()));
    }

    ALOG(INFO) << "Setting Up MPI Environment of Reconstructors";
    FOR_EACH_CLASS
        _reco[i]->setMPIEnv(_commSize, _commRank, _hemi);

    ALOG(INFO) << "Refreshing Projectors";
    refreshProj();

    ALOG(INFO) << "Refreshing Reconstructors";
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
        LOG(FATAL) << "Incorrect Size of Appending Reference"
                   << ": _size = " << _size
                   << ", nCol = " << ref.nColRL()
                   << ", nRow = " << ref.nRowRL()
                   << ", nSlc = " << ref.nSlcRL();

    _ref.push_back(std::move(ref));
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
    MLOG(INFO) << "Setting Size of _FSC";

    _FSC.set_size(_r * _pf, _k);

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO) << "Gathering References from Hemisphere A and Hemisphere B";

    FOR_EACH_CLASS
    {
        IF_MASTER
        {
            MPI_Status stat;

            Volume A(_size * _pf, _size * _pf, _size * _pf, FT_SPACE);
            Volume B(_size * _pf, _size * _pf, _size * _pf, FT_SPACE);

            if ((&A[0] == NULL) && (&B[0] == NULL))
                LOG(FATAL) << "Failed to Allocate Space for Storing a Reference";

            MLOG(INFO) << "Receiving Reference " << i << " from Hemisphere A";

            MPI_Recv(&A[0],
                     A.sizeFT(),
                     MPI_DOUBLE_COMPLEX,
                     HEMI_A_LEAD,
                     i,
                     MPI_COMM_WORLD,
                     &stat);

            MLOG(INFO) << "Receiving Reference " << i << " from Hemisphere B";

            MPI_Recv(&B[0],
                     B.sizeFT(),
                     MPI_DOUBLE_COMPLEX,
                     HEMI_B_LEAD,
                     i,
                     MPI_COMM_WORLD,
                     &stat);

            // TODO: check transporting using MPI_Status

            MLOG(INFO) << "Calculating FSC of Reference " << i;
            // FSC(_FSC.col(i), A, B, _r);
            vec fsc(_r * _pf);
            FSC(fsc, A, B);
            _FSC.col(i) = fsc;

            MLOG(INFO) << "Averaging A and B" << i;
            ADD_FT(A, B);
            SCALE_FT(A, 0.5);
            _ref[i] = A.copyVolume();

            /***
            MLOG(INFO) << "Sending Average Reference to Hemisphere A";
            MPI_Ssend(&A[0],
                      A.sizeFT(),
                      MPI_DOUBLE_COMPLEX,
                      HEMI_A_LEAD,
                      i,
                      MPI_COMM_WORLD);

            MLOG(INFO) << "Sending Average Reference to Hemisphere B";
            MPI_Ssend(&B[0],
                      B.sizeFT(),
                      MPI_DOUBLE_COMPLEX,
                      HEMI_B_LEAD,
                      i,
                      MPI_COMM_WORLD);
                      ***/
        }
        else if ((_commRank == HEMI_A_LEAD) ||
                 (_commRank == HEMI_B_LEAD))
        {
            ALOG(INFO) << "Sending Reference " << i << " from Hemisphere A";
            BLOG(INFO) << "Sending Reference " << i << " from Hemisphere B";

            MPI_Ssend(&_ref[i][0],
                      _ref[i].sizeFT(),
                      MPI_DOUBLE_COMPLEX,
                      MASTER_ID,
                      i,
                      MPI_COMM_WORLD);

            /***
            ALOG(INFO) << "Receiving Average Reference from Hemisphere A";
            ***/
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MLOG(INFO) << "Broadcasting Average Reference from MASTER";

        MPI_Bcast(&_ref[i][0],
                  _ref[i].sizeFT(),
                  MPI_DOUBLE_COMPLEX,
                  MASTER_ID,
                  MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MLOG(INFO) << "Broadcasting FSC from MASTER";

    MPI_Bcast(_FSC.memptr(),
              _FSC.n_elem,
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
    _SNR.copy_size(_FSC);

    FOR_EACH_CLASS
        _SNR.col(i) = _FSC.col(i) / (1 - _FSC.col(i));
}

int MLModel::resolutionP(const int i) const
{
    return uvec(find(_SNR.col(i) > 1, 1, "last"))(0) / _pf;
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
        if (_FSC.col(i)(_pf * _r - 1) > 0.2)
        {
            _r += MIN(MAX_GAP, AROUND(double(_size) / 16));
            _r = MIN(_r, _size / 2 - _a);
            return;
        }

    _r += MIN_GAP;
    _r = MIN(_r, _size / 2 - _a);
}

void MLModel::clear()
{
    _ref.clear();
    _FSC.clear();
    _SNR.clear();
    _proj.clear();
    _reco.clear();
}
