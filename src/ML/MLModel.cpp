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
        _proj.push_back(Projector());
        _reco.push_back(Reconstructor());
    }

    ALOG(INFO) << "Setting Up MPI Environment of Reconstructors";
    FOR_EACH_CLASS
        _reco[i].setMPIEnv(_commSize, _commRank, _hemi);

    ALOG(INFO) << "Refreshing Projectors";
    refreshProj();

    ALOG(INFO) << "Refreshing Reconstructors";
    refreshReco();
}

Volume& MLModel::ref(const int i)
{
    return _ref[i];
}

void MLModel::appendRef(const Volume& ref)
{
    _ref.push_back(ref);
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
    return _reco[i];
}

void MLModel::BcastFSC()
{
    MPI_Barrier(MPI_COMM_WORLD);

    FOR_EACH_CLASS
    {
        if (_commRank == MASTER_ID)
        {
            Volume A(_size, _size, _size, FT_SPACE);
            Volume B(_size, _size, _size, FT_SPACE);
            MPI_Recv(&A[0],
                     A.sizeFT(),
                     MPI_DOUBLE_COMPLEX,
                     HEMI_A_LEAD,
                     i,
                     MPI_COMM_WORLD,
                     NULL);
            MPI_Recv(&B[0],
                     B.sizeFT(),
                     MPI_DOUBLE_COMPLEX,
                     HEMI_B_LEAD,
                     i,
                     MPI_COMM_WORLD,
                     NULL);
            FSC(_FSC[i], A, B, _r);
        }
        else if ((_commRank == HEMI_A_LEAD) ||
                 (_commRank == HEMI_B_LEAD))
        {
            MPI_Ssend(&_ref[i],
                      _ref[i].sizeFT(),
                      MPI_DOUBLE_COMPLEX,
                      MASTER_ID,
                      i,
                      MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double* FSC = new double[_k * _r];

    if (_commRank == MASTER_ID)
        FOR_EACH_CLASS
            for (int j = 0; j < _r; j++)
                FSC[i * _r + j] = _FSC[i](j);

    MPI_Bcast(FSC,
              _k * _r,
              MPI_DOUBLE,
              MASTER_ID,
              MPI_COMM_WORLD);

    if (_commRank != MASTER_ID)
        FOR_EACH_CLASS
        {
            _FSC[i].resize(_r);
            for (int j = 0; j < _r; j++)
                _FSC[i](j) = FSC[i * _r + j];
        }

    delete[] FSC;

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
    FOR_EACH_CLASS
        _SNR[i] = _FSC[i] / (1 + _FSC[i]);
}

int MLModel::resolutionP(const int i) const
{
    return uvec(find(_SNR[i] > 1, 1))(0);
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
    // TODO: considering padding factor
    return resP2A(resolutionP(i), _size, _pixelSize);
}

double MLModel::resolutionA() const
{
    // TODO: considering padding factor
    return resP2A(resolutionP(), _size, _pixelSize);
}

void MLModel::refreshProj()
{
    FOR_EACH_CLASS
    {
        _proj[i].setProjectee(_ref[i]);
        _proj[i].setMaxRadius(_r);
        _proj[i].setPf(_pf);
    }
}

void MLModel::refreshReco()
{
    FOR_EACH_CLASS
    {
        _reco[i].init(_size / _pf,
                      _pf,
                      _sym,
                      _a,
                      _alpha);
        _reco[i].setMaxRadius(_r);
    }
}

void MLModel::updateR()
{
    // TODO: considering padding factor
    FOR_EACH_CLASS
        if (_FSC[i](_r) > 0.2)
        {
            _r += AROUND(double(_size) / 8);
            _r = MIN(_r, _size / 2 - _a);
            return;
        }

    _r += 10;
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
