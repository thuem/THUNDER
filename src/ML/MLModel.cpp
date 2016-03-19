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

void MLModel::init(const int r,
                   const int pf,
                   const double pixelSize,
                   const double a,
                   const double alpha)
{
    _r = r;
    _pf = pf;
    _pixelSize = pixelSize;
    _a = a;
    _alpha = alpha;
}

void MLModel::appendRef(const Volume& ref)
{
    _ref.push_back(ref);
}

int MLModel::K() const
{
    return _ref.size();
}

int MLModel::size() const
{
    return _ref[0].nColRL();
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
            Volume A(size(), size(), size(), FT_SPACE);
            Volume B(size(), size(), size(), FT_SPACE);
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

    double* FSC = new double[K() * _r];

    if (_commRank == MASTER_ID)
        FOR_EACH_CLASS
            for (int j = 0; j < _r; j++)
                FSC[i * _r + j] = _FSC[i](j);

    MPI_Bcast(FSC,
              K() * _r,
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
    return resP2A(resolutionP(i), size(), _pixelSize);
}

double MLModel::resolutionA() const
{
    // TODO: considering padding factor
    return resP2A(resolutionP(), size(), _pixelSize);
}

void MLModel::refreshProjector()
{
    FOR_EACH_CLASS
    {
        _proj[i].setProjectee(_ref[i]);
        _proj[i].setMaxRadius(_r);
    }
}

void MLModel::updateR()
{
    // TODO: considering padding factor
    FOR_EACH_CLASS
        if (_FSC[i](_r) > 0.2)
        {
            _r += AROUND(double(size()) / 8);
            _r = MIN(_r, size() / 2 - _a);
            return;
        }

    _r += 10;
    _r = MIN(_r, size() / 2 - _a);
}

void MLModel::clear()
{
    _ref.clear();
    _FSC.clear();
    _SNR.clear();
    _proj.clear();
    _reco.clear();
}
