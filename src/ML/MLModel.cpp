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

MLModel::MLMode() {}

MLModel::appendRef(const Volume& ref)
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

void MLModel::BCastFSC()
{
    FOR_EACH_CLASS
    {
        // if master
        {
            Volume A(size(), size(), size(), FT_SPACE);
            Volume B(size(), size(), size(), FT_SPACE);
            // get A and B
            FSC(_FSC[i], A, B, _r);
        }
        // if A[0]
        // if B[0]
        // other
    }
    FOR_EACH_CLASS
    {
        // Broadcast FSC
    }
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
        _SNR(i) = _FSC(i) / (1 + _FSC(i));
}

int MLModel::resolutionP(const int i) const
{
    return uvec(find(_SNR(i) > 1, 1))(0);
}

int MLModel::resolutionP() const
{
    int result = 0;

    FOR_EACH_CLASS
        if (result < resolution(i))
            result = resolution(i);

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
        _proj[i].setProjectee(_ref[i]);
}

/***
void MLModel::BCastSNR()
{
}
***/
