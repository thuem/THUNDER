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
    for (int i = 0; i < K; i++)
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
    for (int i = 0; i < K; i++)
    {
        // Broadcast FSC
    }
}

/***
void MLModel::BCastSNR()
{
}
***/
