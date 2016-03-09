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

int MLModel::K() const
{
    return _ref.size();
}

int MLModel::size() const
{
    return _ref[0].nColRL();
}

void MLModel::BCastFSC()
{
}

void MLModel::BCastSNR()
{
}
