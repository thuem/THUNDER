/*******************************************************************************
 * Author: Mingxu Hu
 * Dependency:
 * Test:
 * Execution:
 * Description:
 *
 * Manual:
 * ****************************************************************************/

#include "Transformation.h"

/***
void symmetryRL(Volume& dst,
                const Volume& src,
                const Symmetry& sym,
                const double r)
{
    dst = src;

    mat33 L, R;
    Volume se(src.nColRL(), src.nRowRL(), src.nSlcRL(), RL_SPACE);
    for (int i = 0; i < sym.nSymmetryElement(); i++)
    {
        sym.get(L, R, i);
        VOL_TRANSFORM_MAT_RL(se, src, R, r);
        ADD_RL(dst, se);
    }
}

void symmetryFT(Volume& dst,
                const Volume& src,
                const Symmetry& sym,
                const double r)
{
}
***/
