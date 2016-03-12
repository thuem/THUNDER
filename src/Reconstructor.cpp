/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Reconstructor.h"

Reconstructor::Reconstructor() {}

Reconstructor::Reconstructor(const int size,
                             const int pf,
                             const Symmetry* sym,
                             const double a,
                             const double alpha,
                             const double zeta)
{
    init(size, pf, sym, a, alpha, zeta);
}

Reconstructor::~Reconstructor() {}

void Reconstructor::init(const int size,
                         const int pf,
                         const Symmetry* sym,
                         const double a,
                         const double alpha,
                         const double zeta)
{
    _size = size;
    _pf = pf;
    _sym = sym;
    _a = a;
    _alpha = alpha;
    _zeta = zeta;

    _maxRadius = (_size / 2 - a) * _pf;

    if (_commRank != MASTER_ID)
    {
        _F.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
        _W.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);
        _C.alloc(PAD_SIZE, PAD_SIZE, PAD_SIZE, FT_SPACE);

        SET_0_FT(_F);
        SET_1_FT(_W);
        SET_0_FT(_C);
    }
}

void Reconstructor::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Reconstructor::insert(const Image& src,
                           const Coordinate5D coord,
                           const double w)
{
    if (_commRank == MASTER_ID) return;

    _coord.push_back(coord);

    Image transSrc(_size, _size, FT_SPACE);
    translate(transSrc, src, -coord.x, -coord.y);

    mat33 mat;
    rotate3D(mat, coord.phi, coord.theta, coord.psi);

    IMAGE_FOR_EACH_PIXEL_FT(transSrc)
    {
        vec3 newCor = {(double)i, (double)j, 0};
        vec3 oldCor = mat * newCor *_pf;
        
        if (norm(oldCor) < _maxRadius)
            _F.addFT(transSrc.getFT(i, j) * w, 
                     oldCor(0), 
                     oldCor(1), 
                     oldCor(2), 
                     _pf * _a, 
                     _alpha);
    }
}

void Reconstructor::reconstruct(Volume& dst)
{
    if (_commRank == MASTER_ID) return;

    double c;
    do
    {
        allReduceW();
        c = checkC();
        printf("checkC = %12f\n", c);
    }
    while (c > _zeta);

    allReduceF();

    dst = _F;

    FFT fft;
    fft.bw(dst);

    VOLUME_FOR_EACH_PIXEL_RL(dst)
    {     
        double r = NORM_3(i, j, k) / PAD_SIZE;

        if (r < 0.25 / _pf)
            dst.setRL(dst.getRL(i, j, k)
                    / MKB_RL(r, _pf * _a, _alpha)
                    / TIK_RL(r),
                      i,
                      j,
                      k);
        else
            dst.setRL(0, i, j, k);
    }
}

void Reconstructor::allReduceW()
{
    SET_0_FT(_C);
    
    for (int i = 0; i < _coord.size(); i++)
    {
        mat33 mat;
        rotate3D(mat, _coord[i].phi, _coord[i].theta, _coord[i].psi);
        
        for (int j = -_size / 2; j < _size / 2; j++)
            for (int i = -_size / 2; i <= _size / 2; i++)
            {
                vec3 newCor = {(double)i, (double)j, 0};
                vec3 oldCor = mat * newCor * _pf;

                if (norm(oldCor) < _maxRadius)
                    _C.addFT(_W.getByInterpolationFT(oldCor(0),
                                                     oldCor(1),
                                                     oldCor(2),
                                                     LINEAR_INTERP),
                             oldCor(0),
                             oldCor(1),
                             oldCor(2),
                             _pf * _a,
                             _alpha);
            }
    }

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE, 
                  &_C[0],
                  _C.sizeFT(),
                  MPI_DOUBLE_COMPLEX, 
                  MPI_SUM, 
                  _hemi);

    MPI_Barrier(_hemi);

    symmetrizeC();
    
    //some problems need to be improved: divede zero
    VOLUME_FOR_EACH_PIXEL_FT(_W)
        if (NORM_3(i, j, k) < _maxRadius)
            _W.setFT(_W.getFT(i, j, k, conjugateNo)
                   / _C.getFT(i, j, k, conjugateNo),
                     i,
                     j,
                     k,
                     conjugateNo);
        else
            _W.setFT(COMPLEX(0, 0), i, j, k, conjugateNo);
}

void Reconstructor::allReduceF()
{
    MUL_FT(_F, _W);

    MPI_Barrier(_hemi);

    MPI_Allreduce(MPI_IN_PLACE, 
                  &_F[0],
                  _F.sizeFT(),
                  MPI_DOUBLE_COMPLEX, 
                  MPI_SUM, 
                  _hemi);

    MPI_Barrier(_hemi);

    symmetrizeF();
}

double Reconstructor::checkC() const
{
    int counter = 0;
    double diff = 0;
    VOLUME_FOR_EACH_PIXEL_FT(_C)
        if (NORM_3(i, j, k) < _maxRadius - _pf * _a)
        {
            counter += 1;
            diff += abs(REAL(_C.getFT(i, j, k)) - 1);
        }
    return diff / counter;
}

void Reconstructor::symmetrizeF()
{
    if (_sym == NULL) return;

    Volume symF;
    SYMMETRIZE_FT(symF, _F, *_sym, _maxRadius);
    _F = symF;
}

void Reconstructor::symmetrizeC()
{
    if (_sym == NULL) return;

    Volume symC;
    SYMMETRIZE_FT(symC, _C, *_sym, _maxRadius);
    _C = symC;
}
