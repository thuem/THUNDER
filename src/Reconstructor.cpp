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
                             const double a,
                             const double alpha)
{
    init(size, pf, a, alpha);
}

Reconstructor::~Reconstructor() {}

void Reconstructor::init(const int size,
                         const int pf,
                         const double a,
                         const double alpha)
{
    _size = size;
    _pf = pf;
    _a = a;
    _alpha = alpha;

    _maxRadius = _size / 2 - a;

    _F.alloc(_size, _size, _size, fourierSpace);
    _W.alloc(_size, _size, _size, fourierSpace);
    _C.alloc(_size, _size, _size, fourierSpace);

    SET_0_FT(_F);
    SET_1_FT(_W);
    SET_0_FT(_C);
}

void Reconstructor::setCommSize(const int commSize) 
{
    _commSize = commSize;
}

void Reconstructor::setCommRank(const int commRank)
{
    _commRank = commRank;
}

void Reconstructor::insert(const Image& src,
                           const Coordinate5D coord,
                           const double u,
                           const double v)
{
    _coord.push_back(coord);

    Image transSrc(_size, _size, fourierSpace);
    translate(transSrc, src, -coord.x, -coord.y);

    mat33 mat;
    rotate3D(mat, coord.phi, coord.theta, coord.psi);

    IMAGE_FOR_EACH_PIXEL_FT(transSrc)
    {
        vec3 newCor = {i, j, 0};
        vec3 oldCor = mat * newCor;
        
        if (norm(oldCor) < _maxRadius)
            _F.addFT(transSrc.getFT(i, j) * u * v, 
                     oldCor(0), 
                     oldCor(1), 
                     oldCor(2), 
                     _a, 
                     _alpha);
    }
}

void Reconstructor::allReduceW(MPI_Comm workers) 
{
    SET_0_FT(_C);
    
    for (int i = 0; i < _coord.size(); i++)
    {
        mat33 mat;
        rotate3D(mat, _coord[i].phi, _coord[i].theta, _coord[i].psi);
        
        for (int j = -_size / 2; j < _size / 2; j++)
            for (int i = -_size / 2; i <= _size / 2; i++)
            {
                vec3 newCor = {i, j, 0};
                vec3 oldCor = mat * newCor;

                if (norm(oldCor) < _maxRadius)
                    _C.addFT(_W.getByInterpolationFT(oldCor(0),
                                                     oldCor(1),
                                                     oldCor(2),
                                                     LINEAR_INTERP),
                             oldCor(0),
                             oldCor(1),
                             oldCor(2),
                             _a,
                             _alpha);
            }
    }

    MPI_Barrier(workers);

    MPI_Allreduce(MPI_IN_PLACE, 
                  &_C[0],
                  _C.sizeFT(),
                  MPI_DOUBLE_COMPLEX, 
                  MPI_SUM, 
                  workers);

    MPI_Barrier(workers);
    
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

void Reconstructor::allReduceF(MPI_Comm world) 
{
    MUL_FT(_F, _W);

    // display(1, "testFWC-after_F*_W");
    MPI_Barrier(world);

    MPI_Allreduce(MPI_IN_PLACE, 
                  &_F[0],
                  _F.sizeFT(),
                  MPI_DOUBLE_COMPLEX, 
                  MPI_SUM, 
                  world);
    MPI_Barrier(world);
}

void Reconstructor::constructor(const char dst[])
{
    Volume result(_pf * _size, _pf * _size, _pf * _size, fourierSpace);
    SET_0_FT(result);
    VOLUME_FOR_EACH_PIXEL_FT(_F)
        result.setFT(_F.getFT(i, j, k), i, j, k);

    FFT fft;
    fft.bw(result);

    VOLUME_FOR_EACH_PIXEL_RL(result)
    {     
        double r = NORM_3(i, j, k) / (_pf * _size);

        if (r < 0.5 / _pf)
            result.setRL((result.getRL(i, j, k)
                        / MKB_RL(r, _a, _alpha)
                        / TIK_RL(r)),
                         i,
                         j,
                         k);
        else
            result.setRL(0, i, j, k);
    }

    ImageFile imf;
    imf.readMetaData(result);
    imf.writeVolume(dst, result);
}

void Reconstructor::display(const int rank, 
                            const char name[])
{
    if (_commRank != rank)
        return;

    FILE *disfile = fopen(name, "w");
    
    VOLUME_FOR_EACH_PIXEL_FT(_W)
    {
        Complex f = _F.getFT(i , j , k, conjugateNo);
        Complex w = _W.getFT(i , j , k, conjugateNo);
        Complex c = _C.getFT(i , j , k, conjugateNo);
        fprintf(disfile, "%5d : %5d : %5d\t %12f,%12f\t   %12f,%12f\t   %12f,%12f\t\n",
                i, j, k,
                REAL(f), IMAG(f),
                REAL(w), IMAG(w),
                REAL(c), IMAG(c));
    }
    fclose(disfile);
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
