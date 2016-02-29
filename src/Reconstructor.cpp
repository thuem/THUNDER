#include "Reconstructor.h"

Reconstructor::Reconstructor() {}

Reconstructor::Reconstructor(const int nCol,
                             const int nRow,
                             const int nSlc,
                             const int nColImg,
                             const int nRowImg,
                             const double a,
                             const double alpha)
{
    init(nCol, nRow, nSlc, nColImg, nRowImg, a, alpha);
}

/***
Reconstructor::Reconstructor(const Reconstructor& that)
{
    *this = that;
}
***/

Reconstructor::~Reconstructor()
{
    _F.clear();
    _W.clear();
    _C.clear();
}

//////////////////////////////////////////////////////////////////////////////


void Reconstructor::init(const int nCol,
                         const int nRow,
                         const int nSlc,
                         const int nColImg,
                         const int nRowImg,
                         const double a,
                         const double alpha)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;
    _a = a;
    _alpha = alpha;
    _nColImg = nColImg;
    _nRowImg = nRowImg;

    _maxRadius = floor(MIN_3(nCol, nRow, nSlc) / 2 - 1);

    _F.alloc(_nCol, _nRow, _nSlc, fourierSpace);
    _W.alloc(_nCol, _nRow, _nSlc, fourierSpace);
    _C.alloc(_nCol, _nRow, _nSlc, fourierSpace);

    VOLUME_FOR_EACH_PIXEL_FT(_W)
    {
        _F.setFT(COMPLEX(0, 0), i, j, k, conjugateNo);
        _W.setFT(COMPLEX(1, 0), i, j, k, conjugateNo);
        _C.setFT(COMPLEX(0, 0), i, j, k, conjugateNo);
    }

}

void Reconstructor::setCommSize(const int commSize) 
{
    _commSize = commSize;
}

void Reconstructor::setCommRank(const int commRank)
{
    _commRank = commRank;
}


/***
Insert 
    CTF(-1)(Ri)F(3D)(Ri)K(R - Ri)

***/



void Reconstructor::insert(const Image& src,
                           const Coordinate5D coord,
                           const double u,
                           const double v)
{
    _coordWeight.push_back(make_pair(coord, u * v));

    Image transSrc(src.nColRL(), src.nRowRL(), fourierSpace);
    translate(transSrc, src, -coord.x, -coord.y);

    mat33 mat;
    rotate3D(mat, -coord.phi, -coord.theta, -coord.psi);

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
    vector<corWeight>::iterator iter;
    for (iter = _coordWeight.begin(); iter != _coordWeight.end(); ++iter)
    {
        mat33 mat;
        rotate3D(mat, -iter->first.phi, -iter->first.theta, -iter->first.psi);
        
        for (int j = -_nRowImg / 2; j < _nRowImg / 2; j++)
        {
            for (int i = 0; i <= _nColImg / 2; i++)
            {
                vec3 newCor = {i, j, 0};
                vec3 oldCor = mat * newCor;

                if (norm(oldCor) < _maxRadius)
                    _C.addFT(_W.getByInterpolationFT(oldCor(0),
                                                     oldCor(1),
                                                     oldCor(2),
                                                     LINEAR_INTERP) 
                           * (iter->second),
                             oldCor(0),
                             oldCor(1),
                             oldCor(2));
            }
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
    {
        _W.setFT(_W.getFT(i, j, k, conjugateNo) /
                 _C.getFT(i, j, k, conjugateNo),
                 i,
                 j,
                 k,
                 conjugateNo);
    }
}



void Reconstructor::reduceF(int root,
                            MPI_Comm world) 
{

    VOLUME_FOR_EACH_PIXEL_FT(_F)
    {
        _F.setFT(_F.getFT(i, j, k, conjugateNo) *
                 _W.getFT(i, j, k, conjugateNo),
                 i,
                 j,
                 k,
                 conjugateNo);
    }
    MPI_Barrier(world);

    if (_commRank == root)
    {   
        MPI_Reduce(MPI_IN_PLACE,
                   &_F[0],
                   _F.sizeFT(),
                   MPI_DOUBLE_COMPLEX,
                   MPI_SUM,
                   root,
                   world);
    }
    else
    {
        MPI_Reduce(&_F[0],
                   &_F[0],
                   _F.sizeFT(),
                   MPI_DOUBLE_COMPLEX,
                   MPI_SUM,
                   root,
                   world); 
    }

    MPI_Barrier(world);
}

void Reconstructor::getF(Volume& dst)
{
    dst = _F;
}

/***
void Reconstructor::constructor(const char *dst) 
{
    ImageFile imf;    
    imf.writeImage(dst, _F);
}
***/
