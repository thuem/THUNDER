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

Reconstructor::Reconstructor(const Reconstructor& that)
{
    *this = that;
}

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

    _F.alloc(_nCol, _nRow, _nSlc, fourierSpace);
    _W.alloc(_nCol, _nRow, _nSlc, fourierSpace);
    _C.alloc(_nCol, _nRow, _nSlc, fourierSpace);

    VOLUME_FOR_EACH_PIXEL_FT(_W)
    {
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

    MPI_Allreduce(&_C[0], 
                  &_C[0], 
                  _nCol * _nRow * _nSlc, 
                  MPI_C_COMPLEX, 
                  MPI_SUM, 
                  workers);

    DIV_FT(_W, _C);
    /***
    VOLUME_FOR_EACH_PIXEL_FT(_W)
    {
        _W.setFT(_W.getFT(i, j, k, conjugateNo) /
                 _C.getFT(i, j, k, conjugateNo),
                 i,
                 j,
                 k,
                 conjugateNo);
    }
    ***/
}



void Reconstructor::reduceF(const int root,
                            MPI_Comm world) 
{
    if (_commRank != 0) return;

    VOLUME_FOR_EACH_PIXEL_FT(_F)
    {
        _F.setFT(_F.getFT(i, j, k, conjugateNo) *
                 _W.getFT(i, j, k, conjugateNo),
                 i,
                 j,
                 k,
                 conjugateNo);
    }

    MPI_Reduce(&_F[0],
               &_F[0],
               _nCol * _nRow * _nSlc,
               MPI_C_COMPLEX,
               MPI_SUM,
               root,
               world);
}

void Reconstructor::constructor(const char *dst) 
{
    ImageFile imf;    
    imf.writeImage(dst, _F);
}


