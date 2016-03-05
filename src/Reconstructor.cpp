#include "Reconstructor.h"

Reconstructor::Reconstructor() {}

Reconstructor::Reconstructor(const int size,
                             const int pf,
                             const double a,
                             const double alpha)
{
    init(size, pf, a, alpha);
}

/***
Reconstructor::Reconstructor(const int nCol,
                             const int nRow,
                             const int nSlc,
                             const int nColImg,
                             const int nRowImg,
                             const int pf,
                             const double a,
                             const double alpha)
{
    init(nCol, nRow, nSlc, nColImg, nRowImg, pf, a, alpha);
}
***/

/***
Reconstructor::Reconstructor(const Reconstructor& that)
{
    *this = that;
}
***/

Reconstructor::~Reconstructor() {}
/***
{
    _F.clear();
    _W.clear();
    _C.clear();
}
***/

/***
Reconstructor& Reconstructor::operator=(const Reconstructor& that)
{
    _nCol = that._nCol;
    _nRow = that._nRow;
    _nSlc = that._nSlc;
    
    _a = that._a;
    _alpha = that._alpha;
        
    _commRank = that._commRank;
    _commSize = that._commSize;

    _F = that._F;
    _W = that._W;
    _C = that._C;

    _coordWeight = that._coordWeight;
    
    _nColImg = that._nColImg;
    _nRowImg = that._nRowImg;

    _maxRadius = that._maxRadius;

}
***/

//////////////////////////////////////////////////////////////////////////////


void Reconstructor::init(const int nCol,
                         const int nRow,
                         const int nSlc,
                         const int nColImg,
                         const int nRowImg,
                         const int pf,
                         const double a,
                         const double alpha)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;
    _nColImg = nColImg;
    _nRowImg = nRowImg;

    _a = a;
    _alpha = alpha;

    _pf = pf;

    // _maxRadius = floor(MIN_3(_pf * nCol, _pf * nRow, _pf * nSlc) / 2 - _pf * a);
    _maxRadius = floor(MIN_3(nCol, nRow, nSlc) / 2 - a);

    _F.alloc(_pf * _nCol, _pf * _nRow, _pf * _nSlc, fourierSpace);
    _W.alloc(_pf * _nCol, _pf * _nRow, _pf * _nSlc, fourierSpace);
    _WN.alloc(_pf * _nCol, _pf * _nRow, _pf * _nSlc, fourierSpace);
    _C.alloc(_pf * _nCol, _pf * _nRow, _pf * _nSlc, fourierSpace);

    VOLUME_FOR_EACH_PIXEL_FT(_W)
    {
        _F.setFT(COMPLEX(0, 0), i, j, k, conjugateNo);
        _W.setFT(COMPLEX(1, 0), i, j, k, conjugateNo);
        _WN.setFT(COMPLEX(0, 0), i, j, k, conjugateNo);
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
    rotate3D(mat, coord.phi, coord.theta, coord.psi);

    meshReverse(transSrc);

    IMAGE_FOR_EACH_PIXEL_FT(transSrc)
    {
        vec3 newCor = {i, j, 0};
        // vec3 oldCor = _pf * mat * newCor;
        vec3 oldCor = mat * newCor;
        
        /***
        if (norm(oldCor) < _maxRadius)
            _F.addFT(transSrc.getFT(i, j) * u * v, 
                     oldCor(0), 
                     oldCor(1), 
                     oldCor(2), 
                     _pf * _a, 
                     _alpha);
                     ***/
        if (norm(oldCor) < _maxRadius)
            _F.addFT(transSrc.getFT(i, j) * u * v, 
                     oldCor(0), 
                     oldCor(1), 
                     oldCor(2), 
                     _a, 
                     _alpha);

        if (norm(oldCor) < _maxRadius)
        {
            /***
            _F.addFT(transSrc.getFT(i, j) * u * v,
                     AROUND(oldCor(0)),
                     AROUND(oldCor(1)),
                     AROUND(oldCor(2)));
                     ***/
            /***
            _WN.addFT(COMPLEX(1, 0),
                      AROUND(oldCor(0)),
                      AROUND(oldCor(1)),
                      AROUND(oldCor(2)));
                      ***/
        }
    }
}

void Reconstructor::allReduceW(MPI_Comm workers) 
{
    initC();
    
    vector<corWeight>::iterator iter;
    for (iter = _coordWeight.begin(); iter != _coordWeight.end(); ++iter)
    {
        mat33 mat;
        rotate3D(mat, iter->first.phi, iter->first.theta, iter->first.psi);
        
        for (int j = -_nRowImg / 2; j < _nRowImg / 2; j++)
        {
            for (int i = 0; i <= _nColImg / 2; i++)
            {
                vec3 newCor = {i, j, 0};
                // vec3 oldCor = _pf * mat * newCor;
                vec3 oldCor = mat * newCor;

                if (norm(oldCor) < _maxRadius)
                    _C.addFT(_W.getByInterpolationFT(oldCor(0),
                                                     oldCor(1),
                                                     oldCor(2),
                                                     LINEAR_INTERP) 
                           * (iter->second),
                             oldCor(0),
                             oldCor(1),
                             oldCor(2),
                             // _pf * _a,
                             _a,
                             _alpha);
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

    
    
    
    ///////////////////////////////////////////////////////////
    //to set no zero
    //VOLUME_FOR_EACH_PIXEL_FT(_C)
    //{
    //    Complex c = _C.getFT(i, j, k, conjugateNo);
    //    if (REAL(c) == 0) {
    //        REAL(c) = DBL_MAX;
    //        _C.setFT(c, i, j ,k, conjugateNo);
    //    }
    //}

    //some problems need to be improved: divede zero
    VOLUME_FOR_EACH_PIXEL_FT(_W)
    {
        if (NORM_3(i, j, k) < _maxRadius)
        {
            Complex c = _C.getFT(i, j, k, conjugateNo);
            Complex w = _W.getFT(i, j, k, conjugateNo); 
            _W.setFT(w / c, i, j, k, conjugateNo);
            /***
            _W.setFT(REAL(c) < 0.0001 ? COMPLEX(0, 0) : w / c,
                     i,
                     j,
                     k,
                     conjugateNo);
                     ***/
    //        _C.setFT(COMPLEX(0, 0), i, j, k, conjugateNo);
        }
        else
        {
            _W.setFT(COMPLEX(0, 0), i, j, k, conjugateNo);
        }
    }

}

void Reconstructor::initC() 
{
    VOLUME_FOR_EACH_PIXEL_FT(_C)
    {
        _C.setFT(COMPLEX(0, 0), i, j, k, conjugateNo);
    }
}


void Reconstructor::reduceF(int root,
                            MPI_Comm world) 
{
    /***
    VOLUME_FOR_EACH_PIXEL_FT(_WN)
        if (REAL(_WN.getFT(i, j, k)) != 0)
            _F.setFT(_F.getFT(i, j, k)
                   * (1.0 / REAL(_WN.getFT(i, j, k))),
                     i, j, k);
   // meshReverse(_F);
   // ***/
   VOLUME_FOR_EACH_PIXEL_FT(_F)
   {
       _F.setFT(_F.getFT(i, j, k, conjugateNo) *
                _W.getFT(i, j, k, conjugateNo),
                i,
                j,
                k,
                conjugateNo);
   }

    display(1, "testFWC-after_F*_W");
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


void Reconstructor::constructor(const char dst[])
{
    Volume result;
    result = _F;
    
    meshReverse(result);

    FFT fft;
    fft.bw(result);

#ifdef DEBUGCONSTRUCTOR
    FILE *disfile1 = fopen("constructor1", "w");
    FILE *disfile2 = fopen("constructor2", "w");
#endif

    VOLUME_FOR_EACH_PIXEL_RL(result)
    {     
        double r = NORM_3(abs(i - _pf * _nCol / 2),
                          abs(j - _pf * _nRow / 2),
                          abs(k - _pf * _nSlc / 2))
                 / (_pf * _nCol);

#ifdef DEBUGCONSTRUCTOR
        /***
        double res1 = result.getRL(i , j , k);
        double mkb = MKB_RL(r / _nCol, _a, _alpha);
        double tik = TIK_RL(r / _nCol);
        fprintf(disfile1, "%5d : %5d : %5d\t %f\t   %f\t   %f\t\n",
                i, j, k, res1, mkb, tik);
                ***/
#endif

        if (r < 0.5 / _pf)
        {
            result.setRL((result.getRL(i, j, k)
                        // / MKB_RL(r, _pf * _a, _alpha)
                        / MKB_RL(r, _a, _alpha)
                        / TIK_RL(r)),
                         i,
                         j,
                         k);
        }
        else
            result.setRL(0, i, j, k);

#ifdef DEBUGCONSTRUCTOR
        double res = result.getRL(i , j , k);
        fprintf(disfile2, "%5d : %5d : %5d\t %f\n",
                i, j, k, res);
#endif
    }

#ifdef DEBUGCONSTRUCTOR
    fclose(disfile1);
    fclose(disfile2);
#endif

    ImageFile imf;
    imf.readMetaData(result);
    //imf.display();
    imf.writeImage(dst, result);

    Image img(_pf * _nCol, _pf * _nRow, realSpace);
    slice(img, result, _pf * _nSlc / 2);
    img.saveRLToBMP("slice.bmp");
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
