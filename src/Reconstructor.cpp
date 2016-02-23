#include "Reconstructor.h"

Reconstructor::Reconstructor() {}

Reconstructor::Reconstructor(const int nCol,
                             const int nRow,
                             const int nSlc,
                             const double a,
                             const double alpha,
                             const int imCol,
                             const int imRow) 
{
    initial(nCol, nRow, nSlc, a, alpha, imCol, imRow);
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


void Reconstructor::initial(const int nCol,
                            const int nRow,
                            const int nSlc,
                            const double a,
                            const double alpha,
                            const int imCol,
                            const int imRow)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;
    _a = a;
    _alpha = alpha;
    _imCol = imCol;
    _imRow = imRow;

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
        
        //_W.addFT(COMPLEX(1, 0), oldCor(0), oldCor(1), oldCor(2), _a, _alpha);
        _F.addFT(transSrc.getFT(i, j) * u * v, oldCor(0), oldCor(1), oldCor(2), _a, _alpha);
    }
}


void Reconstructor::reduceWM()
{
    if (_commRank != 0) return;


    

}
void Reconstructor::reduceWS() 
{
    if (_commRank == 0) return;

    for (vector<corWeight>::iterator itera = _coordWeight.begin(); itera != _coordWeight.end(); ++itera)
    {
        Coordinate5D coord;
        double wGetPC;

        coord = (*itera).first;
        wGetPC = (*itera).second;
        
        mat33 mat;
        rotate3D(mat, -coord.phi, -coord.theta, -coord.psi);
        
        for (int j = -_imRow / 2; j < _imRow / 2; j++)
        {
            for (int i = 0; i <= _imCol / 2; i++)
            {
                vec3 newCor = {i, j, 0};
                vec3 oldCor = mat * newCor;
                Complex wGetByIntpola = _W.getByInterpolationFT(oldCor(0), oldCor(1), oldCor(2), LINEAR_INTERP);
                _C.addFT(wGetByIntpola * wGetPC, oldCor(0), oldCor(1), oldCor(2), _a, _alpha);
            }
        }
    }





}

void Reconstructor::broadcastWM() 
{


}
void Reconstructor::broadcastWS() 
{


}


void Reconstructor::reduceFM() {


}

void Reconstructor::reduceFS() {


}



