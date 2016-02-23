#include "Reconstructor.h"

Reconstructor::Reconstructor() {}

Reconstructor::Reconstructor(const int nCol,
                             const int nRow,
                             const int nSlc,
                             const double a,
                             const double alpha) 
{
    initial(nCol, nRow, nSlc, a, alpha);
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
                            const double alpha)
{
    _nCol = nCol;
    _nRow = nRow;
    _nSlc = nSlc;
    _a = a;
    _alpha = alpha;

    _F.alloc(_nCol, _nRow, _nSlc, fourierSpace);
    _W.alloc(_nCol, _nRow, _nSlc, fourierSpace);
    _C.alloc(_nCol, _nRow, _nSlc, fourierSpace);

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
    _coordWeight.push_back(make_pair(coord, v));

    Image transSrc(src.nColRL(), src.nRowRL(), fourierSpace);
    translate(transSrc, src, coord.x, coord.y);
    

    mat33 mat;
    rotate3D(mat, coord.phi, coord.theta, coord.psi);

    IMAGE_FOR_EACH_PIXEL_FT(transSrc)
    {
        vec3 imgCor = {i, j, 0};
        vec3 oriCor = mat * imgCor;

        _W.addFT(COMPLEX(1, 0), oriCor(0), oriCor(1), oriCor(2), _a, _alpha);
        _F.addFT(transSrc.getFT(i, j), oriCor(0), oriCor(1), oriCor(2), _a, _alpha);
    }

}


void Reconstructor::reduceAllWeight_master()
{
    if (_commRank != 0) return;


    

}

void Reconstructor::reduceAllWeight_slave() {


}

void Reconstructor::scatterAllWeight_master() {


}
void Reconstructor::scatterAllWeight_slave() {


}


void Reconstructor::reduceAllFTImage_master() {


}

void Reconstructor::reduceAllFTImage_slave() {


}



