
#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

//#include <float.h>
#include <armadillo>
#include <utility>
#include <vector>
#include <mpi.h>

#include "Coordinate5D.h"
#include "Functions.h"

#include "Euler.h"

#include "FFT.h"

#include "Image.h"
#include "ImageFunctions.h"
#include "ImageFile.h"
#include "Volume.h"

#define DEBUGCONSTRUCTOR

using namespace arma;

using namespace std;

typedef pair<Coordinate5D, double> corWeight;

class Reconstructor
{
    private:

        int _nCol;
        int _nRow;
        int _nSlc;

        double _a = 2;
        double _alpha = 3.6;
        
        int _commRank = 0;
        int _commSize = 1;

        Volume _F;
        Volume _W;
        Volume _C;

        vector<corWeight> _coordWeight;
        int _nColImg;
        int _nRowImg;

        int _maxRadius;

    public:

        Reconstructor();

        Reconstructor(const int nCol,
                      const int nRow,
                      const int nSlc,
                      const int nColImg,
                      const int nRowImg,
                      const double a,
                      const double alpha);

        // Reconstructor(const Reconstructor& that);

        ~Reconstructor();

        Reconstructor& operator=(const Reconstructor& that);
////////////////////////////////////////////////////////////////////////////////////////


        void init(const int nCol,
                  const int nRow,
                  const int nSlc,
                  const int nColImg,
                  const int nRowImg,
                  const double a,
                  const double alpha);

        void setCommRank(int commRank);
        void setCommSize(int commSize);


        void insert(const Image& src,
                    const Coordinate5D coord,
                    const double u,
                    const double v);

        void allReduceW(MPI_Comm workers);

        void reduceF(int root,
                     MPI_Comm world);

        void getF(Volume& dst);

        void constructor(const char dst[]);

        void display(const int rank,
                     const char name[]);
};

#endif //RECONSTRUCTOR_H


