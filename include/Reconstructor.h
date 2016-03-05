/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

#include <utility>
#include <vector>

#include <armadillo>
#include <mpi.h>

#include "Coordinate5D.h"
#include "Functions.h"
#include "Euler.h"
#include "FFT.h"
#include "Image.h"
#include "Volume.h"
#include "ImageFunctions.h"
#include "ImageFile.h"

#define DEBUGCONSTRUCTOR

using namespace std;
using namespace arma;

typedef pair<Coordinate5D, double> corWeight;

class Reconstructor
{
    private:

        int _size;

        /***
        int _nCol;
        int _nRow;
        int _nSlc;
        ***/

        double _a = 1.9;
        double _alpha = 15;
        
        int _commRank = 0;
        int _commSize = 1;

        Volume _F;
        Volume _W;
        Volume _C;

        // Volume _WN;

        vector<corWeight> _coordWeight;

        /***
        int _nColImg;
        int _nRowImg;
        ***/

        int _maxRadius;

        int _pf; // padding factor

    public:

        Reconstructor();

        Reconstructor(const int size,
                      const int pf,
                      const double a = 1.9,
                      const double alpha = 15);

        /***
        Reconstructor(const int nCol,
                      const int nRow,
                      const int nSlc,
                      const int nColImg,
                      const int nRowImg,
                      const int pf,
                      const double a,
                      const double alpha);
                      ***/

        // Reconstructor(const Reconstructor& that);

        ~Reconstructor();

        // Reconstructor& operator=(const Reconstructor& that);

        void init(const int size,
                  const int pf,
                  const double a = 1.9,
                  const double alpha = 15);

        /***
        void init(const int nCol,
                  const int nRow,
                  const int nSlc,
                  const int nColImg,
                  const int nRowImg,
                  const int pf,
                  const double a,
                  const double alpha);
                  ***/

        void setCommRank(const int commRank);
        void setCommSize(const int commSize);

        void insert(const Image& src,
                    const Coordinate5D coord,
                    const double u,
                    const double v);

        void allReduceW(MPI_Comm workers);

        void initC();

        void reduceF(int root,
                     MPI_Comm world);

        void getF(Volume& dst);

        void constructor(const char dst[]);

        void display(const int rank,
                     const char name[]);

        double checkC() const;
        /* calculate the distance between C and all 1 */
};

#endif //RECONSTRUCTOR_H


