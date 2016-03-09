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
#include "Symmetry.h"
#include "Transformation.h"

using namespace std;
using namespace arma;

class Reconstructor
{
    private:

        int _size;

        double _a = 1.9;
        double _alpha = 15;
        
        int _commRank = 0;
        int _commSize = 1;

        Volume _F;
        Volume _W;
        Volume _C;

        vector<Coordinate5D> _coord;

        int _maxRadius;

        int _pf; // padding factor

        const Symmetry* _sym;

    public:

        Reconstructor();

        Reconstructor(const int size,
                      const int pf,
                      const Symmetry* sym = NULL,
                      const double a = 1.9,
                      const double alpha = 15);

        ~Reconstructor();

        void init(const int size,
                  const int pf,
                  const Symmetry* sym = NULL,
                  const double a = 1.9,
                  const double alpha = 15);

        void setSymmetry(const Symmetry* sym);

        void setCommRank(const int commRank);
        void setCommSize(const int commSize);

        void insert(const Image& src,
                    const Coordinate5D coord,
                    const double u,
                    const double v);

        void allReduceW(MPI_Comm workers);

        void allReduceF(MPI_Comm world);

        void constructor(const char dst[]);

        double checkC() const;
        /* calculate the distance between C and 1 */

    private:

        void symmetrizeF();

        void symmetrizeC();
};

#endif //RECONSTRUCTOR_H


