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
#include <functional>

#include <armadillo>
#include <mpi.h>

#include "Parallel.h"
#include "Coordinate5D.h"
#include "Functions.h"
#include "Euler.h"
#include "FFT.h"
#include "Image.h"
#include "Volume.h"
#include "ImageFunctions.h"
#include "Symmetry.h"
#include "Transformation.h"
#include "TabFunction.h"

using namespace std;
using namespace arma;
using namespace placeholders;

#define PAD_SIZE (_pf * _size)

class Reconstructor : public Parallel
{
    private:

        int _size;

        Volume _F;
        Volume _W;
        Volume _C;

        vector<Coordinate5D> _coord;

        int _maxRadius;

        int _pf = 2; // padding factor

        const Symmetry* _sym = NULL;

        double _a = 1.9;
        double _alpha = 10;
        
        double _zeta = 0.15;

        TabFunction _kernel;

    public:

        Reconstructor();

        Reconstructor(const int size,
                      const int pf = 2,
                      const Symmetry* sym = NULL,
                      const double a = 1.9,
                      const double alpha = 10,
                      const double zeta = 0.15);

        ~Reconstructor();

        void init(const int size,
                  const int pf = 2,
                  const Symmetry* sym = NULL,
                  const double a = 1.9,
                  const double alpha = 10,
                  const double zeta = 0.15);

        void setSymmetry(const Symmetry* sym);

        void insert(const Image& src,
                    const Coordinate5D coord,
                    const double w);

        // void insertCoord(const Coordinate5D coord,
        //                  const double w);

        void reconstruct(Volume& dst);

    private:

        void allReduceW();

        void allReduceF();

        void symmetrizeF();

        void symmetrizeC();

        double checkC() const;
        /* calculate the distance between C and 1 */
};

#endif //RECONSTRUCTOR_H
