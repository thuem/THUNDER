/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef PARTICLE_H
#define PARTICLE_H

#include <numeric>
#include <gsl/gsl_math.h>

#include "Typedef.h"
#include "Macro.h"

#include "Coordinate5D.h"
#include "Random.h"
#include "Euler.h"
#include "Symmetry.h"


#define _PARTICLEDIM 7
#define _EX 0
#define _EY 1
#define _EZ 2
#define _PSI 3
#define _X 4
#define _Y 5
#define _W 6

using namespace arma;

using namespace std;

class Particle
{
    private:

        int _N;

        double _maxX;
        double _maxY;

        mat _particles;
        
        const Symmetry* _sym;

    public:

        Particle();

        Particle(const int N,
                 const double maxX,
                 const double maxY,
                 const Symmetry* sym = NULL);

        ~Particle();

        void init(const int N,
                  const double maxX,
                  const double maxY,
                  const Symmetry* sym = NULL);

        int N() const;

        void coord(Coordinate5D& dst,
                   const int i) const;
        /* return the coordinate of the ith particle */

        void setSymmetry(const Symmetry* sym);

        void perturb();

        void resample();

        double neff() const;
    
    private:

        void symmetrise();
};

void display(const Particle& particle);

#endif  //PARTICLE_H
