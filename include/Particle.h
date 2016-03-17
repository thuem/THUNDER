/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#ifndef PARTICLE_H
#define PARTICLE_H

#include <iostream>
#include <numeric>

#include <gsl/gsl_math.h>
#include <armadillo>
#include <bingham.h>

#include "Typedef.h"
#include "Macro.h"

#include "Coordinate5D.h"
#include "Random.h"
#include "Euler.h"
#include "Functions.h"
#include "Symmetry.h"

using namespace arma;

using namespace std;

class Particle
{
    private:

        int _N;

        double _maxX;
        double _maxY;

        double** _r = NULL;
        /* rotation, quaternion */

        mat _t; // translation

        vec _w; // weight
        
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
