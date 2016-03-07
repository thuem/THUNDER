


#ifndef PARTICLE_H
#define PARTICLE_H


#include "Random.h"
#include "Euler.h"







using namespace std;


class Particle
{
    private:

        int _N;
        double _maxX;
        double _maxY;

        double* _ex = NULL;
        double* _ey = NULL;
        double* _ez = NULL;

        double* _psi = NULL;
        double* _x = NULL;
        double* _y = NULL;


        double* _w = NULL;

    public:

        Particle();

        Particle(const int N,
                 const double maxX,
                 const double maxY);

        ~Particle();

        void init();

        void perturb();

        void resample();

        double neff();

};

#endif  //PARTICLE_H
