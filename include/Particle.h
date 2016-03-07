


#ifndef PARTICLE_H
#define PARTICLE_H


#include <armadillo>




using namespace std;


class Particle
{
    private:

        int _N;

        double* _ex = NULL;
        double* _ey = NULL;
        double* _ez = NULL;

        double* _psi = NULL;
        double* _x = NULL;
        double* _y = NULL;


        double* _w = NULL;

    public:

        Particle();

        Particle(const int N);

        ~Particle();

        void init();


};

#endif  //PARTICLE_H
