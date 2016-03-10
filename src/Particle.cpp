/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

Particle::Particle() {}

Particle::Particle(const int N,
                   const double maxX,
                   const double maxY,
                   const Symmetry* sym)
{    
    init(N, maxX, maxY, sym);
}

Particle::~Particle() 
{
    SAVE_DELETE(_ex);
    SAVE_DELETE(_ey);
    SAVE_DELETE(_ez);

    SAVE_DELETE(_psi);

    SAVE_DELETE(_x);
    SAVE_DELETE(_y);

    SAVE_DELETE(_w);
}

void Particle::init(const int N,
                    const double maxX,
                    const double maxY,
                    const Symmetry* sym)
{
    _N = N;

    _maxX = maxX;
    _maxY = maxY;

    _sym = sym;

    _ex = new double[_N];
    _ey = new double[_N];
    _ez = new double[_N];
    
    _psi = new double[_N];

    _x = new double[_N];
    _y = new double[_N];
    
    _w = new double[_N];
    
    int i = 0;
    while (i < _N) 
    {
        gsl_ran_dir_3d(RANDR, &_ex[i], &_ey[i], &_ez[i]);
     
        // vec3 src = {_ex[i], _ey[i], _ez[i]};
        /***
        double phi, theta;
        angle(phi, theta, src);
        ***/

        if (!asymmetryUnit(vec3({_ex[i], _ey[i], _ez[i]}), *sym))
            continue;
         
        _psi[i] = gsl_ran_flat(RANDR, 0, M_PI);
        
        _x[i] = gsl_ran_flat(RANDR, -_maxX, _maxX); 
        _y[i] = gsl_ran_flat(RANDR, -_maxY, _maxY);
                
        _w[i] = 1.0 / _N;

        i++;
   }
}

int Particle::N() const
{
    return _N;
}

void Particle::coord(Coordinate5D& dst,
                     const int i) const
{
    // vec3 src = {_ex[i], _ey[i], _ez[i]};
    angle(dst.phi, dst.theta, vec3({_ex[i], _ey[i], _ez[i]}));

    dst.psi = _psi[i];
    dst.x = _x[i];
    dst.y = _y[i];
}

void Particle::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Particle::perturb()
{
}

void Particle::resample()
{
    double* cdf = new double[_N];
    partial_sum(_w, _w + _N, cdf);

    double u0 = gsl_ran_flat(RANDR, 0, 1.0 / _N);  
    
    int i = 0;
    for (int j = 0; j < _N; j++)
    {
        double uj = u0 + j * 1.0 / _N;

        while (uj > cdf[i])
            i++;
        
        _ex[j] = _ex[i];
        _ey[j] = _ey[i];
        _ez[j] = _ez[i];
        
        _psi[j] = _psi[i];

        _x[j] = _x[i];
        _y[j] = _y[i];

        _w[j] = 1.0 / _N;
    }

    delete[] cdf;
}

double Particle::neff() const
{
    return 1.0 / cblas_ddot(_N, _w, 1, _w, 1);
}
