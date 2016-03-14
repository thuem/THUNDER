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

Particle::~Particle() {}

void Particle::init(const int N,
                    const double maxX,
                    const double maxY,
                    const Symmetry* sym)
{
    _N = N;

    _maxX = maxX;
    _maxY = maxY;

    _sym = sym;

    _c.resize(_N, _DIM);
    _w.resize(_N);

    for (int i = 0; i < _N; i++)
    {
        gsl_ran_dir_3d(RANDR, 
                       &_c(i, _EX), 
                       &_c(i, _EY), 
                       &_c(i, _EZ));
        
        _c(i, _PSI) = gsl_ran_flat(RANDR, 0, M_PI);
        
        _c(i, _X) = gsl_ran_flat(RANDR, -_maxX, _maxX); 
        _c(i, _Y) = gsl_ran_flat(RANDR, -_maxY, _maxY);
                
        _w(i) = 1.0 / _N;
    }

    symmetrise();
}

int Particle::N() const
{
    return _N;
}

void Particle::coord(Coordinate5D& dst,
                     const int i) const
{
    // vec3 src = {_ex[i], _ey[i], _ez[i]};
    angle(dst.phi, dst.theta, vec3({_c(i, _EX),
                                    _c(i, _EY),
                                    _c(i, _EZ)}));

    dst.psi = _c(i, _PSI);
    dst.x = _c(i, _X);
    dst.y = _c(i, _Y);
}

void Particle::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Particle::perturb()
{
    cout << cov(_c) << endl;
    mat L = chol(cov(_c), "lower");

    mat d(_N, _DIM);

    for (int i = 0; i < _N; i++)
        d.row(i) = (L * randu<vec>(_DIM)).t();

    _c += d;

    for (int i = 0; i < _N; i++)
    {
        _c.row(i).head(3) /= sum(_c.row(i).head(3));
        if (GSL_IS_ODD(periodic(_c.row(i)(_PSI), M_PI)))
            _c.row(i).head(3) *= -1;
    }

    symmetrise();
}

void Particle::resample()
{
    vec cdf = cumsum(_w);
    // vec cdf = cumsum(_c.col(_PARTICLEDIM - 1));

    double u0 = gsl_ran_flat(RANDR, 0, 1.0 / _N);  
    
    int i = 0;
    for (int j = 0; j < _N; j++)
    {
        double uj = u0 + j * 1.0 / _N;

        while (uj > cdf[i])
            i++;
        
        _c.row(j) = _c.row(i);
        /***
        _c(j, _EX) = _c(i, _EX);
        _c(j, _EY) = _c(i, _EY);
        _c(j, _EZ) = _c(i, _EZ);
        _c(j, _X) = _c(i, _X);
        _c(j, _Y) = _c(i, _Y); 
        ***/

        _w(j) = 1.0 / _N;
    }
}

double Particle::neff() const
{
    return 1.0 / gsl_pow_2(norm(_w, 2));
}

void Particle::symmetrise()
{
    if (_sym == NULL) return;

    for (int i = 0; i < _N; i++)
        symmetryCounterpart(_c(i, _EX), 
                            _c(i, _EY), 
                            _c(i, _EZ), 
                            *_sym);
}

void display(const Particle& particle)
{
    Coordinate5D coord;
    for (int i = 0; i < particle.N(); i++)
    {
        particle.coord(coord, i);
        display(coord);
    }
}
