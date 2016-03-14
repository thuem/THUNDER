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

    _particles.set_size(N, _PARTICLEDIM);

    for (int i = 0; i < _N; i++)
    {
        gsl_ran_dir_3d(RANDR, 
                       &_particles(i, _EX), 
                       &_particles(i, _EY), 
                       &_particles(i, _EZ));
        
        _particles(i, _PSI) = gsl_ran_flat(RANDR, 0, M_PI);
        
        _particles(i, _X) = gsl_ran_flat(RANDR, -_maxX, _maxX); 
        _particles(i, _Y) = gsl_ran_flat(RANDR, -_maxY, _maxY);
                
        _particles(i, _W) = 1.0 / _N;
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
    angle(dst.phi, dst.theta, vec3({_particles(i, _EX),
                                    _particles(i, _EY),
                                    _particles(i, _EZ)}));

    dst.psi = _particles(i, _PSI);
    dst.x = _particles(i, _X);
    dst.y = _particles(i, _Y);
}

void Particle::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Particle::perturb()
{
    mat L = chol(cov(_particles.cols(0, _PARTICLEDIM - 1)), "lower");

    mat delta(_N, _PARTICLEDIM - 1);

    for (int i = 0; i < _N; i++)
    {
        delta.row(i) = (L * randu<vec>(_PARTICLEDIM - 1)).t();
        _particles.cols(0, _PARTICLEDIM -1).row(i) += delta.row(i);
    }

}

void Particle::resample()
{
    vec cdf = cumsum(_particles.col(_PARTICLEDIM - 1));

    double u0 = gsl_ran_flat(RANDR, 0, 1.0 / _N);  
    
    int i = 0;
    for (int j = 0; j < _N; j++)
    {
        double uj = u0 + j * 1.0 / _N;

        while (uj > cdf[i])
            i++;
        
        _particles(j, _EX) = _particles(i, _EX);
        _particles(j, _EY) = _particles(i, _EY);
        _particles(j, _EZ) = _particles(i, _EZ);
        _particles(j, _X) = _particles(i, _X);
        _particles(j, _Y) = _particles(i, _Y); 
        _particles(j, _W) = 1.0 / _N;
    }

}

double Particle::neff() const
{
    return 1.0 / gsl_pow_2(norm(_particles.col(_PARTICLEDIM - 1), 2));
}

void Particle::symmetrise()
{
    if (_sym == NULL) return;

    for (int i = 0; i < _N; i++)
        symmetryCounterpart(_particles(i, _EX), 
                            _particles(i, _EY), 
                            _particles(i, _EZ), 
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
