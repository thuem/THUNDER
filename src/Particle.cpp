#include <Particle.h>

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
    delete[] _ex;
    delete[] _ey;    
    delete[] _ez;

    delete[] _psi;
    delete[] _x;
    delete[] _y;

    delete[] _w;


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
    _ex  = new double[_N];
    _ey  = new double[_N];
    _ez  = new double[_N];
    
    _psi = new double[_N];
    _x   = new double[_N];
    _y   = new double[_N];
    
    _w   = new double[_N];
    
    
    for (int i = 0; i < _N; i++) 
    {
        gsl_ran_dir_3d(RANDR, &_ex[i], &_ey[i], &_ez[i]);
         
        _psi[i] = gsl_ran_flat(RANDR, 0, M_PI);
        
        _x[i] = gsl_ran_flat(RANDR, -_maxX, _maxX); 
        _y[i] = gsl_ran_flat(RANDR, -_maxY, _maxY);
                
        _w[i] = 1.0 / _N;
   }
}


void Particle::perturb()
{
}




void Particle::resample()
{
    double* cdf = new double[_N];
    partial_sum(_w, _w + _N, cdf);

    int i = 0;
    double u0 = gsl_ran_flat(RANDR, 0, 1.0 / _N);  
    
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

    delete []cdf;
}

double Particle::neff() 
{
    return 1.0 / cblas_ddot(_N, _w, 1, _w, 1);
}

void Particle::symmetrize()
{
    if (_sym == NULL) return;

}

