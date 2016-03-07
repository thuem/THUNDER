#include <Particle.h>

Particle::Particle() {}

Particle::Particle(const int N) 
{
    _N = N;
    init();
}

Particle::~Particle() {}

void Particle::init() 
{
    _ex  = new double[_N];
    _ey  = new double[_N];
    _ez  = new double[_N];
    
    _psi = new double[_N];
    _x   = new double[_N];
    _y   = new double[_N];
    
    _w   = new double[_N];

}
