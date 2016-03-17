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
    if (_r != NULL)
        free_matrix2(_r);
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

    _r = new_matrix2(_N, 4);
    _t.resize(_N, 2);
    _w.resize(_N);

    bingham_t B;
    double v0[4] = {0, 1, 0, 0};
    double v1[4] = {0, 0, 1, 0};
    double v2[4] = {0, 0, 0, 1};

    bingham_new_S3(&B, v0, v1, v2, 0, 0, 0);
    /* uniform bingham distribution */
    bingham_sample(_r, &B, _N);
    /* draw _N samples from it */

    for (int i = 0; i < _N; i++)
    {
        _t(i, 0) = gsl_ran_flat(RANDR, -_maxX, _maxX); 
        _t(i, 1) = gsl_ran_flat(RANDR, -_maxY, _maxY);
                
        _w(i) = 1.0 / _N;
    }

    bingham_free(&B);

    symmetrise();
}

int Particle::N() const
{
    return _N;
}

void Particle::coord(Coordinate5D& dst,
                     const int i) const
{
    angle(dst.phi,
          dst.theta,
          dst.psi,
          vec4({_r[i][0],
                _r[i][1],
                _r[i][2],
                _r[i][3]}));

    dst.x = _t(i, 0);
    dst.y = _t(i, 1);
}

void Particle::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Particle::perturb()
{
    // cout << cov(_t) << endl;
    // translation perturbation
    mat L = chol(cov(_t), "lower");
    for (int i = 0; i < _N; i++)
        _t.row(i) += (L * randu<vec>(2)).t();

    // rotation perturbation
    bingham_t B;
    bingham_fit(&B, _r, _N, 4);
    printf("%f %f %f\n",
           B.Z[0],
           B.Z[1],
           B.Z[2]);
    // TODO: sample form B and perturb

    bingham_free(&B);

    symmetrise();
}

void Particle::resample()
{
    vec cdf = cumsum(_w);

    double u0 = gsl_ran_flat(RANDR, 0, 1.0 / _N);  
    
    double** r = new_matrix2(_N, 4);
    mat t(_N, 2);

    int i = 0;
    for (int j = 0; j < _N; j++)
    {
        double uj = u0 + j * 1.0 / _N;

        while (uj > cdf[i])
            i++;
        
        memcpy(r[j], _r[i], sizeof(double) * 4);
        t.row(j) = _t.row(i);

        _w(j) = 1.0 / _N;
    }

    _t = t;
    for (int i = 0; i < _N; i++)
        memcpy(_r[i], r[i], sizeof(double) * 4);

    free_matrix2(r);
}

double Particle::neff() const
{
    return 1.0 / gsl_pow_2(norm(_w, 2));
}

void Particle::symmetrise()
{
    double phi, theta, psi;
    vec4 quat;
    for (int i = 0; i < _N; i++)
    {
        angle(phi, theta, psi, vec4({_r[i][0],
                                     _r[i][1],
                                     _r[i][2],
                                     _r[i][3]}));

        // make psi in range [0, M_PI)
        if (GSL_IS_ODD(periodic(psi, M_PI)))
        {
            phi *= -1;
            theta *= -1;
        }

        // make phi and theta in the asymetric unit
        symmetryCounterpart(phi, theta, *_sym);

        quaternoin(quat, phi, theta, psi);
        _r[i][0] = quat(0);
        _r[i][1] = quat(1);
        _r[i][2] = quat(2);
        _r[i][3] = quat(3);
    }
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
