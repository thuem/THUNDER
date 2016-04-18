/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

Particle::Particle() {}

Particle::Particle(const int n,
                   const double maxX,
                   const double maxY,
                   const Symmetry* sym)
{    
    init(n, maxX, maxY, sym);
}

Particle::~Particle()
{
    clear();
}

void Particle::init(const int n,
                    const double maxX,
                    const double maxY,
                    const Symmetry* sym)
{
    clear();

    _n = n;

    _maxX = maxX;
    _maxY = maxY;

    _sym = sym;

    _r = new_matrix2(_n, 4);
    _t.resize(_n, 2);
    _w.resize(_n);

    reset();
}

void Particle::reset()
{
    bingham_t B;
    bingham_new_S3(&B, e0, e1, e2, 0, 0, 0);
    // uniform bingham distribution
    bingham_sample(_r, &B, _n);
    // draw _n samples from it

    for (int i = 0; i < _n; i++)
    {
        _t(i, 0) = gsl_ran_flat(RANDR, -_maxX, _maxX); 
        _t(i, 1) = gsl_ran_flat(RANDR, -_maxY, _maxY);
                
        _w(i) = 1.0 / _n;
    }

    bingham_free(&B);

    symmetrise();
}

int Particle::n() const { return _n; }

void Particle::vari(double& k0,
                    double& k1,
                    double& k2,
                    double& s0,
                    double& s1,
                    double& rho) const
{
    k0 = _k0;
    k1 = _k1;
    k2 = _k2;
    s0 = _s0;
    s1 = _s1;
    rho = _rho;
}

double Particle::w(const int i) const { return _w(i); }

void Particle::setW(const double w,
                    const int i)
{
    _w(i) = w;
}

void Particle::mulW(const double w,
                    const int i)
{
    _w(i) *= w;
}

void Particle::normW()
{
    _w /= sum(_w);
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

void Particle::rot(mat33& dst,
                   const int i) const
{
    rotate3D(dst, vec4({_r[i][0],
                        _r[i][1],
                        _r[i][2],
                        _r[i][3]}));
}

void Particle::t(vec2& dst,
                 const int i) const
{
    dst(0) = _t(i, 0);
    dst(1) = _t(i, 1);
}

void Particle::quaternion(vec4& dst,
                          const int i) const
{
    dst(0) = _r[i][0];
    dst(1) = _r[i][1];
    dst(2) = _r[i][2];
    dst(3) = _r[i][3];
}

void Particle::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Particle::calVari()
{
    _s0 = sqrt(gsl_stats_covariance(_t.colptr(0),
                                    1,
                                    _t.colptr(0),
                                    1,
                                    _t.n_rows));
    _s1 = sqrt(gsl_stats_covariance(_t.colptr(1),
                                    1,
                                    _t.colptr(1),
                                    1,
                                    _t.n_rows));
    /***
    _rho = gsl_stats_covariance(_t.colptr(0),
                                1,
                                _t.colptr(1),
                                1,
                                _t.n_rows) / _s0 / _s1;
                                ***/
    _rho = 0;

    bingham_t B;
    bingham_fit(&B, _r, _n, 4);

    _k0 = B.Z[0];
    _k1 = B.Z[1];
    _k2 = B.Z[2];

    bingham_free(&B);
}

void Particle::perturb()
{
    calVari();

    _t.each_row([this](rowvec& row)
                {
                    double x, y;
                    gsl_ran_bivariate_gaussian(RANDR, _s0, _s1, _rho, &x, &y);
                    row(0) += x / 5;
                    row(1) += y / 5;
                });

    /***
    mat L = chol(cov(_t), "lower");
    for (int i = 0; i < _n; i++)
        _t.row(i) += (L * randn<vec>(2)).t() / 3;
        ***/

    // rotation perturbation

    bingham_t B;
    // bingham_new_S3(&B, e0, e1, e2, _k0, _k1, _k2);
    bingham_new_S3(&B, e0, e1, e2, 5 * _k0, 5 * _k1, 5 * _k2);
    /***
    // bingham_new_S3(&B, e0, e1, e2, _k0, _k1, _k2);
    // bingham_new_S3(&B, e0, e1, e2, -1, -1, -1);
    bingham_new_S3(&B, e0, e1, e2, -300, -200, -100);
    ***/
    double** d = new_matrix2(_n, 4);
    bingham_sample(d, &B, _n);

    for (int i = 0; i < _n; i++)
    {
        /***
        printf("d: %10f %10f %10f %10f\n", d[i][0], d[i][1], d[i][2], d[i][3]);
        printf("r: %10f %10f %10f %10f\n", _r[i][0], _r[i][1], _r[i][2], _r[i][3]);
        ***/
        quaternion_mul(_r[i], _r[i], d[i]);
        // printf("r: %10f %10f %10f %10f\n\n", _r[i][0], _r[i][1], _r[i][2], _r[i][3]);
    }

    bingham_free(&B);
    free_matrix2(d);

    symmetrise();
}

void Particle::resample(const double alpha)
{
    resample(_n, alpha);
}

void Particle::resample(const int n,
                        const double alpha)
{
    vec cdf = cumsum(_w);

    // DLOG(INFO) << "Recording New Number of Sampling Points";
    _n = n;
    _w.set_size(n);

    // number of global sampling points
    int nG = AROUND(alpha * n);

    // number of local sampling points
    int nL = n - nG;

    // DLOG(INFO) << "Allocating Temporary Storage";

    double** r = new_matrix2(n, 4);
    mat t(n, 2);
    
    // DLOG(INFO) << "Generate Global Sampling Points";

    bingham_t B;
    bingham_new_S3(&B, e0, e1, e2, 0, 0, 0);
    bingham_sample(r, &B, nG);
    bingham_free(&B);
    
    for (int i = 0; i < nG; i++)
    {
        t(i, 0) = gsl_ran_flat(RANDR, -_maxX, _maxX); 
        t(i, 1) = gsl_ran_flat(RANDR, -_maxY, _maxY);
                
        _w(i) = 1.0 / n;
    }

    // DLOG(INFO) << "Generate Local Sampling Points";

    double u0 = gsl_ran_flat(RANDR, 0, 1.0 / nL);  

    int i = 0;
    for (int j = 0; j < nL; j++)
    {
        double uj = u0 + j * 1.0 / nL;

        while (uj > cdf[i])
            i++;
        
        memcpy(r[nG + j], _r[i], sizeof(double) * 4);
        t.row(nG + j) = _t.row(i);

        _w(nG + j) = 1.0 / n;
    }

    // DLOG(INFO) << "Recording Results";

    _t.set_size(n, 2);
    _t = t;

    free_matrix2(_r);
    _r = new_matrix2(n, 4);
    
    for (int i = 0; i < n; i++)
        memcpy(_r[i], r[i], sizeof(double) * 4);

    free_matrix2(r);
}

double Particle::neff() const
{
    return 1.0 / gsl_pow_2(norm(_w, 2));
}

uvec Particle::iSort() const
{
    return sort_index(_w, "descend");
}

void Particle::symmetrise()
{
    double phi, theta, psi;
    vec4 quat;
    for (int i = 0; i < _n; i++)
    {
        angle(phi, theta, psi, vec4({_r[i][0],
                                     _r[i][1],
                                     _r[i][2],
                                     _r[i][3]}));

        /***
        // make psi in range [0, M_PI)
        if (GSL_IS_ODD(periodic(psi, M_PI)))
        {
            phi *= -1;
            theta *= -1;
        }
        ***/

        // make phi and theta in the asymetric unit
        if (_sym != NULL) symmetryCounterpart(phi, theta, *_sym);

        quaternoin(quat, phi, theta, psi);
        _r[i][0] = quat(0);
        _r[i][1] = quat(1);
        _r[i][2] = quat(2);
        _r[i][3] = quat(3);
    }
}

void Particle::clear()
{
    if (_r != NULL)
    {
        free_matrix2(_r);
        _r = NULL;
    }
}

void display(const Particle& particle)
{
    Coordinate5D coord;
    for (int i = 0; i < particle.n(); i++)
    {
        particle.coord(coord, i);
        display(coord);
    }
}

void save(const char filename[],
          const Particle& par)
{
    FILE* file = fopen(filename, "w");

    vec4 q;
    vec2 t;
    for (int i = 0; i < par.n(); i++)
    {
        par.quaternion(q, i);
        par.t(t, i);
        fprintf(file,
                "%10f %10f %10f %10f %10f %10f %10f\n",
                 q(0),q(1),q(2),q(3),
                 t(0), t(1),
                 par.w(i));
    }

    fclose(file);
}
