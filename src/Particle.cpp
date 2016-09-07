/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang, Siyuan Ren
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

Particle::Particle() {}

Particle::Particle(const int n,
                   const double transS,
                   const double transQ,
                   const Symmetry* sym)
{    
    init(n, transS, transQ, sym);
}

Particle::~Particle()
{
    clear();
}

void Particle::init(const double transS,
                    const double transQ,
                    const Symmetry* sym)
{
    clear();

    _transS = transS;
    _transQ = transQ;

    _sym = sym;
}

void Particle::init(const int n,
                    const double transS,
                    const double transQ,
                    const Symmetry* sym)
{
    init(transS, transQ, sym);

    _n = n;

    _r.resize(_n, 4);
    _t.resize(_n, 2);

    _w.resize(_n);

    reset();
}

void Particle::reset()
{
    // sample from Angular Central Gaussian Distribution with identity matrix
    sampleACG(_r, 1, 1, _n);

    // sample from 2D Gaussian Distribution
    auto engine = get_random_engine();
    for (int i = 0; i < _n; i++)
    {
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &_t(i, 0),
                                   &_t(i, 1));
                
        _w(i) = 1.0 / _n;
    }

    symmetrise();
}

void Particle::reset(const int n)
{
    _n = n;

    _r.resize(_n, 4);
    _t.resize(_n, 2);
    _w.resize(_n);

    reset();
}

void Particle::reset(const int nR,
                     const int nT)
{
    _n = nR * nT;

    _r.resize(_n, 4);
    _t.resize(_n, 2);
    _w.resize(_n);

    mat4 r(nR, 4);

    // sample from Angular Central Gaussian Distribution with identity matrix
    sampleACG(r, 1, 1, nR);
    
    mat2 t(nT, 2);

    // sample from 2D Gaussian Distribution
    auto engine = get_random_engine();
    for (int i = 0; i < nT; i++)
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &t(i, 0),
                                   &t(i, 1));

    for (int j = 0; j < nR; j++)
        for (int i = 0; i < nT; i++)
        {
            _r.row(j * nT + i) = r.row(j);
            _t.row(j * nT + i) = t.row(i);

            _w(j * nT + i) = 1.0 / _n;
        }
}

int Particle::n() const { return _n; }

void Particle::setN(const int n) { _n = n; }

double Particle::transS() const { return _transS; }

void Particle::setTransS(const double transS) { _transS = transS; }

double Particle::transQ() const { return _transQ; }

void Particle::setTransQ(const double transQ) { _transQ = transQ; }

mat4 Particle::r() const { return _r; }

void Particle::setR(const mat4& r) { _r = r; }

mat2 Particle::t() const { return _t; }

void Particle::setT(const mat2& t) { _t = t; }

vec Particle::w() const { return _w; }

void Particle::setW(const vec& w) { _w = w; }

void Particle::vari(double& k0,
                    double& k1,
                    double& s0,
                    double& s1,
                    double& rho) const
{
    k0 = _k0;
    k1 = _k1;
    s0 = _s0;
    s1 = _s1;
    rho = _rho;
}

void Particle::vari(double& rVari,
                    double& s0,
                    double& s1) const
{
    rVari = sqrt(_k1) / sqrt(_k0);
    s0 = _s0;
    s1 = _s1;
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
    _w /= _w.sum();
}

void Particle::coord(Coordinate5D& dst,
                     const int i) const
{
    vec4 quat = _r.row(i).transpose();
    angle(dst.phi,
          dst.theta,
          dst.psi,
          quat);

    dst.x = _t(i, 0);
    dst.y = _t(i, 1);
}

void Particle::rot(mat33& dst,
                   const int i) const
{
    rotate3D(dst, _r.row(i).transpose());
}

void Particle::t(vec2& dst,
                 const int i) const
{
    dst = _t.row(i).transpose();
}

void Particle::setT(const vec2& src,
                    const int i)
{
    _t.row(i) = src.transpose();
}

void Particle::quaternion(vec4& dst,
                          const int i) const
{
    dst = _r.row(i).transpose();
}

void Particle::setQuaternion(const vec4& src,
                             const int i) 
{
    _r.row(i) = src.transpose();
}

void Particle::setSymmetry(const Symmetry* sym)
{
    _sym = sym;
}

void Particle::calVari()
{
    //TODO: gsl_stats_sd
    _s0 = sqrt(gsl_stats_covariance(_t.col(0).data(),
                                    1,
                                    _t.col(0).data(),
                                    1,
                                    _t.rows()));
    _s1 = sqrt(gsl_stats_covariance(_t.col(1).data(),
                                    1,
                                    _t.col(1).data(),
                                    1,
                                    _t.rows()));
    /***
    _rho = gsl_stats_covariance(_t.colptr(0),
                                1,
                                _t.colptr(1),
                                1,
                                _t.n_rows) / _s0 / _s1;
                                ***/
    _rho = 0;

    inferACG(_k0, _k1, _r);
}

void Particle::perturb(const double pf)
{
    calVari();

    /***
    CLOG(WARNING, "LOGGER_SYS") << "_k0 = " << _k0
                                << ", _k1 = " << _k1
                                << ", _s0 = " << _s0
                                << ", _s1 = " << _s1;
    ***/

    // translation perturbation

    auto engine = get_random_engine();

    for (int i = 0; i < _t.rows(); i++)
    {
        double x, y;
        gsl_ran_bivariate_gaussian(engine, _s0, _s1, _rho, &x, &y);
        //_t(i, 0) += x / sqrt(PERTURB_FACTOR);
        //_t(i, 1) += y / sqrt(PERTURB_FACTOR);
        //_t(i, 0) += x / PERTURB_FACTOR;
        //_t(i, 1) += y / PERTURB_FACTOR;
        _t(i, 0) += x * sqrt(pf);
        _t(i, 1) += y * sqrt(pf);
    }

    // rotation perturbation

    mat4 d(_n, 4);
    //sampleACG(d, pow(PERTURB_FACTOR, 1.0 / 3) * _k0, _k1, _n);
    //sampleACG(d, pow(PERTURB_FACTOR, 2.0 / 3) * _k0, _k1, _n);
    sampleACG(d, pow(pf, -2.0 / 3) * _k0, _k1, _n);

    for (int i = 0; i < _n; i++)
    {
        vec4 quat = _r.row(i).transpose();
        vec4 pert = d.row(i).transpose();
        quaternion_mul(quat, quat, pert);
        _r.row(i) = quat.transpose();
    }

    symmetrise();

    reCentre();
}

void Particle::resample(const double alpha)
{
    resample(_n, alpha);
}

void Particle::resample(const int n,
                        const double alpha)
{
    // record the current most likely coordinate (highest weight)

    uvec rank = iSort();

    quaternion(_topR, rank[0]);
    t(_topT, rank[0]);

    // perform resampling

    vec cdf = cumsum(_w);

    // CLOG(INFO, "LOGGER_SYS") << "Recording New Number of Sampling Points";

    _n = n;
    _w.resize(n);

    // number of global sampling points
    int nG = AROUND(alpha * n);

    // number of local sampling points
    int nL = n - nG;

    // CLOG(INFO, "LOGGER_SYS") << "Allocating Temporary Storage";

    mat4 r(n, 4);
    mat2 t(n, 2);
    
    // CLOG(INFO, "LOGGER_SYS") << "Generate Global Sampling Points";

    sampleACG(r, 1, 1, nG);

    auto engine = get_random_engine();
    
    for (int i = 0; i < nG; i++)
    {
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &t(i, 0),
                                   &t(i, 1));
                
        _w(i) = 1.0 / n;
    }

    // CLOG(INFO, "LOGGER_SYS") << "Generate Local Sampling Points";
    // CLOG(INFO, "LOGGER_SYS") << "nL = " << nL << ", nG = " << nG;

    double u0 = gsl_ran_flat(engine, 0, 1.0 / nL);  

    int i = 0;
    for (int j = 0; j < nL; j++)
    {
        double uj = u0 + j * 1.0 / nL;

        while (uj > cdf[i])
            i++;
        
        r.row(nG + j) = _r.row(i);
        t.row(nG + j) = _t.row(i);

        _w(nG + j) = 1.0 / n;
    }

    // CLOG(INFO, "LOGGER_SYS") << "Recording Results";

    _t.resize(n, 2);
    _t = t;

    _r.resize(n, 4);
    _r = r;
    
    // CLOG(INFO, "LOGGER_SYS") << "Symmetrize";
    
    symmetrise();
}

double Particle::neff() const
{
    return 1.0 / _w.squaredNorm();
}

void Particle::sort(const int n)
{
    if (n > _n)
        CLOG(FATAL, "LOGGER_SYS") << "Can not Select Top K from N when K > N";

    uvec order = iSort();

    mat4 r(n, 4);
    mat2 t(n, 2);
    vec w(n);

    for (int i = 0; i < n; i++)
    {
        r.row(i) = _r.row(order(i));
        t.row(i) = _t.row(order(i));
        w(i) = _w(order(i));
    }

    _n = n;

    _r = r;
    _t = t;
    _w = w;

    // normalise weight again
    normW();
}

uvec Particle::iSort() const
{
    return index_sort_descend(_w);
}

double Particle::diffTopR()
{
    double diff = 1 - abs(_topRPrev.dot(_topR));

    _topRPrev = _topR;

    return diff;
}

double Particle::diffTopT()
{
    double diff = (_topTPrev - _topT).norm();

    _topTPrev = _topT;

    return diff;
}

void Particle::rank1st(vec4& quat,
                       vec2& tran) const
{
    quat = _topR;
    tran = _topT;
    /***
    uvec rank = iSort();

    quaternion(quat, rank[0]);
    t(tran, rank[0]);
    ***/
}

void Particle::rank1st(mat33& rot,
                       vec2& tran) const
{
    vec4 quat;
    rank1st(quat, tran);

    rotate3D(rot, quat);
}

void Particle::rand(vec4& quat,
                    vec2& tran) const
{
    auto engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _n);

    quaternion(quat, u);
    t(tran, u);
}

void Particle::rand(mat33& rot,
                    vec2& tran) const
{
    vec4 quat;
    rand(quat, tran);

    rotate3D(rot, quat);
}

void Particle::shuffle()
{
    uvec s = uvec(_n);

    for (int i = 0; i < _n; i++) s(i) = i;

    auto engine = get_random_engine();

    gsl_ran_shuffle(engine, s.data(), _n, sizeof(unsigned int));

    mat4 r(_n, 4);
    mat2 t(_n, 2);
    vec w(_n);

    for (int i = 0; i < _n; i++)
    {
        r.row(s(i)) = _r.row(i);
        t.row(s(i)) = _t.row(i);
        w(s(i)) = _w(i);
    }

    _r = r;
    _t = t;
    _w = w;
}

void Particle::symmetrise()
{
    double phi, theta, psi;
    vec4 quat;
    for (int i = 0; i < _n; i++)
    {
        vec4 quat = _r.row(i).transpose();
        angle(phi, theta, psi, quat);

        // make phi and theta in the asymetric unit
        if (_sym != NULL) symmetryCounterpart(phi, theta, *_sym);

        quaternoin(quat, phi, theta, psi);
        _r.row(i) = quat.transpose();
    }
}

void Particle::reCentre()
{
    double transM = _transS * gsl_cdf_chisq_Qinv(_transQ, 2);

    //CLOG(INFO, "LOGGER_SYS") << "transM = " << transM;

    auto engine = get_random_engine();

    for (int i = 0; i < _n; i++)
        if (NORM(_t(i, 0), _t(i, 1)) > transM)
        {
            gsl_ran_bivariate_gaussian(engine,
                                       _transS,
                                       _transS,
                                       0,
                                       &_t(i, 0),
                                       &_t(i, 1));
        }
}

void Particle::clear() {}

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
                "%15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf\n",
                q(0), q(1), q(2), q(3),
                t(0), t(1),
                par.w(i));
    }

    fclose(file);
}

void load(Particle& par,
          const char filename[])
{
    FILE* file = fopen(filename, "r");

    vec4 q;
    vec2 t;
    double w;

    char buf[FILE_LINE_LENGTH];

    int nLine = 0;
    while (fgets(buf, FILE_LINE_LENGTH, file)) nLine++;

    par.reset(nLine);

    rewind(file);

    for (int i = 0; i < nLine; i++) 
    {
        fscanf(file,
               "%lf %lf %lf %lf %lf %lf %lf",
               &q(0), &q(1), &q(2), &q(3),
               &t(0), &t(1),
               &w);

        par.setQuaternion(q, i);
        par.setT(t, i);
        par.setW(w, i);
    }

    fclose(file);
}
