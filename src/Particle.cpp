/*******************************************************************************
 * Author: Hongkun Yu, Mingxu Hu, Kunpeng Wang, Siyuan Ren
 * Dependecy:
 * Test:
 * Execution:
 * Description:
 * ****************************************************************************/

#include "Particle.h"

Particle::Particle()
{
    defaultInit();
}

Particle::Particle(const int mode,
                   const int m,
                   const int n,
                   const double transS,
                   const double transQ,
                   const Symmetry* sym)
{
    defaultInit();

    init(mode, m, n, transS, transQ, sym);
}

Particle::~Particle()
{
    clear();
}

void Particle::init(const int mode,
                    const double transS,
                    const double transQ,
                    const Symmetry* sym)
{
    clear();

    _mode = mode;

    _transS = transS;
    _transQ = transQ;

    _sym = sym;
}

void Particle::init(const int mode,
                    const int m,
                    const int n,
                    const double transS,
                    const double transQ,
                    const Symmetry* sym)
{
    init(mode, transS, transQ, sym);

    _m = m;

    _n = n;

    _c.resize(_n);
    _r.resize(_n, 4);
    _t.resize(_n, 2);
    _d.resize(_n);

    _w.resize(_n);

    reset();
}

void Particle::reset()
{
    gsl_rng* engine = get_random_engine();

    // class, sample from flat distribution
    for (int i = 0; i < _n; i++)
        _c(i) = gsl_rng_uniform_int(engine, _m);

    switch (_mode)
    {
        // rotation, MODE_2D, sample from von Mises Distribution with kappa = 0
        case MODE_2D:
            sampleVMS(_r, vec4(1, 0, 0, 0), 0, _n);
            break;

        // rotation, MODE_3D, sample from Angular Central Gaussian Distribution
        // with identity matrix
        case MODE_3D:
            sampleACG(_r, 1, 1, _n);
            break;

        default:
            CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                      << ": INEXISTENT MODE";
            break;
    }


#ifdef PARTICLE_TRANS_INIT_GAUSSIAN
    // sample from 2D Gaussian Distribution
    for (int i = 0; i < _n; i++)
    {
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &_t(i, 0),
                                   &_t(i, 1));
    }
#endif

#ifdef PARTICLE_TRANS_INIT_FLAT
    // sample for 2D Flat Distribution in a Circle
    for (int i = 0; i < _n; i++)
    {
        double r = gsl_ran_flat(engine, 0, _transS);
        double t = gsl_ran_flat(engine, 0, 2 * M_PI);

        _t(i, 0) = r * cos(t);
        _t(i, 1) = r * sin(t);
    }
#endif

    // the default value of all defocus factor is 1
    for (int i = 0; i < _n; i++)
        _d(i) = 1;

    // initialise weight
    for (int i = 0; i < _n; i++)
        _w(i) = 1.0 / _n;

    if (_mode == MODE_3D) symmetrise();
}

void Particle::reset(const int m,
                     const int n)
{
    _m = m;

    _n = n;

    _c.resize(_n);
    _r.resize(_n, 4);
    _t.resize(_n, 2);
    _d.resize(_n);

    _w.resize(_n);

    reset();
}

void Particle::reset(const int m,
                     const int nR,
                     const int nT)
{
    gsl_rng* engine = get_random_engine();

    _m = m;

    _n = m * nR * nT;

    _c.resize(_n);
    _r.resize(_n, 4);
    _t.resize(_n, 2);
    _d.resize(_n);

    _w.resize(_n);

    uvec c(m);

    // sample from 0 to (m - 1)
    for (int i = 0; i < m; i++)
        c(i) = i;

    mat4 r(nR, 4);

    switch (_mode)
    {
        case MODE_2D:
            // sample from von Mises Distribution with kappa = 0
            sampleVMS(r, vec4(1, 0, 0, 0), 0, nR);
            break;

        case MODE_3D:
            // sample from Angular Central Gaussian Distribution with identity matrix
            sampleACG(r, 1, 1, nR);
            break;

        default:
            CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                      << ": INEXISTENT MODE";
            break;
    }

    mat2 t(nT, 2);

#ifdef PARTICLE_TRANS_INIT_GAUSSIAN
    // sample from 2D Gaussian Distribution
    for (int i = 0; i < nT; i++)
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &t(i, 0),
                                   &t(i, 1));
#endif

#ifdef PARTICLE_TRANS_INIT_FLAT
    // sample for 2D Flat Distribution in a Circle
    for (int i = 0; i < nT; i++)
    {
        double r = gsl_ran_flat(engine, 0, _transS);
        double t = gsl_ran_flat(engine, 0, 2 * M_PI);

        t(i, 0) = r * cos(t);
        t(i, 1) = r * sin(t);
    }
#endif

    for (int k = 0; k < m; k++)
        for (int j = 0; j < nR; j++)
            for (int i = 0; i < nT; i++)
            {
                _c(k * nR * nT + j * nT + i) = c(k);

                _r.row(k * nR * nT + j * nT + i) = r.row(j);

                _t.row(k * nR * nT + j * nT + i) = t.row(i);

                _d(k * nR * nT + j * nT + i) = 1;

                _w(k * nR * nT + j * nT + i) = 1.0 / _n;
            }

    if (_mode == MODE_3D) symmetrise();
}

int Particle::mode() const { return _mode; }

void Particle::setMode(const int mode) { _mode = mode; }

int Particle::m() const { return _m; }

void Particle::setM(const int m) { _m = m; }

int Particle::n() const { return _n; }

void Particle::setN(const int n) { _n = n; }

double Particle::transS() const { return _transS; }

void Particle::setTransS(const double transS) { _transS = transS; }

double Particle::transQ() const { return _transQ; }

void Particle::setTransQ(const double transQ) { _transQ = transQ; }

uvec Particle::c() const { return _c; }

void Particle::setC(const uvec& c) { _c = c; }

mat4 Particle::r() const { return _r; }

void Particle::setR(const mat4& r) { _r = r; }

mat2 Particle::t() const { return _t; }

void Particle::setT(const mat2& t) { _t = t; }

vec Particle::d() const { return _d; }

void Particle::setD(const vec& d) { _d = d; }

vec Particle::w() const { return _w; }

void Particle::setW(const vec& w) { _w = w; }

const Symmetry* Particle::symmetry() const { return _sym; }

void Particle::setSymmetry(const Symmetry* sym) { _sym = sym; }

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
    switch (_mode)
    {
        case MODE_2D:
            rVari = 1.0 / (1 + _k); // TODO: it is a approximation
            break;

        case MODE_3D:
            /***
            if (_k0 == 0) CLOG(FATAL, "LOGGER_SYS") << "k0 = 0";
            if (gsl_isnan(_k0)) CLOG(FATAL, "LOGGER_SYS") << "k0 NAN";
            if (gsl_isnan(_k1)) CLOG(FATAL, "LOGGER_SYS") << "k1 NAN";
            ***/

            rVari = sqrt(_k1) / sqrt(_k0);

            break;

        default:
            CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                      << ": INEXISTENT MODE";
            break;
    }

    s0 = _s0;
    s1 = _s1;
}

double Particle::compressTrans() const
{
    return _s0 * _s1 / gsl_pow_2(_transS);
}

double Particle::compressPerDim() const
{
    double cmp = compress();

    switch (_mode)
    {
        case MODE_2D:
            return pow(cmp, 1.0 / 3);

        case MODE_3D:
            return pow(cmp, 1.0 / 5);

        default:
            REPORT_ERROR("INEXISTENT MODE");
            abort();
    }
}

double Particle::compress() const
{
    double rVari, s0, s1;

    vari(rVari, s0, s1);

    switch (_mode)
    {
        case MODE_2D:
            return rVari * s0 * s1 / gsl_pow_2(_transS);
            
        case MODE_3D:
            return gsl_pow_3(rVari) * s0 * s1 / gsl_pow_2(_transS);

        default:
            REPORT_ERROR("INEXISTENT MODE");
            abort();
    }
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

void Particle::c(int& dst,
                 const int i) const
{
    dst = _c(i);
}

void Particle::setC(const int src,
                    const int i)
{
    _c(i) = src;
}

void Particle::rot(mat22& dst,
                   const int i) const
{
    rotate2D(dst, vec2(_r(i, 0), _r(i, 1)));
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

void Particle::calVari()
{
    // variance of translation

#ifdef PARTICLE_CAL_VARI_TRANS_ZERO_MEAN
    _s0 = gsl_stats_sd_m(_t.col(0).data(), 1, _t.rows(), 0);
    _s1 = gsl_stats_sd_m(_t.col(1).data(), 1, _t.rows(), 0);
#else
    _s0 = gsl_stats_sd(_t.col(0).data(), 1, _t.rows());
    _s1 = gsl_stats_sd(_t.col(1).data(), 1, _t.rows());
#endif

    _rho = 0;

    // variance of rotation

    if (_mode == MODE_2D)
        inferVMS(_k, _r);
    else if (_mode == MODE_3D)
        inferACG(_k0, _k1, _r);
    else
        REPORT_ERROR("INEXISTENT MODE");

    // variance of defocus factor

    _s = gsl_stats_sd(_d.data(), 1, _d.size());
}

void Particle::perturb(const double pf)
{
    calVari();

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Translation Perturbation";
#endif

    gsl_rng* engine = get_random_engine();

    for (int i = 0; i < _t.rows(); i++)
    {
        double x, y;
        gsl_ran_bivariate_gaussian(engine, _s0, _s1, _rho, &x, &y);
        _t(i, 0) += x * pf;
        _t(i, 1) += y * pf;
        /***
        _t(i, 0) += x * sqrt(pf);
        _t(i, 1) += y * sqrt(pf);
        ***/
    }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Rotation Perturbation";
#endif

    mat4 d(_n, 4);

    switch (_mode)
    {
        case MODE_2D:
            // for more sparse, pf > 1
            // for more dense, 0 < pf < 1
            sampleVMS(d, vec4(1, 0, 0, 0), _k / pf, _n);
            break;

        case MODE_3D:
            //sampleACG(d, pow(pf, -2.0 / 3) * _k0, _k1, _n);
            sampleACG(d, pow(pf, -2.0) * _k0, _k1, _n);
            break;

        default:
            CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                      << ": INEXISTENT MODE";
            break;
    }

    for (int i = 0; i < _n; i++)
    {
        vec4 quat = _r.row(i).transpose();
        vec4 pert = d.row(i).transpose();
        quaternion_mul(quat, quat, pert);
        _r.row(i) = quat.transpose();
    }

    if (_mode == MODE_3D) symmetrise();

    for (int i = 0; i < _d.size(); i++)
        _d(i) += gsl_ran_gaussian(engine, _s) * pf;

    reCentre();
}

void Particle::resample(const double alpha)
{
    resample(_n, alpha);
}

void Particle::resample(const int n,
                        const double alpha)
{
#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Recording the Current Most Likely Coordinate";
#endif

    uvec rank = iSort();

    c(_topC, rank(0));
    quaternion(_topR, rank(0));
    t(_topT, rank(0));

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Performing Resampling";
#endif

    vec cdf = cumsum(_w);

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Recording New Number of Sampling Points";
#endif

    _n = n;
    _w.resize(n);

    // number of global sampling points
    int nG = AROUND(alpha * n);

    // number of local sampling points
    int nL = n - nG;

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Allocating Temporary Storage";
#endif

    uvec c(n);
    mat4 r(n, 4);
    mat2 t(n, 2);
    vec d(n);
    
#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points";
#endif

    gsl_rng* engine = get_random_engine();

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points for Class";
#endif

    for (int i = 0; i < nG; i++)
        c(i) = gsl_rng_uniform_int(engine, _m);

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points for Rotation";
#endif

    switch (_mode)
    {
        case MODE_2D:
            sampleVMS(r, vec4(1, 0, 0, 0), 0, nG);
            break;

        case MODE_3D:
            sampleACG(r, 1, 1, nG);
            break;

        default:
            CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                      << ": INEXISTENT MODE";
    }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points for Translation";
#endif
    
    for (int i = 0; i < nG; i++)
    {
        gsl_ran_bivariate_gaussian(engine,
                                   _transS,
                                   _transS,
                                   0,
                                   &t(i, 0),
                                   &t(i, 1));
                
    }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Global Sampling Points for Defocus Factor";
#endif

    for (int i = 0; i < nG; i++)
        _d(i) = 1;

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Weights for Global Sampling Points";
#endif

    for (int i = 0; i < nG; i++)
        _w(i) = 1.0 / n;

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Generating Local Sampling Points";
#endif

    double u0 = gsl_ran_flat(engine, 0, 1.0 / nL);  

    int i = 0;
    for (int j = 0; j < nL; j++)
    {
        double uj = u0 + j * 1.0 / nL;

        while (uj > cdf[i])
            i++;
        
        c(nG + j) = _c(i);

        r.row(nG + j) = _r.row(i);

        t.row(nG + j) = _t.row(i);

        d(nG + j) = _d(i);

        _w(nG + j) = 1.0 / n;
    }

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Recording Results";
#endif

    _c.resize(n);
    _c = c;

    _r.resize(n, 4);
    _r = r;

    _t.resize(n, 2);
    _t = t;

    _d.resize(n);
    _d = d;

#ifdef VERBOSE_LEVEL_4
    CLOG(INFO, "LOGGER_SYS") << "Symmetrizing";
#endif
    
    if (_mode == MODE_3D) symmetrise();
}

void Particle::downSample(const int n,
                          const double alpha)
{
    if (n < _n)
    {
        sort(n);
        shuffle();
    }

    resample(n, alpha);
}

double Particle::neff() const
{
    return 1.0 / _w.squaredNorm();
}

void Particle::sort(const int n)
{
    if (n > _n)
        CLOG(FATAL, "LOGGER_SYS") << __FUNCTION__
                                  << ": CANNOT SELECT TOP K FROM N WHEN K > N";

    uvec order = iSort();

    uvec c(n);
    mat4 r(n, 4);
    mat2 t(n, 2);
    vec d(n);
    vec w(n);

    for (int i = 0; i < n; i++)
    {
        c(i) = _c(order(i));
        r.row(i) = _r.row(order(i));
        t.row(i) = _t.row(order(i));
        d(i) = _d(order(i));
        w(i) = _w(order(i));
    }

    _n = n;

    _c = c;
    _r = r;
    _t = t;
    _d = d;
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
    double diff = 1 - fabs(_topRPrev.dot(_topR));

    _topRPrev = _topR;

    return diff;
}

double Particle::diffTopT()
{
    double diff = (_topTPrev - _topT).norm();

    _topTPrev = _topT;

    return diff;
}

void Particle::rank1st(int& cls) const
{
    cls = _topC;
}

void Particle::rank1st(vec4& quat) const
{
    quat = _topR;
}

void Particle::rank1st(mat22& rot) const
{
    vec4 quat;
    rank1st(quat);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rank1st(mat33& rot) const
{
    vec4 quat;
    rank1st(quat);

    rotate3D(rot, quat);
}

void Particle::rank1st(vec2& tran) const
{
    tran = _topT;
}

void Particle::rank1st(int& cls,
                       vec4& quat,
                       vec2& tran) const
{
    cls = _topC;
    quat = _topR;
    tran = _topT;
}

void Particle::rank1st(int& cls,
                       mat22& rot,
                       vec2& tran) const
{
    vec4 quat;
    rank1st(cls, quat, tran);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rank1st(int& cls,
                       mat33& rot,
                       vec2& tran) const
{
    vec4 quat;
    rank1st(cls, quat, tran);

    rotate3D(rot, quat);
}

void Particle::rand(int& cls) const
{
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _n);

    c(cls, u);
}

void Particle::rand(vec4& quat) const
{
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _n);

    quaternion(quat, u);
}

void Particle::rand(mat33& rot) const
{
    vec4 quat;
    rand(quat);

    rotate3D(rot, quat);
}

void Particle::rand(vec2& tran) const
{
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _n);

    t(tran, u);
}

void Particle::rand(int& cls,
                    vec4& quat,
                    vec2& tran) const
{
    gsl_rng* engine = get_random_engine();

    size_t u = gsl_rng_uniform_int(engine, _n);

    c(cls, u);
    quaternion(quat, u);
    t(tran, u);
}

void Particle::rand(int& cls,
                    mat22& rot,
                    vec2& tran) const
{
    vec4 quat;
    rand(cls, quat, tran);

    rotate2D(rot, vec2(quat(0), quat(1)));
}

void Particle::rand(int& cls,
                    mat33& rot,
                    vec2& tran) const
{
    vec4 quat;
    rand(cls, quat, tran);

    rotate3D(rot, quat);
}

void Particle::shuffle()
{
    uvec s = uvec(_n);

    for (int i = 0; i < _n; i++) s(i) = i;

    gsl_rng* engine = get_random_engine();

    gsl_ran_shuffle(engine, s.data(), _n, sizeof(unsigned int));

    uvec c(_n);
    mat4 r(_n, 4);
    mat2 t(_n, 2);
    vec d(_n);
    vec w(_n);

    for (int i = 0; i < _n; i++)
    {
        c(s(i)) = _c(i);
        r.row(s(i)) = _r.row(i);
        t.row(s(i)) = _t.row(i);
        d(s(i)) = _d(i);
        w(s(i)) = _w(i);
    }

    _c = c;
    _r = r;
    _t = t;
    _d = d;
    _w = w;
}

void Particle::copy(Particle& that) const
{
    that.setMode(_mode);
    that.setN(_n);
    that.setTransS(_transS);
    that.setTransQ(_transQ);
    that.setC(_c);
    that.setR(_r);
    that.setT(_t);
    that.setD(_d);
    that.setW(_w);
    that.setSymmetry(_sym);
}

Particle Particle::copy() const
{
    Particle that;

    copy(that);

    return that;
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

    gsl_rng* engine = get_random_engine();

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

void display(const Particle& par)
{
    int c;
    vec4 q;
    vec2 t;
    for (int i = 0; i < par.n(); i++)
    {
        par.c(c, i);
        par.quaternion(q, i);
        par.t(t, i);
        printf("%03d %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf\n",
               c,
               q(0), q(1), q(2), q(3),
               t(0), t(1),
               par.w(i));
    }
}

void save(const char filename[],
          const Particle& par)
{
    FILE* file = fopen(filename, "w");

    int c;
    vec4 q;
    vec2 t;
    for (int i = 0; i < par.n(); i++)
    {
        par.c(c, i);
        par.quaternion(q, i);
        par.t(t, i);
        fprintf(file,
                "%03d %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf %15.9lf\n",
                c,
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

    int c;
    vec4 q;
    vec2 t;
    double w;

    char buf[FILE_LINE_LENGTH];

    int nLine = 0;
    while (fgets(buf, FILE_LINE_LENGTH, file)) nLine++;

    par.reset(1, nLine);

    rewind(file);

    for (int i = 0; i < nLine; i++) 
    {
        fscanf(file,
               "%d %lf %lf %lf %lf %lf %lf %lf",
               &c,
               &q(0), &q(1), &q(2), &q(3),
               &t(0), &t(1),
               &w);

        par.setC(c, i);
        par.setQuaternion(q, i);
        par.setT(t, i);
        par.setW(w, i);
    }

    fclose(file);
}
